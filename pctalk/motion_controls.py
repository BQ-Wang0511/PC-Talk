"""LAC and EMC modules operating on LivePortrait motion templates."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from pykalman import KalmanFilter

from .models.emotion_motion_predictor import EmotionMotionPredictor
from .models.lip_motion_predictor import (
    LipMotionPredictor,
    LipRefinementNetwork,
    SpeakingStyleEncoder,
)

from .config import (
    EMCConfig,
    EmotionCondition,
    LACConfig,
    LipArticulationEdit,
)


LIP_INDICES: Tuple[int, ...] = (6, 12, 14, 17, 19, 20)
EYE_INDICES: Tuple[int, ...] = (11, 13, 15, 16, 18)
BROW_INDICES: Tuple[int, ...] = (0, 1, 2)
EMOTIONS: Dict[str, int] = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fear": 4,
    "disgusted": 5,
    "surprised": 6,
    "contempt": 7,
}
# Sparse semantic directions used by the original PC-Talk research scripts.
# They live in LivePortrait expression coordinates.  LAC only predicts the six
# lip-related keypoints, so directions are intentionally restricted to them.
ARTICULATION_VECTORS = {
    "pursing": (((14, 1), 0.001), ((17, 2), -0.0005)),
    "widening": (((20, 2), -0.001), ((20, 1), -0.001), ((14, 1), -0.001)),
    "opening": (((19, 1), 0.001), ((19, 2), 0.0001), ((17, 1), -0.0001)),
}


def _checkpoint_value(checkpoint_args, name: str, default=None):
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name, default)
    return getattr(checkpoint_args, name, default)


def _load_checkpoint(path: str):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")
    try:
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(str(checkpoint), map_location="cpu")
    model_state = state.get("model", state) if isinstance(state, dict) else state
    checkpoint_args = state.get("args", {}) if isinstance(state, dict) else {}
    if not isinstance(model_state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")
    model_state = {
        key.removeprefix("module."): value for key, value in model_state.items()
    }
    return state, model_state, checkpoint_args


def _submodule_state(model_state, prefix: str):
    prefix = prefix.rstrip(".") + "."
    selected = {
        key[len(prefix) :]: value
        for key, value in model_state.items()
        if key.startswith(prefix)
    }
    return selected or model_state


def _infer_person_num(model_state, checkpoint_args, configured: Optional[int]) -> int:
    if configured is not None:
        return configured
    from_checkpoint = _checkpoint_value(checkpoint_args, "person_num")
    if from_checkpoint:
        return int(from_checkpoint)
    for key, value in model_state.items():
        if key.endswith("obj_vector.weight") and value.ndim == 2:
            return int(value.shape[1])
    raise ValueError(
        "person_num is missing from both LACConfig and the checkpoint metadata"
    )


def _infer_window_size(
    model_state,
    checkpoint_args,
    configured: Optional[int],
    default: int = 25,
) -> int:
    if configured is not None:
        return int(configured)
    from_checkpoint = _checkpoint_value(checkpoint_args, "T")
    if from_checkpoint:
        return int(from_checkpoint)

    # Older research checkpoints did not serialize their argparse Namespace.
    # TicPositionalEncoding stores T indirectly in PPE.pe's repeated length.
    positional = model_state.get("PPE.pe")
    if positional is not None:
        encoded_length = int(positional.shape[1])
        for candidate in (5, 10, 20, 25, 50, 100):
            if candidate * (600 // candidate + 1) == encoded_length:
                return candidate
    return default


def _frame_array(frame: dict, key: str) -> np.ndarray:
    return np.asarray(frame[key], dtype=np.float32)


def _scale_value(frame: dict) -> float:
    return float(_frame_array(frame, "scale").reshape(-1)[0])


def render_keypoints(frame: dict, expression=None) -> np.ndarray:
    """Apply LivePortrait's keypoint transform without running the renderer."""

    kp = _frame_array(frame, "kp").reshape(1, 21, 3)
    rotation_key = "R" if "R" in frame else "R_d"
    rotation = _frame_array(frame, rotation_key).reshape(1, 3, 3)
    if expression is None:
        expression = _frame_array(frame, "exp").reshape(1, 21, 3)
    else:
        expression = np.asarray(expression, dtype=np.float32).reshape(1, 21, 3)
    transformed = (kp @ rotation + expression) * _scale_value(frame)
    translation = _frame_array(frame, "t").reshape(1, 1, 3).copy()
    translation[..., 2] = 0
    transformed[..., :2] += translation[..., :2]
    return transformed.astype(np.float32)


def undeformed_keypoints(frame: dict) -> np.ndarray:
    return render_keypoints(frame, np.zeros((1, 21, 3), dtype=np.float32))


def _smooth(sequence: np.ndarray, observation_variance: float = 1e-6) -> np.ndarray:
    if len(sequence) < 3:
        return sequence
    shape = sequence.shape
    flat = sequence.reshape(len(sequence), -1)
    kalman = KalmanFilter(
        initial_state_mean=flat[0],
        n_dim_obs=flat.shape[1],
        transition_covariance=1e-5 * np.eye(flat.shape[1]),
        observation_covariance=observation_variance * np.eye(flat.shape[1]),
    )
    means, _ = kalman.smooth(flat)
    return means.reshape(shape).astype(np.float32)


def _one_hot(size: int, index: int, device: torch.device) -> torch.Tensor:
    if not 0 <= index < size:
        raise ValueError(f"style/person id {index} is outside [0, {size})")
    value = torch.zeros(1, size, dtype=torch.float32, device=device)
    value[:, index] = 1
    return value


def _edit_articulations(
    deformation: np.ndarray,
    keypoint_indices: Sequence[int],
    edits: Sequence[LipArticulationEdit],
) -> np.ndarray:
    """Scale projections onto semantic lip-articulation directions."""

    if not edits:
        return deformation
    local_indices = {keypoint: index for index, keypoint in enumerate(keypoint_indices)}
    result = deformation.copy()
    for edit in edits:
        if edit.articulation not in ARTICULATION_VECTORS:
            raise ValueError(f"Unknown lip articulation: {edit.articulation}")
        direction = np.zeros((len(keypoint_indices), 3), dtype=np.float32)
        for (keypoint, axis), value in ARTICULATION_VECTORS[edit.articulation]:
            if keypoint in local_indices:
                direction[local_indices[keypoint], axis] = value
        norm_squared = float(np.sum(direction * direction))
        if norm_squared == 0:
            raise ValueError(
                f"Articulation {edit.articulation} is unavailable for the "
                "configured LAC keypoints"
            )
        projection = np.sum(result * direction[None], axis=(1, 2)) / norm_squared
        result += (edit.scale - 1.0) * projection[:, None, None] * direction[None]
    return result.astype(np.float32)


def _windowed_prediction(
    audio_features: np.ndarray,
    reference: np.ndarray,
    window_size: int,
    overlap: Optional[int],
    predict: Callable[
        [torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor
    ],
    device: torch.device,
) -> np.ndarray:
    """Autoregressive overlapping-window inference shared by LAC and EMC."""

    overlap = window_size // 5 if overlap is None else overlap
    if not 0 <= overlap < window_size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size")
    stride = window_size - overlap
    total = min(len(audio_features), len(reference))
    if total < 1:
        raise ValueError("Audio and reference sequences must not be empty")
    chunks = []
    previous_tail = None

    for start in range(0, total, stride):
        end = min(start + window_size, total)
        length = end - start
        warm_frames = 0 if previous_tail is None else min(overlap, length - 1)
        warm_start = None
        if warm_frames:
            warm_start = previous_tail[:, -warm_frames:].to(device)

        audio = torch.from_numpy(audio_features[start:end]).unsqueeze(0).to(device)
        ref = torch.from_numpy(reference[start:end]).unsqueeze(0).to(device)
        with torch.inference_mode():
            prediction = predict(audio, ref, warm_start)
        if prediction is None:
            raise RuntimeError("The autoregressive model produced no frames")

        prediction = prediction.float()
        tail_length = min(overlap, prediction.shape[1])
        previous_tail = prediction[:, -tail_length:].detach() if tail_length else None
        accepted = prediction[:, warm_frames:]
        chunks.append(accepted.squeeze(0).cpu().numpy())
        if end >= total:
            break

    return np.concatenate(chunks, axis=0)[:total].astype(np.float32)


class LipAudioAlignmentControl:
    """Paper-aligned LAC module owned by :class:`PCTalkPipeline`."""

    def __init__(self, config: LACConfig, device: torch.device):
        self.config = config
        self.device = device
        _, model_state, checkpoint_args = _load_checkpoint(config.checkpoint)
        self.person_num = _infer_person_num(
            model_state, checkpoint_args, config.person_num
        )
        self.window_size = _infer_window_size(
            model_state,
            checkpoint_args,
            config.window_size,
        )
        self.predict_all = config.predict_all_keypoints
        self.keypoint_indices = tuple(range(21)) if self.predict_all else LIP_INDICES

        args = SimpleNamespace(
            feature_dim=512,
            vertice_dim=len(self.keypoint_indices) * 3,
            T=self.window_size,
            person_num=self.person_num,
            device=device,
            style_template=False,
            style_freeze=False,
            use_arc=False,
        )
        self.model = LipMotionPredictor(args).to(device)
        self.model.load_state_dict(model_state, strict=True)
        self.model.eval().requires_grad_(False)

        self.refinement = None
        if config.use_refinement and config.refinement_checkpoint:
            if self.predict_all:
                raise ValueError(
                    "LipRefinementNetwork only supports the six-keypoint LAC mode"
                )
            refinement_args = SimpleNamespace()
            self.refinement = LipRefinementNetwork(refinement_args).to(device)
            try:
                refinement_state = torch.load(
                    config.refinement_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
            except TypeError:
                refinement_state = torch.load(
                    config.refinement_checkpoint, map_location="cpu"
                )
            refinement_state = refinement_state.get("model_ft", refinement_state)
            refinement_state = {
                key.removeprefix("module."): value
                for key, value in refinement_state.items()
            }
            self.refinement.load_state_dict(refinement_state, strict=True)
            self.refinement.eval().requires_grad_(False)

        self.style_encoder = None
        if config.style_encoder_checkpoint:
            style_args = SimpleNamespace(
                style_T=config.style_frames,
                vertice_dim=len(self.keypoint_indices) * 3,
                feature_dim=512,
                style_dim=512,
                person_num=self.person_num,
                device=device,
            )
            self.style_encoder = SpeakingStyleEncoder(style_args).to(device)
            _, style_state, _ = _load_checkpoint(config.style_encoder_checkpoint)
            style_state = _submodule_state(style_state, "obj_vector")
            self.style_encoder.load_state_dict(style_state, strict=True)
            self.style_encoder.eval().requires_grad_(False)

    def encode_reference_style(self, template: dict) -> torch.Tensor:
        if self.style_encoder is None:
            raise ValueError(
                "style_reference requires LACConfig.style_encoder_checkpoint"
            )
        deformation = []
        for frame in template["motion"][: self.config.style_frames]:
            current = render_keypoints(frame) - undeformed_keypoints(frame)
            deformation.append(current[0, self.keypoint_indices].T)
        ref_style = torch.from_numpy(np.stack(deformation)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            style_embedding, _ = self.style_encoder(ref_style)
        return style_embedding

    def apply(
        self,
        template: dict,
        audio_features: np.ndarray,
        style_template: Optional[dict] = None,
    ) -> dict:
        result = deepcopy(template)
        reference = np.stack(
            [
                undeformed_keypoints(frame)[0, self.keypoint_indices].T
                for frame in result["motion"]
            ]
        ).astype(np.float32)
        style_embedding = (
            self.encode_reference_style(style_template)
            if style_template is not None
            else None
        )
        style_code = _one_hot(self.person_num, self.config.person_id, self.device)

        def predict(audio, ref, warm_start):
            prediction = self.model.predict(
                audio,
                ref,
                style_code,
                vertice_init=warm_start,
                style_embedding=style_embedding,
            )
            if self.refinement is not None:
                prediction = self.refinement(audio, ref + prediction) - ref
            return prediction

        prediction = _windowed_prediction(
            audio_features,
            reference,
            self.window_size,
            self.config.overlap,
            predict,
            self.device,
        )
        # Model layout is [T, 3, N]; templates use [1, N, 3].
        prediction = prediction.transpose(0, 2, 1)
        if self.config.smooth:
            prediction = _smooth(prediction)
        prediction *= self.config.lip_scale
        prediction = _edit_articulations(
            prediction,
            self.keypoint_indices,
            self.config.articulation_edits,
        )

        result["n_frames"] = len(prediction)
        result["motion"] = result["motion"][: len(prediction)]
        for index, delta in enumerate(prediction):
            frame = result["motion"][index]
            expression = _frame_array(frame, "exp").reshape(1, 21, 3).copy()
            expression[0, self.keypoint_indices] = delta / _scale_value(frame)
            frame["exp"] = expression.astype(np.float32)
            frame["x_s"] = render_keypoints(frame)
        return result


class EmotionControl:
    """EMC module using emotional-minus-neutral deformation decomposition."""

    def __init__(self, config: EMCConfig, device: torch.device):
        if not config.checkpoint:
            raise ValueError("EMCConfig.checkpoint is required when EMC is enabled")
        self.config = config
        self.device = device
        _, model_state, checkpoint_args = _load_checkpoint(config.checkpoint)
        self.person_num = _infer_person_num(
            model_state, checkpoint_args, config.person_num
        )
        self.window_size = _infer_window_size(
            model_state,
            checkpoint_args,
            config.window_size,
        )
        args = SimpleNamespace(
            feature_dim=512,
            vertice_dim=21 * 3,
            T=self.window_size,
            person_num=self.person_num,
            emo_num=len(EMOTIONS),
            emo_type=config.emotion_type,
            device=device,
            style_template=False,
            person_embedding_bias="obj_vector.bias" in model_state,
        )
        self.model = EmotionMotionPredictor(args).to(device)
        self.model.load_state_dict(model_state, strict=True)
        self.model.eval().requires_grad_(False)

    @staticmethod
    def _region_indices(region: str) -> Tuple[int, ...]:
        if region == "all":
            return tuple(range(21))
        if region == "lips":
            return LIP_INDICES
        if region == "eyes":
            return EYE_INDICES
        if region == "brows":
            return BROW_INDICES
        if region == "upper_face":
            return tuple(sorted(set(EYE_INDICES + BROW_INDICES)))
        if region == "lower_face":
            return LIP_INDICES
        raise ValueError(f"Unknown emotion region: {region}")

    def _predict(
        self,
        emotion: str,
        audio_features: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        if emotion not in EMOTIONS:
            raise ValueError(f"Unknown emotion: {emotion}")
        person_code = _one_hot(self.person_num, self.config.person_id, self.device)
        emotion_code = _one_hot(len(EMOTIONS), EMOTIONS[emotion], self.device)

        def predict(audio, ref, warm_start):
            return self.model.predict(
                audio,
                ref,
                person_code,
                emotion_code,
                warm_start,
            )

        output = _windowed_prediction(
            audio_features,
            reference,
            self.window_size,
            self.config.overlap,
            predict,
            self.device,
        ).transpose(0, 2, 1)
        return _smooth(output) if self.config.smooth else output

    def apply(
        self,
        template: dict,
        audio_features: np.ndarray,
        conditions: Optional[Sequence[EmotionCondition]] = None,
    ) -> dict:
        conditions = tuple(conditions or self.config.conditions)
        if not conditions:
            return template

        result = deepcopy(template)
        reference = np.stack(
            [undeformed_keypoints(frame)[0].T for frame in result["motion"]]
        ).astype(np.float32)
        neutral = self._predict("neutral", audio_features, reference)
        emotion_cache = {"neutral": neutral}

        for condition in conditions:
            if condition.emotion == "neutral" or condition.intensity == 0:
                continue
            if condition.emotion not in emotion_cache:
                emotion_cache[condition.emotion] = self._predict(
                    condition.emotion, audio_features, reference
                )
            pure_emotion = emotion_cache[condition.emotion] - neutral
            indices = self._region_indices(condition.region)
            for frame_index, frame in enumerate(result["motion"]):
                expression = _frame_array(frame, "exp").reshape(1, 21, 3).copy()
                expression[0, indices] += (
                    pure_emotion[frame_index, indices]
                    * condition.intensity
                    / _scale_value(frame)
                )
                frame["exp"] = expression.astype(np.float32)

        for frame in result["motion"]:
            frame["x_s"] = render_keypoints(frame)
        return result
