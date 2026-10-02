"""A single, integrated LivePortrait + LAC + EMC inference pipeline."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from .liveportrait.config.argument_config import ArgumentConfig
from .liveportrait.config.crop_config import CropConfig
from .liveportrait.config.inference_config import InferenceConfig
from .liveportrait.live_portrait_pipeline import LivePortraitPipeline
from .liveportrait.utils.helper import is_image, is_template, is_video
from .liveportrait.utils.io import (
    dump,
    load,
    load_image_rgb,
    load_video,
    resize_to_limit,
)
from .liveportrait.utils.video import get_fps

from .audio_features import AudioFeatureExtractor
from .config import AudioConfig, EMCConfig, EmotionCondition, LACConfig
from .motion_controls import EmotionControl, LipAudioAlignmentControl
from .output import random_numeric_output_name


class PCTalkPipeline(LivePortraitPipeline):
    """PC-Talk implemented as a proper extension of LivePortrait.

    The pipeline owns every model and passes motion in memory.  There are no
    subprocess calls and no temporary pickle between audio-to-motion and
    motion-to-image.  A motion pickle can still be exported explicitly for
    reproducibility or motion editing.
    """

    def __init__(
        self,
        inference_cfg: InferenceConfig,
        crop_cfg: CropConfig,
        audio_cfg: AudioConfig,
        lac_cfg: LACConfig,
        emc_cfg: Optional[EMCConfig] = None,
    ):
        super().__init__(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
        self.audio_cfg = audio_cfg
        self.lac_cfg = lac_cfg
        self.emc_cfg = emc_cfg
        self.torch_device = torch.device(self.live_portrait_wrapper.device)

        self.audio_feature_extractor = AudioFeatureExtractor(
            audio_cfg, self.torch_device
        )
        self.lac = LipAudioAlignmentControl(lac_cfg, self.torch_device)
        self.emc = (
            EmotionControl(emc_cfg, self.torch_device)
            if emc_cfg is not None and emc_cfg.checkpoint
            else None
        )

    def _extract_motion_template(self, media_path: str) -> dict:
        """Extract a LivePortrait motion template with the already loaded M."""

        if is_template(media_path):
            return load(media_path)

        inf_cfg = self.live_portrait_wrapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg
        if is_image(media_path):
            frames = [load_image_rgb(media_path)]
            fps = self.audio_cfg.fps
        elif is_video(media_path):
            frames = load_video(media_path)
            fps = int(get_fps(media_path))
        else:
            raise ValueError(f"Unsupported motion-reference format: {media_path}")

        frames = [
            resize_to_limit(frame, inf_cfg.source_max_dim, inf_cfg.source_division)
            for frame in frames
        ]
        crop_result = self.cropper.crop_source_video(
            frames,
            crop_cfg,
            face_idx=inf_cfg.face_idx,
        )
        cropped = crop_result["frame_crop_lst"]
        landmarks = crop_result["lmk_crop_lst"]
        if not cropped:
            raise RuntimeError(f"No face was detected in: {media_path}")
        cropped_256 = [cv2.resize(frame, (256, 256)) for frame in cropped]
        eye_ratios, lip_ratios = self.live_portrait_wrapper.calc_ratio(landmarks)
        prepared = self.live_portrait_wrapper.prepare_videos(cropped_256)
        return self.make_motion_template(
            prepared,
            eye_ratios,
            lip_ratios,
            output_fps=fps,
        )

    @staticmethod
    def _ping_pong_indices(length: int, target_length: int):
        if length < 1:
            raise ValueError("Cannot extend an empty motion template")
        if length == 1:
            return [0] * target_length
        cycle = list(range(length)) + list(range(length - 1, -1, -1))
        return [cycle[index % len(cycle)] for index in range(target_length)]

    def _extend_template(self, template: dict, target_length: int) -> dict:
        """Match a reference image/video template to the audio duration."""

        source = deepcopy(template)
        indices = self._ping_pong_indices(len(source["motion"]), target_length)
        result = deepcopy(source)
        result["motion"] = [deepcopy(source["motion"][i]) for i in indices]
        result["n_frames"] = target_length
        result["output_fps"] = self.audio_cfg.fps

        for key, fallback_shape in (("c_eyes_lst", (1, 2)), ("c_lip_lst", (1, 1))):
            values = source.get(key)
            if values:
                result[key] = [deepcopy(values[i]) for i in indices]
            else:
                result[key] = [
                    np.zeros(fallback_shape, dtype=np.float32)
                    for _ in range(target_length)
                ]
        return result

    def generate_motion(
        self,
        audio_path: str,
        motion_reference: str,
        emotion_conditions: Optional[Sequence[EmotionCondition]] = None,
    ) -> dict:
        """Run Audio Encoder -> LAC -> EMC and return an in-memory template."""

        audio_features = self.audio_feature_extractor.extract(audio_path)
        base_template = self._extend_template(
            self._extract_motion_template(motion_reference),
            len(audio_features),
        )

        style_template = None
        if self.lac_cfg.style_reference:
            style_template = self._extend_template(
                self._extract_motion_template(self.lac_cfg.style_reference),
                self.lac_cfg.style_frames,
            )
        motion = self.lac.apply(base_template, audio_features, style_template)
        if self.emc is not None:
            motion = self.emc.apply(motion, audio_features, emotion_conditions)
        motion["n_frames"] = len(motion["motion"])
        motion["output_fps"] = self.audio_cfg.fps
        return motion

    def execute(
        self,
        args: ArgumentConfig,
        audio_path: str,
        motion_reference: Optional[str] = None,
        emotion_conditions: Optional[Sequence[EmotionCondition]] = None,
        output_motion_path: Optional[str] = None,
    ):
        """Generate controlled motion and render it through LivePortrait."""

        motion_reference = motion_reference or args.source
        motion = self.generate_motion(
            audio_path,
            motion_reference,
            emotion_conditions=emotion_conditions,
        )
        if output_motion_path:
            Path(output_motion_path).parent.mkdir(parents=True, exist_ok=True)
            dump(output_motion_path, motion)

        # PC-Talk predicts an absolute expression in the reference coordinate
        # system.  LivePortrait retargeting ratios were measured before LAC/EMC
        # changed the expression, so those optional paths must stay disabled.
        integrated_overrides = {
            "flag_relative_motion": False,
            "flag_eye_retargeting": False,
            "flag_lip_retargeting": False,
            "animation_region": "all",
        }
        for key, value in integrated_overrides.items():
            setattr(args, key, value)
        self.live_portrait_wrapper.update_config(integrated_overrides)

        output_name = random_numeric_output_name(args.output_dir)

        return super().execute(
            args,
            driving_template_dct=motion,
            driving_audio_path=audio_path,
            output_name=output_name,
            output_fps_override=self.audio_cfg.fps,
            driving_expression_exact=True,
        )
