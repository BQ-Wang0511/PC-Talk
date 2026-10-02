"""Typed configuration for the integrated PC-Talk pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = PACKAGE_ROOT / "checkpoints"

EmotionName = Literal[
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgusted",
    "surprised",
    "contempt",
]
EmotionRegion = Literal[
    "all",
    "lips",
    "eyes",
    "brows",
    "upper_face",
    "lower_face",
]
ArticulationName = Literal["pursing", "widening", "opening"]


@dataclass(frozen=True)
class AudioConfig:
    """Audio-visual encoder settings used by both LAC and EMC."""

    checkpoint: str = str(CHECKPOINT_ROOT / "audio_encoder.pth")
    sample_rate: int = 16_000
    fps: int = 25
    mel_step_size: int = 16
    batch_size: int = 64


@dataclass(frozen=True)
class LipArticulationEdit:
    """Scale one semantic articulation component of the LAC deformation.

    ``scale=1`` leaves the component unchanged, values above one enhance it,
    and values between zero and one attenuate it.
    """

    articulation: ArticulationName
    scale: float = 1.0


@dataclass(frozen=True)
class LACConfig:
    """Lip-Audio Alignment Control configuration."""

    checkpoint: str = str(CHECKPOINT_ROOT / "lac.pth")
    window_size: Optional[int] = None
    overlap: Optional[int] = None
    person_num: Optional[int] = None
    person_id: int = 0
    lip_scale: float = 1.0
    articulation_edits: Tuple[LipArticulationEdit, ...] = ()
    predict_all_keypoints: bool = False
    use_refinement: bool = True
    refinement_checkpoint: Optional[str] = str(CHECKPOINT_ROOT / "lac.pth")
    style_reference: Optional[str] = None
    style_encoder_checkpoint: Optional[str] = None
    style_frames: int = 100
    smooth: bool = True


@dataclass(frozen=True)
class EmotionCondition:
    """One regional emotional deformation to compose into the motion."""

    emotion: EmotionName
    intensity: float = 0.5
    region: EmotionRegion = "all"


@dataclass(frozen=True)
class EMCConfig:
    """Emotion Motion Control configuration.

    A missing checkpoint disables EMC.  Multiple ``EmotionCondition`` values
    can be supplied at inference time to form a compound expression.
    """

    checkpoint: Optional[str] = None
    window_size: Optional[int] = None
    overlap: Optional[int] = None
    person_num: Optional[int] = None
    person_id: int = 0
    emotion_type: int = 3
    smooth: bool = False
    conditions: Tuple[EmotionCondition, ...] = ()
