"""PC-Talk: precise audio-driven portrait animation control."""

from .config import (
    AudioConfig,
    EMCConfig,
    EmotionCondition,
    LACConfig,
    LipArticulationEdit,
)

__all__ = [
    "AudioConfig",
    "EMCConfig",
    "EmotionCondition",
    "LACConfig",
    "LipArticulationEdit",
    "PCTalkPipeline",
]


def __getattr__(name):
    # Keep configuration utilities importable in lightweight environments that
    # do not have PyTorch installed.  The heavy models are loaded on demand.
    if name == "PCTalkPipeline":
        from .pipeline import PCTalkPipeline

        return PCTalkPipeline
    raise AttributeError(name)
