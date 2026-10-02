"""Command-line entry point for the integrated PC-Talk pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import (
    AudioConfig,
    EMCConfig,
    EmotionCondition,
    LACConfig,
    LipArticulationEdit,
)


EMOTION_NAMES = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgusted",
    "surprised",
    "contempt",
)
REGION_NAMES = ("all", "lips", "eyes", "brows", "upper_face", "lower_face")
ARTICULATION_NAMES = ("pursing", "widening", "opening")


def parse_emotion(value: str) -> EmotionCondition:
    """Parse EMOTION[:REGION[:INTENSITY]]."""

    fields = value.split(":")
    if len(fields) > 3:
        raise argparse.ArgumentTypeError(
            "emotion must use EMOTION[:REGION[:INTENSITY]]"
        )
    emotion = fields[0]
    region = fields[1] if len(fields) >= 2 else "all"
    try:
        intensity = float(fields[2]) if len(fields) == 3 else 0.5
    except ValueError as exc:
        raise argparse.ArgumentTypeError("emotion intensity must be a number") from exc
    if emotion not in EMOTION_NAMES:
        raise argparse.ArgumentTypeError(f"unknown emotion: {emotion}")
    if region not in REGION_NAMES:
        raise argparse.ArgumentTypeError(f"unknown emotion region: {region}")
    return EmotionCondition(emotion=emotion, region=region, intensity=intensity)


def parse_articulation(value: str) -> LipArticulationEdit:
    """Parse ARTICULATION:SCALE."""

    fields = value.split(":")
    if len(fields) != 2 or fields[0] not in ARTICULATION_NAMES:
        raise argparse.ArgumentTypeError(
            "articulation must use pursing|widening|opening:SCALE"
        )
    try:
        scale = float(fields[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("articulation scale must be a number") from exc
    return LipArticulationEdit(articulation=fields[0], scale=scale)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PC-Talk: integrated LAC + EMC + LivePortrait inference"
    )
    parser.add_argument("--source", "-s", required=True, help="portrait image/video")
    parser.add_argument("--audio", "-a", required=True, help="driving audio")
    parser.add_argument(
        "--motion-reference",
        help=(
            "optional image/video/PKL pose override; omit it to use the "
            "source image or source video's own motion"
        ),
    )
    checkpoint_root = Path(__file__).resolve().parent / "checkpoints"
    parser.add_argument(
        "--lac-checkpoint",
        default=str(checkpoint_root / "lac.pth"),
    )
    parser.add_argument(
        "--emc-checkpoint",
        help="defaults to the bundled EMC checkpoint when --emotion is used",
    )
    parser.add_argument(
        "--audio-encoder-checkpoint",
        default=AudioConfig().checkpoint,
    )
    parser.add_argument(
        "--refinement-checkpoint",
        default=str(checkpoint_root / "lac.pth"),
        help="defaults to model_ft in the bundled LAC checkpoint",
    )
    parser.add_argument("--style-reference", help="reference speaking-style video/PKL")
    parser.add_argument(
        "--style-encoder-checkpoint",
        help="defaults to the bundled style encoder with --style-reference",
    )
    parser.add_argument("--style-frames", type=int, default=100)
    parser.add_argument("--person-num", type=int)
    parser.add_argument("--person-id", type=int, default=0)
    parser.add_argument("--emotion-person-id", type=int, default=0)
    parser.add_argument("--emotion-person-num", type=int)
    parser.add_argument("--lip-scale", type=float, default=1.0)
    parser.add_argument(
        "--articulation",
        action="append",
        type=parse_articulation,
        default=[],
        metavar="NAME:SCALE",
        help="repeatable semantic lip edit: pursing, widening, or opening",
    )
    parser.add_argument("--lac-window", type=int, help="defaults to the checkpoint's T")
    parser.add_argument("--emc-window", type=int, help="defaults to the checkpoint's T")
    parser.add_argument("--overlap", type=int)
    parser.add_argument(
        "--emotion",
        action="append",
        type=parse_emotion,
        default=[],
        metavar="EMOTION[:REGION[:INTENSITY]]",
        help="repeat to compose regional emotions; intensity defaults to 0.5",
    )
    parser.add_argument("--predict-all-keypoints", action="store_true")
    parser.add_argument("--output-dir", "-o", default="animations")
    parser.add_argument(
        "--output-motion", help="optionally export generated motion PKL"
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--no-pasteback", action="store_true")
    parser.add_argument("--no-stitching", action="store_true")
    parser.add_argument("--crop-scale", type=float, default=2.3)
    parser.add_argument("--face-index", type=int, default=0)
    parser.add_argument("--no-motion-smoothing", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    cli = parser.parse_args()
    checkpoint_root = Path(__file__).resolve().parent / "checkpoints"
    if cli.emotion and not cli.emc_checkpoint:
        cli.emc_checkpoint = str(checkpoint_root / "emc.pth")
    if cli.style_reference and not cli.style_encoder_checkpoint:
        cli.style_encoder_checkpoint = str(checkpoint_root / "style_encoder.pth")
    for input_path in (cli.source, cli.audio, cli.lac_checkpoint):
        if not Path(input_path).exists():
            parser.error(f"input does not exist: {input_path}")
    optional_paths = (
        cli.motion_reference,
        cli.emc_checkpoint,
        cli.audio_encoder_checkpoint,
        cli.refinement_checkpoint,
        cli.style_reference,
        cli.style_encoder_checkpoint,
    )
    for input_path in optional_paths:
        if input_path and not Path(input_path).exists():
            parser.error(f"input does not exist: {input_path}")
    from .pipeline import PCTalkPipeline
    from .liveportrait.config.argument_config import ArgumentConfig
    from .liveportrait.config.crop_config import CropConfig
    from .liveportrait.config.inference_config import InferenceConfig

    audio_cfg = AudioConfig(checkpoint=cli.audio_encoder_checkpoint)
    lac_cfg = LACConfig(
        checkpoint=cli.lac_checkpoint,
        window_size=cli.lac_window,
        overlap=cli.overlap,
        person_num=cli.person_num,
        person_id=cli.person_id,
        lip_scale=cli.lip_scale,
        articulation_edits=tuple(cli.articulation),
        predict_all_keypoints=cli.predict_all_keypoints,
        refinement_checkpoint=cli.refinement_checkpoint,
        style_reference=cli.style_reference,
        style_encoder_checkpoint=cli.style_encoder_checkpoint,
        style_frames=cli.style_frames,
        smooth=not cli.no_motion_smoothing,
    )
    emc_cfg = None
    if cli.emc_checkpoint:
        emc_cfg = EMCConfig(
            checkpoint=cli.emc_checkpoint,
            window_size=cli.emc_window,
            overlap=cli.overlap,
            person_num=cli.emotion_person_num,
            person_id=cli.emotion_person_id,
            conditions=tuple(cli.emotion),
        )

    use_half_precision = not cli.no_half and not cli.force_cpu
    inference_cfg = InferenceConfig(
        device_id=cli.device_id,
        flag_force_cpu=cli.force_cpu,
        flag_use_half_precision=use_half_precision,
        flag_pasteback=not cli.no_pasteback,
        flag_stitching=not cli.no_stitching,
        flag_relative_motion=False,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        output_fps=audio_cfg.fps,
        face_idx=cli.face_index,
    )
    crop_cfg = CropConfig(
        device_id=cli.device_id,
        flag_force_cpu=cli.force_cpu,
        scale=cli.crop_scale,
    )
    render_args = ArgumentConfig(
        source=cli.source,
        driving=cli.audio,
        output_dir=cli.output_dir,
        device_id=cli.device_id,
        flag_force_cpu=cli.force_cpu,
        flag_use_half_precision=use_half_precision,
        flag_pasteback=not cli.no_pasteback,
        flag_stitching=not cli.no_stitching,
        flag_relative_motion=False,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        scale=cli.crop_scale,
        face_idx=cli.face_index,
    )

    pipeline = PCTalkPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg,
        audio_cfg=audio_cfg,
        lac_cfg=lac_cfg,
        emc_cfg=emc_cfg,
    )
    output = pipeline.execute(
        render_args,
        audio_path=cli.audio,
        motion_reference=cli.motion_reference,
        emotion_conditions=cli.emotion,
        output_motion_path=cli.output_motion,
    )
    print(f"video: {output}")


if __name__ == "__main__":
    main()
