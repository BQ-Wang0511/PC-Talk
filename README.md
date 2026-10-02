# PC-Talk

Official repository for the CVPR 2026 paper

**PC-Talk: Precise Facial Animation Control for Audio-Driven Talking Face Generation**

<a href="https://scholar.google.cz/citations?user=pBF9Mn8AAAAJ&amp;hl=zh-CN&amp;oi=ao" target="_blank">Baiqin Wang</a><sup>1,2</sup>, <a href="https://xiangyuzhu-open.github.io/homepage/" target="_blank">Xiangyu Zhu</a><sup>1,2,*</sup>, Fan Shen<sup>3</sup>, Hao Xu<sup>3,4</sup>, <a href="https://scholar.google.cz/citations?hl=zh-CN&amp;user=cuJ3QG8AAAAJ" target="_blank">Zhen Lei</a><sup>1,2,5,6</sup>

<sup>1</sup>MAIS, Institute of Automation, Chinese Academy of Sciences, <sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences, <sup>3</sup>Psyche AI.INC, <sup>4</sup>The Hong Kong University of Science and Technology, <sup>5</sup>CAIR, HKISI, Chinese Academy of Sciences, <sup>6</sup>SCSE, FIE, M.U.S.T

<sup>*</sup>Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-2503.14295-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.14295)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f?logo=googlechrome&logoColor=white)](https://bq-wang0511.github.io/PC-Talk/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Checkpoints-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/doubi-killer/PC-Talk)

## Standalone inference

This directory is a self-contained source release of **PC-Talk: Precise Facial
Animation Control for Audio-Driven Talking Face Generation**. It does not
import code or configuration from the parent research repository. Runtime
checkpoints are distributed separately on
[Hugging Face](https://huggingface.co/doubi-killer/PC-Talk), together with their
SHA-256 integrity manifest.

## Integrated architecture

```text
audio -> bundled AV encoder -> LAC -> EMC -> LivePortrait warping/decoder -> video
                               ^      ^
                          style/lip  emotion controls
```

`PCTalkPipeline` extends the bundled `LivePortraitPipeline` and owns the audio
encoder, LAC, and EMC models. Motion stays in memory throughout inference; no
subprocess or temporary pickle is used between audio-to-motion and rendering.
Motion pickle export remains available as an explicit diagnostic feature.

- **LAC** predicts the six semantic lip-keypoint deformations. It supports
  preset style IDs, reference-video styles, overlapping autoregressive
  inference, refinement MLP, lip-motion scale, and pursing/widening/opening
  articulation editing.
- **EMC** computes pure emotion as emotional combined deformation minus neutral
  combined deformation. Multiple emotions can be composed by intensity and
  facial region.
- **LivePortrait** is bundled under `pctalk/liveportrait` and performs face
  cropping, implicit-keypoint extraction, warping, decoding, stitching,
  paste-back, and audio muxing.

## Directory layout

```text
opensource/
├── pctalk/
│   ├── models/                 # audio encoder, LAC, EMC, style/refinement
│   ├── liveportrait/           # bundled human LivePortrait backend
│   ├── checkpoints/            # checkpoint manifest and local weight location
│   ├── audio_features.py
│   ├── motion_controls.py
│   ├── pipeline.py
│   └── cli.py
├── tests/
├── examples/
│   ├── assets/obama.jpg
│   ├── run_image.sh
│   └── run_video.sh
├── inference.py
├── pyproject.toml
└── requirements.txt
```

Runtime checkpoints are intentionally excluded from Git. Download them from
[doubi-killer/PC-Talk on Hugging Face](https://huggingface.co/doubi-killer/PC-Talk)
and place them under `pctalk/checkpoints` using the instructions below.

## Installation

Clone the source repository and create an environment with a PyTorch build
suitable for the target CUDA/CPU platform, then install:

```bash
git clone https://github.com/BQ-Wang0511/PC-Talk.git
cd PC-Talk
pip install -r requirements.txt
pip install -e .
```

FFmpeg and FFprobe must be available on `PATH`. Inference uses 16 kHz mono
audio and produces 25 fps video.

## Download and configure checkpoints

All 12 runtime model files are hosted at
[https://huggingface.co/doubi-killer/PC-Talk](https://huggingface.co/doubi-killer/PC-Talk).
They include the PC-Talk audio encoder, LAC/refinement, EMC, reference-style
encoder, LivePortrait models, landmark model, and face detectors (about 920 MB
in total). Model files are not stored in the GitHub source repository.

Run the following commands **from the PC-Talk repository root**:

```bash
pip install -U huggingface_hub
hf download doubi-killer/PC-Talk --local-dir pctalk/checkpoints
```

The download preserves the model directory structure. Do not create an extra
`PC-Talk/` or `checkpoints/` folder inside `pctalk/checkpoints`:

```text
pctalk/checkpoints/
├── audio_encoder.pth
├── lac.pth
├── emc.pth
├── style_encoder.pth
├── SHA256SUMS
├── liveportrait/
│   ├── landmark.onnx
│   ├── base_models/
│   │   ├── appearance_feature_extractor.pth
│   │   ├── motion_extractor.pth
│   │   ├── spade_generator.pth
│   │   └── warping_module.pth
│   └── retargeting_models/
│       └── stitching_retargeting_module.pth
└── insightface/
    └── models/buffalo_l/
        ├── 2d106det.onnx
        └── det_10g.onnx
```

Alternatively, download through the Python API:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="doubi-killer/PC-Talk",
    local_dir="pctalk/checkpoints",
)
```

For a manual download, open the Hugging Face repository's **Files and versions**
tab and copy the files into the exact paths shown above, preserving the
`liveportrait/` and `insightface/` subdirectories.

The CLI uses this directory by default; no source-code changes are required.
`lac.pth` contains both the LAC model and its refinement model. `emc.pth` is
used for emotion control, and `style_encoder.pth` is used with reference-style
input. Custom LAC and EMC locations can be passed with `--lac-checkpoint` and
`--emc-checkpoint`.

On Linux, verify all 12 model files from the repository root:

```bash
sha256sum -c pctalk/checkpoints/SHA256SUMS
```

Checkpoint files remain ignored by Git. Third-party weights retain their
upstream usage terms; in particular, the InsightFace model assets are for
non-commercial research use and should be replaced for commercial deployment.

## Image-driven example

The repository includes `examples/assets/obama.jpg` for the single-image
inference example.

Run it with any speech WAV file:

```bash
./examples/run_image.sh /path/to/speech.wav
```

The equivalent command is:

```bash
python -m pctalk \
  --source examples/assets/obama.jpg \
  --audio /path/to/speech.wav \
  --lac-checkpoint pctalk/checkpoints/lac.pth \
  --emc-checkpoint pctalk/checkpoints/emc.pth \
  --person-id 192 \
  --lip-scale 0.6 \
  --emotion happy:all:0.5 \
  --output-dir outputs/image_example
```

Each run writes only the final animation. PC-Talk does not create a
side-by-side comparison file. The video uses an unused random 10-digit numeric
filename such as `4839201746.mp4`, and the CLI prints its full output path.

Without a motion reference, a source image supplies a static head pose while
LAC and EMC animate its mouth and expression. To borrow head motion from
another image, video, or trusted LivePortrait motion template, optionally add:

```bash
--motion-reference /path/to/reference.mp4
```

## Video-driven example

When `--source` is a video, do not pass `--motion-reference` for normal use.
PC-Talk automatically extracts and preserves the source video's own pose
sequence:

```bash
./examples/run_video.sh /path/to/source.mp4 /path/to/speech.wav
```

Equivalent command:

```bash
python -m pctalk \
  --source /path/to/source.mp4 \
  --audio /path/to/speech.wav \
  --lac-checkpoint pctalk/checkpoints/lac.pth \
  --emc-checkpoint pctalk/checkpoints/emc.pth \
  --person-id 192 \
  --lip-scale 0.6 \
  --emotion happy:all:0.5 \
  --output-dir outputs/video_example
```

Use `--motion-reference` with a video only when intentionally replacing its
pose sequence with another reference.

## Additional inference controls

From inside this directory, either the installed command or the standalone
script can be used:

```bash
pctalk \
  --source path/to/source.jpg \
  --audio path/to/speech.wav \
  --person-id 192 \
  --lip-scale 0.6 \
  --output-dir outputs
```

Equivalent without installing the console entry point:

```bash
python -m pctalk \
  --source path/to/source.jpg \
  --audio path/to/speech.wav \
  --person-id 192 \
  --lip-scale 0.6
```

`python inference.py` remains available as a compatibility entry point.

Reference speaking style and emotion:

```bash
pctalk \
  --source source.jpg \
  --audio speech.wav \
  --style-reference style.mp4 \
  --person-id 192 \
  --lip-scale 0.6 \
  --emotion happy:all:0.5
```

Regional compound emotion and articulation control:

```bash
pctalk \
  --source source.jpg \
  --audio speech.wav \
  --person-id 192 \
  --emotion happy:lips:0.5 \
  --emotion sad:upper_face:0.3 \
  --lip-scale 0.6 \
  --articulation pursing:1.3
```

Useful controls:

- `--person-id N`: select a preset speaking style. Recommended starting IDs
  are `192`, `166`, and `202`; they produce different learned articulation
  styles from the bundled LAC checkpoint.
- `--emotion EMOTION[:REGION[:INTENSITY]]`: apply EMC with a default intensity
  of `0.5`; keep values near or below `0.5` for restrained expressions.
- `--motion-reference`: use another image, video, or trusted motion pickle for
  the base pose.
- `--articulation NAME:SCALE`: edit `pursing`, `widening`, or `opening`; the
  option may be repeated.
- `--output-motion path.pkl`: explicitly export generated motion.
- `--no-motion-smoothing`: disable the post-generation Kalman smoother.

## Python API

```python
from pctalk import (
    AudioConfig,
    EMCConfig,
    EmotionCondition,
    LACConfig,
    LipArticulationEdit,
    PCTalkPipeline,
)
from pctalk.liveportrait.config.crop_config import CropConfig
from pctalk.liveportrait.config.inference_config import InferenceConfig

pipeline = PCTalkPipeline(
    inference_cfg=InferenceConfig(flag_relative_motion=False),
    crop_cfg=CropConfig(),
    audio_cfg=AudioConfig(),
    lac_cfg=LACConfig(
        articulation_edits=(LipArticulationEdit("pursing", 1.2),),
    ),
    emc_cfg=EMCConfig(checkpoint="pctalk/checkpoints/emc.pth"),
)

motion = pipeline.generate_motion(
    "speech.wav",
    "source.jpg",
    [EmotionCondition("happy", intensity=0.5, region="lips")],
)
```

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q pctalk inference.py
```

Run the tests after placing the separately distributed checkpoints in their
documented locations. The lightweight asset test verifies that every required
runtime file is present but does not load the neural networks.

## Citation

If you find PC-Talk useful in your research, please cite:

```bibtex
@inproceedings{wang2026pc,
  title     = {PC-Talk: Precise Facial Animation Control for Audio-Driven Talking Face Generation},
  author    = {Wang, Baiqin and Zhu, Xiangyu and Shen, Fan and Xu, Hao and Lei, Zhen},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {25153--25162},
  year      = {2026}
}
```

## Acknowledgements

PC-Talk builds on ideas and implementation foundations from the following
open-source projects. We sincerely thank their authors and contributors:

- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) provides the portrait
  animation, warping, stitching, retargeting, and rendering backbone.
- [FaceFormer](https://github.com/EvelynFan/FaceFormer) inspired the
  transformer-based autoregressive facial-motion modeling components.
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) provides the foundation for
  the audio feature encoder and mel-spectrogram preprocessing design.

## Security and licensing

Motion templates use Python pickle and must only be loaded from trusted
sources. See `THIRD_PARTY_NOTICES.md` and `RELEASE_CHECKLIST.md` before public
distribution. In particular, the bundled InsightFace-compatible model assets
are restricted to non-commercial research use and should be replaced for
commercial deployment.
