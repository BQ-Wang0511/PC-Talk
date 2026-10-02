# PC-Talk release checklist

## Required before publishing

- [x] Record SHA-256 hashes for all bundled runtime checkpoints in
      `pctalk/checkpoints/SHA256SUMS`.
- [ ] Add public download URLs for LAC, EMC, refinement, style, LivePortrait,
      landmark, and audio-encoder checkpoints.
- [ ] Publish a model card describing HDTF/MEAD training data, intended use,
      limitations, demographic bias, and consent/privacy considerations.
- [ ] Confirm redistribution rights for every checkpoint. Code and model
      licenses are separate.
- [ ] Add the PC-Talk authors' copyright notice and select a license for the
      newly added PC-Talk code. The existing repository license currently
      names the LivePortrait copyright holder.
- [ ] Keep the notice that bundled InsightFace models are restricted to
      non-commercial research, or replace them with commercially compatible
      face detection and landmark models.
- [ ] Review the audio-visual synchronization encoder and checkpoint terms
      before redistribution.
- [ ] Remove local absolute paths, cached datasets, generated media, notebooks,
      `__pycache__`, temporary files, and evaluation outputs from the release.
- [ ] Never publish private HDTF/MEAD-derived motion pickles or identity/style
      embeddings without confirming dataset terms.

## Reproducibility

- [ ] Record Python, PyTorch, CUDA, cuDNN, FFmpeg, and ONNX Runtime versions.
- [ ] Provide exact preprocessing commands: 16 kHz mono audio and 25 fps
      video.
- [ ] Export checkpoint training arguments (`T`, `person_num`, `emo_type`,
      feature dimensions) under each checkpoint's `args` key.
- [ ] Publish train/validation/test identity splits without publishing the
      underlying restricted media.
- [ ] Add deterministic single-image inference and a short regression asset.
- [ ] Record LSE-C/LSE-D, FID, NIQE, FVD, Accemo, and E-FID evaluation commands.

## Tests to run with release checkpoints

- [ ] Neutral image + audio, preset style.
- [ ] Neutral video + audio, verifying exactly 25 fps output.
- [ ] Reference speaking style.
- [ ] Lip-scale sweep (for example 0.8, 1.0, 1.2).
- [ ] Pursing, widening, and opening articulation-direction sweeps.
- [ ] Every EMC emotion at intensity 1.0.
- [ ] Compound emotion on lips and upper face.
- [ ] Motion export followed by normal LivePortrait template playback.
- [ ] CPU error message or CPU smoke test, depending on the supported matrix.
- [ ] Multi-face input with explicit `--face-index`.
- [ ] Audio shorter than one second and audio longer than the reference video.

## Standalone repository layout

```text
pctalk/
├── models/                 # cleaned LAC/EMC/audio model definitions
├── liveportrait/           # bundled renderer and cropping backend
├── checkpoints/            # bundled runtime assets + PC-Talk weights
├── pipeline.py             # integrated inference pipeline
├── config.py
└── cli.py
tests/
README.md
LICENSE
THIRD_PARTY_NOTICES.md
```
