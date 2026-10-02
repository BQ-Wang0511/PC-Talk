# Examples

`assets/obama.jpg` provides the source portrait for the single-image example.

Run image-driven inference with any speech WAV file:

```bash
./examples/run_image.sh /path/to/speech.wav
```

Inference writes only the final animation, named with a random 10-digit number
such as `4839201746.mp4`; no comparison video is created.

The source image supplies a static base pose. To borrow head motion from
another image, video, or trusted LivePortrait motion template, add
`--motion-reference PATH` to the command in `run_image.sh`.

Run video-driven inference:

```bash
./examples/run_video.sh /path/to/source.mp4 /path/to/speech.wav
```

The video example deliberately does not pass `--motion-reference`. PC-Talk
extracts and preserves the source video's own pose sequence automatically.
