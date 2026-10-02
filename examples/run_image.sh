#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/speech.wav" >&2
    exit 2
fi

audio_path="$(realpath "$1")"
example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${example_dir}/.." && pwd)"

cd "${project_root}"
python -m pctalk \
    --source examples/assets/obama.jpg \
    --audio "${audio_path}" \
    --lac-checkpoint pctalk/checkpoints/lac.pth \
    --emc-checkpoint pctalk/checkpoints/emc.pth \
    --person-id 192 \
    --lip-scale 0.6 \
    --emotion happy:all:0.5 \
    --output-dir outputs/image_example
