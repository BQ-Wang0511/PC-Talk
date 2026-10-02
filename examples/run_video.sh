#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 /path/to/source.mp4 /path/to/speech.wav" >&2
    exit 2
fi

source_path="$(realpath "$1")"
audio_path="$(realpath "$2")"
example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${example_dir}/.." && pwd)"

cd "${project_root}"
python -m pctalk \
    --source "${source_path}" \
    --audio "${audio_path}" \
    --lac-checkpoint pctalk/checkpoints/lac.pth \
    --emc-checkpoint pctalk/checkpoints/emc.pth \
    --person-id 192 \
    --lip-scale 0.6 \
    --emotion happy:all:0.5 \
    --output-dir outputs/video_example
