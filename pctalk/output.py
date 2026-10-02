"""Output naming utilities."""

from pathlib import Path
import secrets


def random_numeric_output_name(output_dir: str) -> str:
    """Return an unused 10-digit filename stem for an inference result."""

    directory = Path(output_dir)
    while True:
        name = str(secrets.randbelow(9_000_000_000) + 1_000_000_000)
        if (
            not (directory / f"{name}.mp4").exists()
            and not (directory / f"{name}.jpg").exists()
        ):
            return name
