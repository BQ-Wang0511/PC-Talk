"""Standalone and package-compatible PC-Talk command-line entry point."""

try:  # ``python -m opensource`` from the parent directory
    from .pctalk.cli import (
        build_parser,
        main,
        parse_articulation,
        parse_emotion,
    )
except ImportError:  # ``python inference.py`` from this standalone directory
    from pctalk.cli import build_parser, main, parse_articulation, parse_emotion

__all__ = ["build_parser", "main", "parse_articulation", "parse_emotion"]


if __name__ == "__main__":
    main()
