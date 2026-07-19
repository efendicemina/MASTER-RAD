"""Module entry point for `python -m defect_classifier`."""

from __future__ import annotations

from .cli import main as cli_main


def main() -> int:
    """Run the command-line interface."""

    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
