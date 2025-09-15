#!/usr/bin/env python3
"""Runner script for WhisperLab application."""

import importlib.util
import sys
from pathlib import Path


def main():
    """Run the WhisperLab application."""
    # Get the path to the main.py file
    src_path = Path(__file__).parent / "src"
    main_file = src_path / "main.py"

    # Load the main module dynamically
    spec = importlib.util.spec_from_file_location("main", main_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load main module from {main_file}")

    main_module = importlib.util.module_from_spec(spec)
    sys.modules["main"] = main_module
    spec.loader.exec_module(main_module)

    # Run the main function
    main_module.main()


if __name__ == "__main__":
    main()
