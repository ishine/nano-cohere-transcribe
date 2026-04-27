#!/usr/bin/env python
"""Convenience wrapper around the ``nano_cohere_transcribe`` CLI.

`python transcribe.py audio.wav --language en` is identical to
`python -m nano_cohere_transcribe audio.wav --language en`.
"""
import sys

from nano_cohere_transcribe.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
