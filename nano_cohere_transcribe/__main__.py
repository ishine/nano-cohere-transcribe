"""CLI entry point: `python -m nano_cohere_transcribe audio.wav --language en`."""
from __future__ import annotations

import argparse
import sys

from .api import from_pretrained, transcribe_file
from .tokenizer import SUPPORTED_LANGUAGES


def main() -> int:
    p = argparse.ArgumentParser(prog="nano-cohere-transcribe")
    p.add_argument("audio", help="Path to an audio file (wav/mp3/ogg/m4a/...).")
    p.add_argument(
        "--model",
        default="CohereLabs/cohere-transcribe-03-2026",
        help="HF repo id or local snapshot dir (default: %(default)s).",
    )
    p.add_argument(
        "--language",
        default="en",
        choices=sorted(SUPPORTED_LANGUAGES),
        help="ISO 639-1 language code (default: en).",
    )
    p.add_argument("--device", default="cuda", help="torch device (default: cuda).")
    p.add_argument("--no-punctuation", action="store_true", help="Disable punctuation/capitalization.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Internal batch size for long-form chunks (default: 8).")
    args = p.parse_args()

    model = from_pretrained(args.model, device=args.device)
    result = transcribe_file(
        model,
        args.audio,
        language=args.language,
        punctuation=not args.no_punctuation,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )
    print(result.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
