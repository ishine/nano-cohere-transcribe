#!/usr/bin/env python3
"""Benchmark the nano-cohere-transcribe pure-PyTorch implementation."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from nano_cohere_transcribe import from_pretrained
from nano_cohere_transcribe.audio import load_audio_16k_mono


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    waveform = load_audio_16k_mono(args.audio)
    duration = waveform.shape[0] / 16_000.0

    t_cold0 = time.perf_counter()
    model = from_pretrained(args.model, device=args.device)  # includes warmup
    cold_s = time.perf_counter() - t_cold0

    # First timed call also acts as a cache-warmer past the internal warmup.
    _ = model.transcribe(waveform, language=args.language)

    times = []
    text = None
    for _ in range(args.runs):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = model.transcribe(waveform, language=args.language)
        if args.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    dt = float(np.mean(times))
    std = float(np.std(times))
    rtf = duration / dt if dt > 0 else float("inf")
    print(
        f"audio_s={duration:.2f}  cold_s={cold_s:.2f}  time_s={dt:.4f}  "
        f"std={std:.4f}  RTF={rtf:.2f}  text={text!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
