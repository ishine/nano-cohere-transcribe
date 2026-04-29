"""Measure cold-start time for nano-cohere-transcribe and the transformers reference.

Reports load time (from_pretrained) and warm-up time (one dummy forward pass)
separately. Run twice and take the second number — the first run pays for the
HF cache snapshot scan + python import warm-up.

Usage:
    python benchmark_coldstart.py
"""
from __future__ import annotations

import time

import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_nano():
    """Cold-start nano-cohere-transcribe."""
    # Force a fresh import so module-level cache doesn't help us.
    import importlib, sys
    for k in list(sys.modules):
        if k.startswith("nano_cohere_transcribe"):
            del sys.modules[k]
    from nano_cohere_transcribe import from_pretrained  # noqa: E402

    t0 = time.perf_counter()
    model = from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026", device="cuda", warmup=False,
    )
    _sync()
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.warmup(duration_s=1.0, batch_size=1)
    _sync()
    t_warmup = time.perf_counter() - t0
    return t_load, t_warmup


def time_transformers():
    """Cold-start the native transformers path."""
    import importlib, sys
    for k in list(sys.modules):
        if k.startswith("transformers"):
            del sys.modules[k]
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration  # noqa: E402

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
    model = CohereAsrForConditionalGeneration.from_pretrained(
        "CohereLabs/cohere-transcribe-03-2026", dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    _sync()
    t_load = time.perf_counter() - t0

    # Warmup: one full forward + generate on a dummy 1 s clip.
    t0 = time.perf_counter()
    import numpy as np
    dummy = np.zeros(16000, dtype="float32")
    inputs = processor(dummy, sampling_rate=16000, return_tensors="pt", language="en", punctuation=False)
    inputs.to(model.device, dtype=model.dtype)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=2)
    _sync()
    t_warmup = time.perf_counter() - t0
    return t_load, t_warmup


def main():
    print("Warming up python imports / HF cache scan...")
    time_nano()  # discard
    time_transformers()  # discard

    print("\n=== nano-cohere-transcribe ===")
    nl, nw = time_nano()
    print(f"  load:    {nl:.2f} s")
    print(f"  warmup:  {nw:.2f} s")
    print(f"  total:   {nl + nw:.2f} s")

    print("\n=== transformers 5.5.4 (native) ===")
    tl, tw = time_transformers()
    print(f"  load:    {tl:.2f} s")
    print(f"  warmup:  {tw:.2f} s")
    print(f"  total:   {tl + tw:.2f} s")


if __name__ == "__main__":
    main()
