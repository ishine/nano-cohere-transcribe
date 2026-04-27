"""Batched transcription tests. Heavy ones are marked @slow."""
from __future__ import annotations

import os

import pytest
import torch


REPO_ID = "CohereLabs/cohere-transcribe-03-2026"
SAMPLE_WAV = os.path.join(os.path.dirname(__file__), "sample.wav")


def _weights_cached() -> bool:
    cache = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{REPO_ID.replace('/', '--')}/snapshots"
    )
    if not os.path.isdir(cache):
        return False
    for name in os.listdir(cache):
        if os.path.exists(os.path.join(cache, name, "model.safetensors")):
            return True
    return False


_skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
_skip_no_weights = pytest.mark.skipif(not _weights_cached(), reason="Cohere weights not cached")
_skip_no_sample = pytest.mark.skipif(not os.path.exists(SAMPLE_WAV), reason="tests/sample.wav missing")


@pytest.mark.slow
@_skip_no_cuda
@_skip_no_weights
@_skip_no_sample
def test_short_batch_matches_independent_calls():
    """Batched transcription of short waveforms is ~equal to N independent calls.

    Small char-level divergence is expected under bf16 + padding — we require
    per-row SequenceMatcher ratio >= 0.80.
    """
    import difflib

    from nano_cohere_transcribe import from_pretrained
    from nano_cohere_transcribe.audio import load_audio_16k_mono

    model = from_pretrained(REPO_ID, device="cuda")
    w = load_audio_16k_mono(SAMPLE_WAV)
    batch = [w[: 10 * 16000], w[: 8 * 16000], w[: 12 * 16000]]

    individual = [model.transcribe(x, language="en", batch_size=1) for x in batch]
    batched = model.transcribe_batch(batch, language="en", batch_size=3)

    assert len(batched) == len(individual)
    for i, (a, b) in enumerate(zip(individual, batched)):
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        assert ratio >= 0.80, f"row {i} diverges: {ratio:.2f}\n  solo={a!r}\n  batch={b!r}"


@pytest.mark.slow
@_skip_no_cuda
@_skip_no_weights
@_skip_no_sample
def test_long_form_batched_chunks_speed_up():
    """On long audio, batch_size>1 should be faster than batch_size=1."""
    import time

    import difflib
    import numpy as np

    from nano_cohere_transcribe import from_pretrained
    from nano_cohere_transcribe.audio import load_audio_16k_mono

    model = from_pretrained(REPO_ID, device="cuda")
    base = load_audio_16k_mono(SAMPLE_WAV).numpy()
    long_audio = torch.from_numpy(np.tile(base, 120))  # ~24 min
    duration_s = long_audio.shape[0] / 16000.0
    model.transcribe(long_audio[:16000], language="en")  # warmup

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    serial = model.transcribe(long_audio, language="en", batch_size=1)
    torch.cuda.synchronize()
    dt_serial = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    batched = model.transcribe(long_audio, language="en", batch_size=8)
    torch.cuda.synchronize()
    dt_batched = time.perf_counter() - t0

    print(
        f"\n[batch] {duration_s/60:.1f} min  serial={dt_serial:.1f}s  "
        f"bs=8={dt_batched:.1f}s  speedup={dt_serial/dt_batched:.2f}x"
    )
    assert dt_batched < dt_serial, "batched should be faster than serial"
    ratio = difflib.SequenceMatcher(None, serial, batched).ratio()
    assert ratio >= 0.80, f"batched vs serial similarity too low: {ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
