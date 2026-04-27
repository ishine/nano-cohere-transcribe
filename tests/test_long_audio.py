"""Long-form audio end-to-end test.

Concatenates ``tests/sample.wav`` many times to produce a ~30-40 min clip
without having to download anything. Exercises:

- Automatic chunking via ``split_audio_chunks_energy``.
- Chunk-wise transcription + join.
- That the output contains content from multiple chunks (not just the first).

The test is heavy — skipped unless CUDA is available AND the model weights
are already cached locally. Use ``pytest -m slow`` or run this file directly
to exercise it. Run time on A100: ~1.5 min for 36 min of audio.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest
import soundfile as sf
import torch


REPO_ID = "CohereLabs/cohere-transcribe-03-2026"
SAMPLE_WAV = os.path.join(os.path.dirname(__file__), "sample.wav")

# ~36 min synthesized from ~12 s x 180 copies.
N_REPEATS = 180


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


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for long-form test")
@pytest.mark.skipif(not _weights_cached(), reason="Cohere weights not cached; skipping long-form test")
@pytest.mark.skipif(not os.path.exists(SAMPLE_WAV), reason="tests/sample.wav missing")
def test_long_audio_transcribes_end_to_end():
    from nano_cohere_transcribe import from_pretrained
    from nano_cohere_transcribe.audio import load_audio_16k_mono
    from nano_cohere_transcribe.chunk import split_audio_chunks_energy

    audio = load_audio_16k_mono(SAMPLE_WAV).numpy()  # float32, 16 kHz mono
    sr = 16000

    long_audio = np.tile(audio, N_REPEATS)
    duration_s = long_audio.shape[0] / sr
    print(f"\n[long-form] generated audio: {duration_s/60:.1f} min ({long_audio.shape[0]} samples)")

    # Sanity: chunker picks the right number of chunks (ballpark).
    chunks = split_audio_chunks_energy(
        long_audio, sample_rate=16000, max_audio_clip_s=35.0,
        overlap_chunk_second=5.0, min_energy_window_samples=1600,
    )
    n_chunks = len(chunks)
    print(f"[long-form] chunked into {n_chunks} segments")
    expected_min = int(duration_s // 35)  # upper bound on chunk duration
    assert expected_min <= n_chunks <= expected_min * 2 + 2, (
        f"unexpected chunk count {n_chunks} for {duration_s:.1f}s audio"
    )

    model = from_pretrained(REPO_ID, device="cuda")
    wave_t = torch.from_numpy(long_audio)

    t0 = time.perf_counter()
    text = model.transcribe(wave_t, language="en")
    elapsed = time.perf_counter() - t0
    rtfx = duration_s / elapsed
    print(f"[long-form] transcribed in {elapsed:.1f}s  RTFx={rtfx:.1f}x  chars={len(text)}")

    assert isinstance(text, str)
    assert len(text) > 500, f"transcript suspiciously short: {len(text)} chars"
    # The sample says "super short timelines" — since we looped it N times, that phrase
    # should appear many times in the output. Allow some chunk boundary loss.
    n_occurrences = text.lower().count("super short timelines")
    assert n_occurrences >= N_REPEATS // 3, (
        f"'super short timelines' appears only {n_occurrences} times in {N_REPEATS}-loop transcript"
    )


if __name__ == "__main__":
    # Manual run: `python tests/test_long_audio.py`
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "-m", "slow"]))
