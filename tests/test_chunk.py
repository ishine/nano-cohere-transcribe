"""Unit tests for energy-based chunking. Offline, no model required."""
from __future__ import annotations

import numpy as np
import pytest

from nano_cohere_transcribe.chunk import (
    NO_SPACE_LANGS,
    get_chunk_separator,
    join_chunk_texts,
    split_audio_chunks_energy,
)

SR = 16000
MAX_CLIP_S = 35.0
OVERLAP_S = 5.0
MIN_ENERGY_WIN = 1600


def _noise(n: int, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    return rng.standard_normal(n).astype(np.float32) * 0.1


def test_short_audio_is_returned_as_single_chunk():
    audio = _noise(SR * 10)  # 10 s, well under 35 s
    out = split_audio_chunks_energy(audio, SR, MAX_CLIP_S, OVERLAP_S, MIN_ENERGY_WIN)
    assert len(out) == 1
    np.testing.assert_allclose(out[0], audio)


def test_chunks_cover_full_audio_with_no_gap():
    audio = _noise(int(SR * 90))  # 90 s
    chunks = split_audio_chunks_energy(audio, SR, MAX_CLIP_S, OVERLAP_S, MIN_ENERGY_WIN)
    # Concatenating chunks reconstructs the full waveform (boundaries are non-overlapping splits).
    reconstructed = np.concatenate(chunks)
    assert reconstructed.shape == audio.shape
    np.testing.assert_allclose(reconstructed, audio)


def test_each_chunk_bounded_by_max_audio_clip_s():
    audio = _noise(int(SR * 120))  # 2 min
    chunks = split_audio_chunks_energy(audio, SR, MAX_CLIP_S, OVERLAP_S, MIN_ENERGY_WIN)
    max_samples = int(round(MAX_CLIP_S * SR))
    for c in chunks:
        assert c.shape[0] <= max_samples, f"chunk too long: {c.shape[0]} > {max_samples}"


def test_split_seeks_quietest_point_in_boundary_window():
    """Inject a silent dip near the boundary; the splitter should cut there."""
    # 60 s waveform: loud everywhere except a 200 ms silent pocket at t=32 s.
    rng = np.random.default_rng(42)
    audio = (rng.standard_normal(SR * 60).astype(np.float32)) * 0.5
    dip_start = 32 * SR
    dip_end = dip_start + int(0.2 * SR)
    audio[dip_start:dip_end] = 0.0

    chunks = split_audio_chunks_energy(audio, SR, MAX_CLIP_S, OVERLAP_S, MIN_ENERGY_WIN)
    assert len(chunks) >= 2
    # First chunk ends within the dip region (exact boundary depends on 100-ms window grid).
    first_end = chunks[0].shape[0]
    assert dip_start - MIN_ENERGY_WIN <= first_end <= dip_end + MIN_ENERGY_WIN, (
        f"first chunk ends at {first_end}, expected near dip [{dip_start}, {dip_end}]"
    )


def test_join_chunk_texts_skips_empty_and_trims():
    assert join_chunk_texts(["Hello ", "", "  world"], separator=" ") == "Hello world"
    assert join_chunk_texts([], separator=" ") == ""
    assert join_chunk_texts(["   "], separator=" ") == ""


@pytest.mark.parametrize(
    "language, expected",
    [("en", " "), ("fr", " "), ("de", " "), ("ja", ""), ("zh", "")],
)
def test_chunk_separator_matches_no_space_langs(language, expected):
    assert get_chunk_separator(language) == expected
    assert (language in NO_SPACE_LANGS) == (expected == "")


def test_rejects_non_mono():
    with pytest.raises(ValueError):
        split_audio_chunks_energy(np.zeros((2, SR * 40), dtype=np.float32), SR, MAX_CLIP_S, OVERLAP_S, MIN_ENERGY_WIN)
