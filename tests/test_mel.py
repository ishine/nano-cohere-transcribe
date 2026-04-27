"""Feature extractor sanity tests. Compares against the transformers reference
if it's available; otherwise checks internal invariants only.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from nano_cohere_transcribe.mel import FilterbankFeatures

try:
    from transformers import AutoFeatureExtractor  # type: ignore

    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False


def _snapshot() -> str:
    """Pick a snapshot that contains BOTH the preprocessor config AND the custom
    remote-code Python files AutoProcessor needs (modeling/processing/tokenization)."""
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots"
    )
    if not os.path.isdir(cache):
        pytest.skip("Cohere snapshot not cached.")
    required = ("preprocessor_config.json", "tokenizer.model", "processing_cohere_asr.py", "tokenization_cohere_asr.py")
    for name in sorted(os.listdir(cache), reverse=True):
        d = os.path.join(cache, name)
        if all(os.path.exists(os.path.join(d, f)) for f in required):
            return d
    pytest.skip("No snapshot with all required remote-code files.")


def test_get_seq_len_matches_reference():
    fb = FilterbankFeatures()
    seq = torch.tensor([16000, 8000], dtype=torch.float)
    # For n_fft=512, hop=160: floor((L + 512 - 512) / 160) = floor(L / 160)
    out = fb.get_seq_len(seq)
    assert out.tolist() == [100, 50]


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_mel_parity_against_transformers():
    """nano FilterbankFeatures ≈ CohereAsrFeatureExtractor on a short waveform.

    Uses the checkpoint's `preprocessor.featurizer.{fb,window}` buffers (shared
    between impls), so the only numerical differences come from the STFT path.
    """
    snap = _snapshot()
    fe = AutoFeatureExtractor.from_pretrained(snap, trust_remote_code=True)

    # 1.5 s sine sweep to exercise mel bins broadly.
    sr = 16000
    t = np.arange(sr * 3 // 2) / sr
    waveform = 0.2 * np.sin(2 * np.pi * np.linspace(200, 3000, len(t)) * t).astype(np.float32)

    # Pull the exact fb/window the reference uses and plug into our module.
    nano = FilterbankFeatures(
        sample_rate=16000, n_window_size=400, n_window_stride=160, n_fft=512, nfilt=128
    )
    nano.fb = fe.filterbank.fb.float().clone()
    if nano.fb.dim() == 2:
        nano.fb = nano.fb.unsqueeze(0)
    nano.window = fe.filterbank.window.float().clone()

    x = torch.from_numpy(waveform).unsqueeze(0)
    seq = torch.tensor([waveform.shape[0]], dtype=torch.long)
    # Disable dither to remove the only non-deterministic path.
    nano.dither = 0.0
    fe.filterbank.dither = 0.0

    ref_feats = fe(waveform, sampling_rate=sr, return_tensors="pt")["input_features"].float()
    feats, _ = nano(x, seq)
    feats = feats[..., : ref_feats.shape[-1]].float()

    max_abs = (feats - ref_feats).abs().max().item()
    assert max_abs < 1e-3, f"max abs diff {max_abs}"
