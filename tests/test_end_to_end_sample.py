"""End-to-end transcription test on a short English clip.

Loads the full 2B-parameter model; skipped unless CUDA is available AND the
model weights are already cached locally (avoid pulling 4 GB in CI).
"""
from __future__ import annotations

import os
import shutil

import pytest
import torch


REPO_ID = "CohereLabs/cohere-transcribe-03-2026"
SAMPLE_WAV = os.path.join(os.path.dirname(__file__), "sample.wav")


def _weights_cached() -> bool:
    cache = os.path.expanduser(f"~/.cache/huggingface/hub/models--{REPO_ID.replace('/', '--')}/snapshots")
    if not os.path.isdir(cache):
        return False
    for name in os.listdir(cache):
        if os.path.exists(os.path.join(cache, name, "model.safetensors")):
            return True
    return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for full-model test")
@pytest.mark.skipif(not _weights_cached(), reason="Cohere weights not cached; skipping end-to-end test")
@pytest.mark.skipif(not os.path.exists(SAMPLE_WAV), reason="tests/sample.wav missing")
def test_english_transcript_is_non_empty():
    from nano_cohere_transcribe import from_pretrained
    from nano_cohere_transcribe.audio import load_audio_16k_mono

    model = from_pretrained(REPO_ID, device="cuda")
    waveform = load_audio_16k_mono(SAMPLE_WAV)
    text = model.transcribe(waveform, language="en")
    assert isinstance(text, str) and len(text) > 0, f"Empty transcript: {text!r}"


if __name__ == "__main__":
    # Manual run convenience: `python tests/test_end_to_end_sample.py`.
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    from nano_cohere_transcribe import from_pretrained
    from nano_cohere_transcribe.audio import load_audio_16k_mono

    model = from_pretrained(REPO_ID, device="cuda")
    waveform = load_audio_16k_mono(SAMPLE_WAV)
    print(model.transcribe(waveform, language="en"))
