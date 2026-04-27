"""Offline tokenizer tests — don't touch model weights."""
from __future__ import annotations

import os

import pytest

from nano_cohere_transcribe.tokenizer import (
    BOS,
    EOS,
    EMO_UNDEFINED,
    NODIARIZE,
    NOITN,
    NOTIMESTAMP,
    PNC,
    STARTOFCONTEXT,
    SUPPORTED_LANGUAGES,
    CohereTokenizer,
)


def _snapshot() -> str:
    # Any snapshot of the gated repo will do; pick the first one with tokenizer.model.
    cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--CohereLabs--cohere-transcribe-03-2026/snapshots"
    )
    if not os.path.isdir(cache):
        pytest.skip("Cohere snapshot not cached; skipping tokenizer test.")
    for name in sorted(os.listdir(cache)):
        path = os.path.join(cache, name, "tokenizer.model")
        if os.path.exists(path):
            return path
    pytest.skip("No snapshot with tokenizer.model.")


def test_prompt_matches_reference_build_prompt():
    tok = CohereTokenizer(_snapshot())
    ids = tok.build_prompt("en", punctuation=True)
    expected_pieces = [
        STARTOFCONTEXT,
        BOS,
        EMO_UNDEFINED,
        "<|en|>",
        "<|en|>",
        PNC,
        NOITN,
        NOTIMESTAMP,
        NODIARIZE,
    ]
    expected_ids = [tok.sp.piece_to_id(p) for p in expected_pieces]
    assert ids == expected_ids


def test_decode_strips_specials():
    tok = CohereTokenizer(_snapshot())
    # Encode plain text, prepend prompt + append EOS, decode.
    plain_ids = tok.sp.EncodeAsIds("hello world")
    raw_ids = tok.build_prompt("en") + plain_ids + [tok.eos_id, 42]
    out = tok.decode(raw_ids, skip_special_tokens=True).strip()
    assert out == "hello world", f"unexpected decode output {out!r}"


def test_decode_keeps_specials_when_requested():
    tok = CohereTokenizer(_snapshot())
    ids = tok.build_prompt("en") + [tok.eos_id]
    # skip_special_tokens=False: sp.DecodeIds will render whatever it can.
    out = tok.decode(ids, skip_special_tokens=False)
    # At minimum the output is a string (we don't assert exact surface form).
    assert isinstance(out, str)


@pytest.mark.parametrize("lang", sorted(SUPPORTED_LANGUAGES))
def test_build_prompt_every_language(lang):
    tok = CohereTokenizer(_snapshot())
    ids = tok.build_prompt(lang)
    assert len(ids) == 9
    assert ids[0] == tok.sp.piece_to_id(STARTOFCONTEXT)
    assert ids[1] == tok.bos_id
    lang_id = tok.sp.piece_to_id(f"<|{lang}|>")
    assert ids[3] == ids[4] == lang_id


def test_rejects_unsupported_language():
    tok = CohereTokenizer(_snapshot())
    with pytest.raises(ValueError):
        tok.build_prompt("xx")
