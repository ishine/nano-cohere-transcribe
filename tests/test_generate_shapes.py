"""Generate loop shape sanity tests with a tiny mocked model.

We exercise the greedy-decode control flow (prefill, step, EOS-stop) without
loading any real weights — the 2B model is too big for CI/dev loops.
"""
from __future__ import annotations

import os

import pytest
import torch

from nano_cohere_transcribe.generate import greedy_generate


class _MockTokenizer:
    eos_id = 3

    def build_prompt(self, language, punctuation=True):
        return [7, 4, 16, 62, 62, 5, 9, 11, 13]  # 9-token prompt (ids irrelevant here)


class _MockDecoder(torch.nn.Module):
    """Deterministic decoder that emits token ids 100, 101, 102, ..., EOS at step N."""

    def __init__(self, hidden=4, vocab=200, stop_after=5):
        super().__init__()
        self.hidden = hidden
        self.vocab = vocab
        self.stop_after = stop_after
        self.calls = 0

    def forward(
        self,
        input_ids,
        positions,
        encoder_hidden_states,
        self_attn_mask,
        cross_attn_mask,
        self_kv_caches,
        cross_kv_caches,
    ):
        B, T = input_ids.shape
        h = torch.zeros(B, T, self.hidden, device=input_ids.device)
        past = 0 if self_kv_caches is None else self_kv_caches[0][0].size(2)
        # Grow caches by T positions.
        new_self = []
        for layer in range(2):
            prev = self_kv_caches[layer] if self_kv_caches else (
                torch.zeros(B, 2, 0, self.hidden, device=input_ids.device),
                torch.zeros(B, 2, 0, self.hidden, device=input_ids.device),
            )
            pad = torch.zeros(B, 2, T, self.hidden, device=input_ids.device)
            new_self.append((torch.cat([prev[0], pad], dim=2), torch.cat([prev[1], pad], dim=2)))
        new_cross = cross_kv_caches or [
            (torch.zeros(B, 2, encoder_hidden_states.size(1), self.hidden), ) * 2 for _ in range(2)
        ]
        self.calls += 1
        return h, new_self, new_cross


class _MockHead(torch.nn.Module):
    def __init__(self, stop_after=5, vocab=200):
        super().__init__()
        self.stop_after = stop_after
        self.vocab = vocab
        self.calls = 0

    def forward(self, h):
        # h: [B, T, hidden]
        B, T, _ = h.shape
        logits = torch.full((B, T, self.vocab), -1e9)
        # Emit token 100 + self.calls for the last position until stop.
        tok = 100 + self.calls
        if self.calls >= self.stop_after:
            tok = 3  # EOS
        logits[:, -1, tok] = 10.0
        self.calls += 1
        return logits


class _MockModel(torch.nn.Module):
    def __init__(self, stop_after=5):
        super().__init__()
        self.transf_decoder = _MockDecoder(stop_after=stop_after)
        self.log_softmax = _MockHead(stop_after=stop_after)
        self.tokenizer = _MockTokenizer()


def test_greedy_generate_stops_on_eos():
    model = _MockModel(stop_after=4)
    enc = torch.zeros(1, 10, 4)
    enc_len = torch.tensor([10])
    out = greedy_generate(
        model, encoder_hidden_states=enc, encoder_lengths=enc_len, language="en", max_new_tokens=32
    )
    assert isinstance(out, list) and len(out) == 1
    # Head call 0 is prefill -> token 100; subsequent steps -> 101, 102, 103; then EOS.
    assert out[0] == [100, 101, 102, 103]


def test_greedy_generate_respects_max_new_tokens():
    model = _MockModel(stop_after=999)
    enc = torch.zeros(1, 10, 4)
    enc_len = torch.tensor([10])
    out = greedy_generate(
        model, encoder_hidden_states=enc, encoder_lengths=enc_len, language="en", max_new_tokens=7
    )
    assert len(out[0]) == 7
