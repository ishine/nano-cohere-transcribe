#!/usr/bin/env python3
"""Benchmark the transformers reference (CohereAsrForConditionalGeneration).

On transformers >= 5.4 the custom ``CohereAsrTokenizer`` does not correctly
split the ``<|...|>`` control tokens before the SentencePiece encode, so
``tokenizer("<|startofcontext|>...")`` emits character-level BPE pieces and
``generate()`` produces garbage. Both the model card's naive snippet and the
model's built-in ``transcribe()`` hit this bug in 5.5.x. We work around it
by building the prompt token ids directly from SentencePiece (the same way
``nano_cohere_transcribe.tokenizer.build_prompt`` does), and we verify the
transformers output matches nano-cohere-transcribe.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import soundfile as sf
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio")
    p.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    # ---- load ----
    t_cold0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model, trust_remote_code=True, device_map=args.device, torch_dtype=torch.bfloat16
    )
    model.eval()

    # ---- audio ----
    audio, sr = sf.read(args.audio)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    duration = len(audio) / 16_000.0

    # ---- prompt: build token ids directly from SentencePiece (see module docstring). ----
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download
    sp = spm.SentencePieceProcessor()
    sp.Load(hf_hub_download(args.model, "tokenizer.model"))
    prompt_pieces = [
        "<|startofcontext|>", "<|startoftranscript|>", "<|emo:undefined|>",
        f"<|{args.language}|>", f"<|{args.language}|>",
        "<|pnc|>", "<|noitn|>", "<|notimestamp|>", "<|nodiarize|>",
    ]
    prompt_ids = [sp.piece_to_id(p) for p in prompt_pieces]
    decoder_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)

    # ---- features ----
    feats = processor.feature_extractor([audio], sampling_rate=16000, return_tensors="pt")
    feats = {k: v.to(model.device) for k, v in feats.items()}

    def run():
        with torch.inference_mode():
            out = model.generate(
                **feats,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=torch.ones_like(decoder_input_ids),
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                decoder_start_token_id=int(decoder_input_ids[0, 0].item()),
                use_cache=True,
            )
        # Strip prompt + special tokens (use our own SP decode — reference decode has the same bug).
        ids = out[0].tolist()
        if ids[: len(prompt_ids)] == prompt_ids:
            ids = ids[len(prompt_ids):]
        eos_id = sp.piece_to_id("<|endoftext|>")
        if eos_id in ids:
            ids = ids[: ids.index(eos_id)]
        # Filter any remaining control tokens.
        specials = {i for i in range(sp.get_piece_size())
                    if sp.id_to_piece(i).startswith("<") and sp.id_to_piece(i).endswith(">")}
        ids = [i for i in ids if i not in specials]
        return sp.DecodeIds(ids).strip()

    # Warm-up
    text = run()
    cold_s = time.perf_counter() - t_cold0

    times = []
    for _ in range(args.runs):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = run()
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
