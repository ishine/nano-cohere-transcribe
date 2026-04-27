#!/usr/bin/env python3
"""Long-form side-by-side benchmark.

Concatenates a short clip N times to build a multi-minute audio stream, then
transcribes it with both ``nano_cohere_transcribe`` and the transformers
reference (``CohereAsrForConditionalGeneration.transcribe``). Reports wall-time,
RTFx, chunk count, and a crude similarity metric.

Requires transformers==5.3.0 (see benchmark_hf.py docstring for why).
"""
from __future__ import annotations

import argparse
import difflib
import re
import time

import numpy as np
import torch

from nano_cohere_transcribe import from_pretrained as nano_from_pretrained
from nano_cohere_transcribe.audio import load_audio_16k_mono
from nano_cohere_transcribe.chunk import split_audio_chunks_energy


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="Short clip to loop (wav/mp3/...).")
    p.add_argument("--repeats", type=int, default=180, help="Copies to concatenate (default: 180).")
    p.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-transformers", action="store_true")
    p.add_argument("--skip-nano", action="store_true")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Chunk batch size for long-form decode (both impls use this).")
    args = p.parse_args()

    base = load_audio_16k_mono(args.audio).numpy()
    audio = np.tile(base, args.repeats)
    duration_s = audio.shape[0] / 16000.0

    chunks = split_audio_chunks_energy(
        audio, sample_rate=16000, max_audio_clip_s=35.0,
        overlap_chunk_second=5.0, min_energy_window_samples=1600,
    )
    print(f"Audio: {duration_s/60:.1f} min ({len(audio)} samples)")
    print(f"Chunks: {len(chunks)} (energy-split, <=35 s each)")
    print()

    nano_text = hf_text = None
    nano_dt = hf_dt = None

    if not args.skip_nano:
        print(f"--- nano-cohere-transcribe (bs={args.batch_size}) ---")
        model = nano_from_pretrained(args.model, device=args.device)
        wave_t = torch.from_numpy(audio)
        model.transcribe(wave_t[:16000], language=args.language)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        nano_text = model.transcribe(wave_t, language=args.language, batch_size=args.batch_size)
        torch.cuda.synchronize()
        nano_dt = time.perf_counter() - t0
        print(f"time={nano_dt:.1f}s  RTFx={duration_s/nano_dt:.1f}x  chars={len(nano_text)}")
        print()

    if not args.skip_transformers:
        print(f"--- transformers reference (bs={args.batch_size}) ---")
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            args.model, trust_remote_code=True, device_map=args.device, torch_dtype=torch.bfloat16
        )
        hf_model.eval()
        hf_model.transcribe(
            processor, language=args.language, audio_arrays=[audio[:16000]],
            sample_rates=[16000], batch_size=1,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hf_text = hf_model.transcribe(
            processor, language=args.language, audio_arrays=[audio],
            sample_rates=[16000], batch_size=args.batch_size,
        )[0]
        torch.cuda.synchronize()
        hf_dt = time.perf_counter() - t0
        print(f"time={hf_dt:.1f}s  RTFx={duration_s/hf_dt:.1f}x  chars={len(hf_text)}")
        print()

    print("=== Summary ===")
    print(f"{'':20} {'transformers':>20} {'nano-cohere':>20} {'Speedup':>10}")
    print("-" * 73)
    def fmt(v, spec, suf=""):
        return f"{v:{spec}}{suf}" if v is not None else "N/A"
    nano_rtf = duration_s/nano_dt if nano_dt else None
    hf_rtf = duration_s/hf_dt if hf_dt else None
    sp = (nano_rtf/hf_rtf) if (nano_rtf and hf_rtf) else None
    print(f'{"RTFx":20} {fmt(hf_rtf,">19.1f","x")} {fmt(nano_rtf,">19.1f","x")} {fmt(sp,">9.2f","x")}')
    print(f'{"Wall time":20} {fmt(hf_dt,">19.1f","s")} {fmt(nano_dt,">19.1f","s")}')
    print(f'{"Audio":20} {fmt(duration_s/60,">19.1f","m")}')
    if nano_text and hf_text:
        ratio = difflib.SequenceMatcher(None, _normalize(hf_text), _normalize(nano_text)).ratio()
        print(f'{"Text similarity":20} {ratio:>19.4f} (1.0 = identical after whitespace/case norm)')
        print(f'{"HF chars":20} {fmt(len(hf_text),">20d")}')
        print(f'{"nano chars":20} {fmt(len(nano_text),">20d")}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
