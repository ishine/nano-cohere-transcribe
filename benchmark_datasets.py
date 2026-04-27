#!/usr/bin/env python3
"""Real-world long-form benchmark on earnings-call datasets.

Supports:

* ``earnings22-aa``   — ``ArtificialAnalysis/Earnings22-Cleaned-AA`` (6 × 15-20 min mp3, with cleaned refs)
* ``earnings22-lb``   — ``hf-audio/asr-leaderboard-longform`` earnings22 subset (140 × ~60 min parquet rows)

For each selected sample we run the audio through both ``nano-cohere-transcribe`` and the
``transformers`` reference (``model.transcribe()``), measure wall time, and compute WER
against the ground-truth transcript with jiwer.

Example:
    python benchmark_datasets.py earnings22-aa --num-samples 2
    python benchmark_datasets.py earnings22-lb --num-samples 1 --skip-transformers
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from nano_cohere_transcribe import from_pretrained as nano_from_pretrained
from nano_cohere_transcribe.audio import load_audio_16k_mono

AA_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA"
LB_REPO = "hf-audio/asr-leaderboard-longform"
OASR_REPO = "hf-audio/open-asr-leaderboard"  # ESB test sets, short clips


@dataclass
class Sample:
    sample_id: str
    audio: np.ndarray        # float32, 16 kHz mono
    reference: str
    duration_s: float


# ---------- dataset loaders ----------


def _effective_limit(num_samples: int, hard_cap: int) -> int:
    """``num_samples <= 0`` means the entire dataset (capped at hard_cap)."""
    return hard_cap if num_samples <= 0 else min(num_samples, hard_cap)


def load_aa(num_samples: int, max_duration_s: float | None) -> list[Sample]:
    jsonl_path = hf_hub_download(AA_REPO, "earnings22_cleaned_aa_v1.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    samples: list[Sample] = []
    for row in rows[:num_samples]:
        audio_path = hf_hub_download(AA_REPO, row["url"], repo_type="dataset")
        audio = load_audio_16k_mono(audio_path).numpy()
        if max_duration_s is not None:
            audio = audio[: int(max_duration_s * 16000)]
        samples.append(
            Sample(
                sample_id=row["id"],
                audio=audio,
                reference=row["transcript"],
                duration_s=len(audio) / 16000.0,
            )
        )
    return samples


def _load_lb_subset(subset: str, num_shards: int, num_samples: int,
                    max_duration_s: float | None) -> list[Sample]:
    """Load N samples from one subset of hf-audio/asr-leaderboard-longform."""
    samples: list[Sample] = []
    shard_idx = 0
    limit = _effective_limit(num_samples, hard_cap=10_000)
    while len(samples) < limit and shard_idx < num_shards:
        path = hf_hub_download(
            LB_REPO, f"{subset}/test-{shard_idx:05d}-of-{num_shards:05d}.parquet",
            repo_type="dataset",
        )
        shard_idx += 1
        table = pq.read_table(path)
        for row in table.to_pylist():
            if len(samples) >= limit:
                break
            audio_dict = row["audio"]
            if not audio_dict or not audio_dict.get("bytes"):
                continue
            audio, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
            if max_duration_s is not None:
                audio = audio[: int(max_duration_s * 16000)]
            sid = Path(audio_dict.get("path") or "").stem or f"{subset}_{shard_idx-1}_{len(samples)}"
            samples.append(Sample(
                sample_id=sid, audio=audio, reference=row["text"],
                duration_s=len(audio) / 16000.0,
            ))
    return samples


def load_lb_earnings22(num_samples: int, max_duration_s: float | None) -> list[Sample]:
    return _load_lb_subset("earnings22", 28, num_samples, max_duration_s)


def load_lb_earnings21(num_samples: int, max_duration_s: float | None) -> list[Sample]:
    return _load_lb_subset("earnings21", 10, num_samples, max_duration_s)


# ---------- text normalization for WER ----------


_whisper_normalizer = None


def _normalize(text: str) -> str:
    """Run Whisper's ``EnglishTextNormalizer``, the normalizer used by the Open
    ASR Leaderboard (https://huggingface.co/spaces/hf-audio/open_asr_leaderboard).

    It lowercases, strips punctuation, removes common fillers (uh/um/hmm),
    normalizes contractions/numbers/currency/dates, and collapses whitespace —
    so WER numbers here are directly comparable to the leaderboard.
    """
    global _whisper_normalizer
    if _whisper_normalizer is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _whisper_normalizer = EnglishTextNormalizer()
    return _whisper_normalizer(text)


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer
    return jiwer.wer([_normalize(r) for r in refs], [_normalize(h) for h in hyps])


# ---------- runners ----------


def run_nano(samples: list[Sample], model, language: str, batch_size: int) -> tuple[list[str], float]:
    # Match the leaderboard-style HF path: punctuation=False for English
    # (`<|nopnc|>`), True otherwise. Different control tokens steer the model
    # toward different text styles; on long audio the divergence compounds over
    # many chunks and shows up as a WER gap even after Whisper normalization.
    punctuation = language != "en"
    model.transcribe(
        torch.from_numpy(samples[0].audio[:16000]), language=language, punctuation=punctuation,
    )  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    hyps = model.transcribe_batch(
        [torch.from_numpy(s.audio) for s in samples],
        language=language, punctuation=punctuation, batch_size=batch_size,
    )
    torch.cuda.synchronize()
    return hyps, time.perf_counter() - t0


def run_hf(
    samples: list[Sample], language: str, device: str, model_repo: str, batch_size: int
) -> tuple[list[str], float]:
    """Leaderboard-style transformers benchmark path.

    Uses transformers-native ``CohereAsrForConditionalGeneration`` +
    ``CohereAsrProcessor`` (``transformers >= 5.5``). For long clips we
    pre-chunk via the same energy splitter nano uses, because the native
    Parakeet/Conformer encoder does not have the remote-code version's
    ``_conv_split_by_batch`` workaround for PyTorch's int32 CUDA indexing limit
    (a single 55-minute clip's encoder activation otherwise exceeds 2^31 floats
    and the forward pass hard-errors).

    The processor handles short clips natively via ``language``/``punctuation``
    kwargs. For long clips we short-circuit: split externally, then rebuild per-
    sample text from ``join_chunk_texts``.
    """
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration

    from nano_cohere_transcribe.chunk import (
        get_chunk_separator, join_chunk_texts, split_audio_chunks_energy,
    )

    processor = AutoProcessor.from_pretrained(model_repo)
    hf_model = CohereAsrForConditionalGeneration.from_pretrained(
        model_repo, dtype=torch.bfloat16, device_map=device,
    )
    hf_model.eval()
    max_audio_s = float(hf_model.config.max_audio_clip_s)
    punctuation = language != "en"

    def _gen_batch(audios: list[np.ndarray]) -> list[str]:
        """Short-form batch: feed ``batch_size`` clips directly to the processor,
        generate, decode. Each clip in ``audios`` must fit in one chunk."""
        if not audios:
            return []
        out = [""] * len(audios)
        for i in range(0, len(audios), batch_size):
            sub = audios[i : i + batch_size]
            inputs = processor(
                sub, sampling_rate=16000, return_tensors="pt",
                language=language, punctuation=punctuation,
            )
            inputs.to(hf_model.device, dtype=hf_model.dtype)
            audio_chunk_index = inputs.get("audio_chunk_index")
            with torch.inference_mode():
                outputs = hf_model.generate(**inputs, max_new_tokens=256)
            texts = processor.decode(
                outputs, skip_special_tokens=True,
                audio_chunk_index=audio_chunk_index, language=language,
            )
            for j, t in enumerate(texts):
                out[i + j] = t.strip() if isinstance(t, str) else str(t).strip()
        return out

    def _transcribe_one(audio: np.ndarray) -> str:
        """Single-sample transcribe with external chunking for long audio."""
        if audio.shape[0] / 16000.0 <= max_audio_s - 5.0:
            return _gen_batch([audio])[0]
        chunks = split_audio_chunks_energy(
            audio, sample_rate=16000, max_audio_clip_s=max_audio_s,
            overlap_chunk_second=5.0, min_energy_window_samples=1600,
        )
        texts = _gen_batch(chunks)
        return join_chunk_texts(texts, separator=get_chunk_separator(language))

    # Warmup.
    _gen_batch([samples[0].audio[:16000]])
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    hyps: list[str] = []
    for s in samples:
        hyps.append(_transcribe_one(s.audio))
    torch.cuda.synchronize()
    return hyps, time.perf_counter() - t0


# ---------- main ----------


def load_open_asr_earnings22(num_samples: int, max_duration_s: float | None) -> list[Sample]:
    """open-asr-leaderboard earnings22 — short clips (~15-35 s each) with refs.

    The shards are sorted by audio length descending, so the first shard contains
    the longest clips. Using shard 0 gives the worst-case batching workload for
    a fair comparison.
    """
    samples: list[Sample] = []
    shard_idx = 0
    limit = _effective_limit(num_samples, hard_cap=10_000)
    while len(samples) < limit and shard_idx < 5:
        path = hf_hub_download(
            OASR_REPO, f"earnings22/test-{shard_idx:05d}-of-00005.parquet", repo_type="dataset"
        )
        shard_idx += 1
        table = pq.read_table(path)
        for row in table.to_pylist():
            if len(samples) >= limit:
                break
            audio_dict = row["audio"]
            if not audio_dict or not audio_dict.get("bytes"):
                continue
            audio, sr = sf.read(io.BytesIO(audio_dict["bytes"]))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
            if max_duration_s is not None:
                audio = audio[: int(max_duration_s * 16000)]
            samples.append(Sample(
                sample_id=str(row.get("id", f"oasr_{len(samples)}")),
                audio=audio,
                reference=row["text"],
                duration_s=len(audio) / 16000.0,
            ))
    return samples


DATASETS = {
    "earnings22-aa": load_aa,
    "earnings22-lb": load_lb_earnings22,
    "earnings21-lb": load_lb_earnings21,
    "earnings22-open": load_open_asr_earnings22,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=sorted(DATASETS.keys()))
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--max-duration-s", type=float, default=None,
                   help="Optional: truncate each sample to this many seconds (for fast smoke tests).")
    p.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-nano", action="store_true")
    p.add_argument("--skip-transformers", action="store_true")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Chunk batch size (both impls; default 8).")
    args = p.parse_args()

    print(f"=== {args.dataset}  ({args.num_samples} samples) ===\n")
    loader = DATASETS[args.dataset]
    samples = loader(args.num_samples, args.max_duration_s)
    total_audio_s = sum(s.duration_s for s in samples)
    print(f"Loaded {len(samples)} samples, total audio: {total_audio_s/60:.1f} min")
    for s in samples:
        print(f"  - {s.sample_id}  {s.duration_s/60:.1f} min  (ref chars={len(s.reference)})")
    print()

    refs = [s.reference for s in samples]
    # Persistent hypothesis log so we can re-score with a different normalizer
    # later without re-running the full benchmark.
    results_dir = Path(__file__).parent / ".bench_out"
    results_dir.mkdir(exist_ok=True)
    tag = f"{args.dataset}_n{args.num_samples}_bs{args.batch_size}"

    nano_hyps, nano_dt = (None, None)
    hf_hyps, hf_dt = (None, None)

    if not args.skip_nano:
        print(f"--- nano-cohere-transcribe (bs={args.batch_size}) ---")
        nano_model = nano_from_pretrained(args.model, device=args.device)
        nano_hyps, nano_dt = run_nano(samples, nano_model, args.language, args.batch_size)
        with open(results_dir / f"{tag}_nano.json", "w") as f:
            json.dump({"wall_s": nano_dt, "refs": refs, "hyps": nano_hyps,
                       "ids": [s.sample_id for s in samples],
                       "durations_s": [s.duration_s for s in samples]}, f)
        rtf = total_audio_s / nano_dt
        nano_wer = compute_wer(refs, nano_hyps)
        print(f"wall={nano_dt:.1f}s  RTFx={rtf:.1f}x  WER={nano_wer:.4f}")
        print()

    if not args.skip_transformers:
        print(f"--- transformers reference (bs={args.batch_size}) ---")
        hf_hyps, hf_dt = run_hf(samples, args.language, args.device, args.model, args.batch_size)
        with open(results_dir / f"{tag}_hf.json", "w") as f:
            json.dump({"wall_s": hf_dt, "refs": refs, "hyps": hf_hyps,
                       "ids": [s.sample_id for s in samples],
                       "durations_s": [s.duration_s for s in samples]}, f)
        rtf = total_audio_s / hf_dt
        hf_wer = compute_wer(refs, hf_hyps)
        print(f"wall={hf_dt:.1f}s  RTFx={rtf:.1f}x  WER={hf_wer:.4f}")
        print()

    # ---- summary ----
    print("=== Summary ===")
    hdr = f"{'':20} {'transformers':>15} {'nano-cohere':>15} {'Speedup':>10}"
    print(hdr)
    print("-" * len(hdr))

    def fmt(v, spec, suf=""):
        return f"{v:{spec}}{suf}" if v is not None else "N/A"

    nano_rtf = (total_audio_s / nano_dt) if nano_dt else None
    hf_rtf = (total_audio_s / hf_dt) if hf_dt else None
    sp = (nano_rtf / hf_rtf) if (nano_rtf and hf_rtf) else None

    print(f"{'Total audio':20} {fmt(total_audio_s/60,'>14.1f','m')}")
    print(f"{'Wall time':20} {fmt(hf_dt,'>14.1f','s')} {fmt(nano_dt,'>14.1f','s')}")
    print(f"{'RTFx':20} {fmt(hf_rtf,'>14.1f','x')} {fmt(nano_rtf,'>14.1f','x')} {fmt(sp,'>9.2f','x')}")
    if hf_hyps is not None and not args.skip_transformers:
        print(f"{'WER (transformers)':20} {compute_wer(refs, hf_hyps):>14.4f}")
    if nano_hyps is not None and not args.skip_nano:
        print(f"{'WER (nano)':20} {'':>15} {compute_wer(refs, nano_hyps):>14.4f}")

    # Per-sample detail
    if nano_hyps or hf_hyps:
        print()
        print("Per-sample WER:")
        for i, s in enumerate(samples):
            parts = [f"  {s.sample_id}  ({s.duration_s/60:.1f} min)"]
            if hf_hyps is not None:
                parts.append(f"hf_wer={compute_wer([s.reference], [hf_hyps[i]]):.4f}")
            if nano_hyps is not None:
                parts.append(f"nano_wer={compute_wer([s.reference], [nano_hyps[i]]):.4f}")
            print("  ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
