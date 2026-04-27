#!/usr/bin/env python3
"""Benchmark the vLLM serving path for Cohere Transcribe.

Talks to a locally-running ``vllm serve`` instance via the OpenAI-compatible
``/v1/audio/transcriptions`` endpoint. Mirrors benchmark_datasets.py: same
samples, same WER computation, so numbers go in the same README table.

Prerequisite: start the server in a *separate* env (vLLM pins a lot of
dependencies that would conflict with the nano-cohere training env):

    conda create -n nano-cohere-vllm python=3.11 -y
    conda activate nano-cohere-vllm
    pip install "vllm[audio]" librosa
    vllm serve CohereLabs/cohere-transcribe-03-2026 --trust-remote-code \
        --dtype bfloat16 --port 8765

Then, from any env that has jiwer + soundfile, run:

    python benchmark_vllm.py earnings22-aa --num-samples 2 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download

try:
    import aiohttp
except ImportError as e:  # pragma: no cover
    raise SystemExit("aiohttp required: pip install aiohttp") from e

AA_REPO = "ArtificialAnalysis/Earnings22-Cleaned-AA"
LB_REPO = "hf-audio/asr-leaderboard-longform"


@dataclass
class Sample:
    sample_id: str
    audio: np.ndarray
    reference: str
    duration_s: float


# ---------- dataset loaders (copy-paste parity with benchmark_datasets.py) ----------


def load_aa(num_samples: int, max_duration_s: float | None) -> list[Sample]:
    jsonl_path = hf_hub_download(AA_REPO, "earnings22_cleaned_aa_v1.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    out: list[Sample] = []
    for row in rows[:num_samples]:
        audio_path = hf_hub_download(AA_REPO, row["url"], repo_type="dataset")
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
        if max_duration_s is not None:
            audio = audio[: int(max_duration_s * 16000)]
        out.append(Sample(
            sample_id=row["id"], audio=audio, reference=row["transcript"],
            duration_s=len(audio) / 16000.0,
        ))
    return out


def load_lb_earnings22(num_samples: int, max_duration_s: float | None) -> list[Sample]:
    out: list[Sample] = []
    shard = 0
    while len(out) < num_samples and shard < 28:
        path = hf_hub_download(
            LB_REPO, f"earnings22/test-{shard:05d}-of-00028.parquet", repo_type="dataset"
        )
        shard += 1
        table = pq.read_table(path)
        for row in table.to_pylist():
            if len(out) >= num_samples:
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
            sid = Path(audio_dict.get("path") or "").stem or f"lb_{shard-1}_{len(out)}"
            out.append(Sample(
                sample_id=sid, audio=audio, reference=row["text"],
                duration_s=len(audio) / 16000.0,
            ))
    return out


DATASETS = {"earnings22-aa": load_aa, "earnings22-lb": load_lb_earnings22}


# ---------- WER ----------


def _normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer
    return jiwer.wer([_normalize(r) for r in refs], [_normalize(h) for h in hyps])


# ---------- vLLM client ----------


def audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Encode float32 mono audio as a 16-bit PCM WAV in-memory."""
    buf = io.BytesIO()
    pcm16 = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


async def transcribe_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    audio: np.ndarray,
    language: str,
    sample_id: str,
) -> tuple[str, float]:
    form = aiohttp.FormData()
    form.add_field("file", audio_to_wav_bytes(audio), filename=f"{sample_id}.wav",
                   content_type="audio/wav")
    form.add_field("model", model)
    form.add_field("language", language)
    t0 = time.perf_counter()
    async with session.post(endpoint, data=form, timeout=aiohttp.ClientTimeout(total=3600)) as r:
        r.raise_for_status()
        payload = await r.json()
    return payload.get("text", ""), time.perf_counter() - t0


async def run(samples: list[Sample], endpoint: str, model: str, language: str,
              concurrency: int) -> tuple[list[str], float]:
    sem = asyncio.Semaphore(concurrency)

    async def one(idx: int, s: Sample) -> tuple[int, str, float]:
        async with sem:
            text, dt = await transcribe_one(session, endpoint, model, s.audio, language, s.sample_id)
            return idx, text, dt

    conn = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=conn) as session:
        t0 = time.perf_counter()
        tasks = [one(i, s) for i, s in enumerate(samples)]
        results = await asyncio.gather(*tasks)
        total = time.perf_counter() - t0
    hyps = [""] * len(samples)
    for i, text, _ in results:
        hyps[i] = text
    return hyps, total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=sorted(DATASETS.keys()))
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--max-duration-s", type=float, default=None)
    p.add_argument("--endpoint", default="http://localhost:8765/v1/audio/transcriptions")
    p.add_argument("--model", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--language", default="en")
    p.add_argument("--concurrency", type=int, default=8,
                   help="Max concurrent requests to the vLLM server.")
    args = p.parse_args()

    print(f"=== vLLM benchmark: {args.dataset}  ({args.num_samples} samples, concurrency={args.concurrency}) ===\n")
    samples = DATASETS[args.dataset](args.num_samples, args.max_duration_s)
    total_audio_s = sum(s.duration_s for s in samples)
    print(f"Loaded {len(samples)} samples, total audio: {total_audio_s/60:.1f} min")
    for s in samples:
        print(f"  - {s.sample_id}  {s.duration_s/60:.1f} min")
    print()

    refs = [s.reference for s in samples]

    # Warmup: one-shot 1-second silence so we don't include the first-request
    # compilation overhead in the measured wall time.
    async def warmup():
        async with aiohttp.ClientSession() as session:
            await transcribe_one(
                session, args.endpoint, args.model,
                np.zeros(16000, dtype=np.float32), args.language, "warmup",
            )
    asyncio.run(warmup())

    print(f"--- vLLM (concurrency={args.concurrency}) ---")
    hyps, dt = asyncio.run(run(samples, args.endpoint, args.model, args.language, args.concurrency))
    rtf = total_audio_s / dt
    wer = compute_wer(refs, hyps)
    print(f"wall={dt:.1f}s  RTFx={rtf:.1f}x  WER={wer:.4f}")
    print()

    print("Per-sample WER:")
    for s, h in zip(samples, hyps):
        w = compute_wer([s.reference], [h])
        print(f"  {s.sample_id}  ({s.duration_s/60:.1f} min)  wer={w:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
