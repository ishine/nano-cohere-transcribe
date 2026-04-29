# nano-cohere-transcribe

Pure-PyTorch inference for **[CohereLabs/cohere-transcribe-03-2026][model]** — a 2B-parameter Conformer encoder + Transformer decoder ASR model covering 14 languages.

`transformers` ships the reference inference code for this model and drags in a heavy import graph (tokenizers, safetensors, accelerate, pretrained-model scaffolding, generate() machinery). `nano-cohere-transcribe` reimplements the forward pass directly in PyTorch with **six dependencies** (`torch`, `numpy`, `soundfile`, `sentencepiece`, `huggingface-hub`, `safetensors`) so you can ship a minimal ASR binary without the `transformers` base image.

Inspired by [nano-parakeet](../nano-parakeet), which does the same trick for NVIDIA Parakeet vs. NeMo.

## Features

- ✅ 14 languages: en, fr, de, es, it, pt, nl, pl, el, ar, ja, zh, vi, ko
- ✅ Greedy autoregressive decoding with self-attn & cross-attn KV cache
- ✅ CUDA-graph decoder step + batched chunk packing — **1.5×–3.6× faster than the native transformers path** (short / long-form, bs=64 → bs=1) and matches the Open ASR Leaderboard's 10.86 % WER on earnings22 within rounding
- ✅ bfloat16 auto-select on Ampere+, fp32 fallback on CPU
- ✅ Long-form audio via automatic energy-based chunking at quiet points
- ✅ Pluggable detokenizer (SentencePiece default, HuggingFace `tokenizers` optional)
- ⚠️ Greedy only — no beam / nucleus / temperature

## Install

The model weights are gated. Accept the license at <https://huggingface.co/CohereLabs/cohere-transcribe-03-2026> and sign in:

```bash
hf auth login
```

Then install with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install nano-cohere-transcribe
```

For audio loading from arbitrary container formats (mp3, m4a, …) you also want `ffmpeg` on `$PATH` (`brew install ffmpeg` / `apt install ffmpeg`).

### Development setup

```bash
git clone https://github.com/Deep-unlearning/nano-cohere-transcribe
cd nano-cohere-transcribe
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
# optional, benchmarks only:
uv pip install "transformers>=5.5.1" librosa jiwer whisper-normalizer datasets pyarrow
```

## Use

CLI:

```bash
python -m nano_cohere_transcribe audio.wav --language en
```

Python:

```python
from nano_cohere_transcribe import from_pretrained
from nano_cohere_transcribe.audio import load_audio_16k_mono

model = from_pretrained("CohereLabs/cohere-transcribe-03-2026", device="cuda")
waveform = load_audio_16k_mono("audio.wav")     # torch.float32 [num_samples]
text = model.transcribe(waveform, language="en")
print(text)
```

## Benchmark

All runs on a single **NVIDIA A100-80GB**, bf16, greedy decoding. WER is computed after Whisper's `EnglishTextNormalizer` (the Open ASR Leaderboard's normalizer) — lowercase, strip punctuation, drop disfluencies, normalize numbers/contractions/currency/dates.

Both impls share the same greedy decoder, the same 35-second energy-based chunker for long audio, and the same bf16 weights. The differences measured below come from (a) nano's inline KV-cache, (b) chunk-level batch packing that doesn't re-enter the transformers `generate()` machinery per chunk, and (c) CUDA-graph capture of the per-step decoder forward (auto-disabled at chunk-batch ≥ 16 since per-shape capture overhead outweighs the win once launch is amortized across a large batch).

### Short-form: `hf-audio/open-asr-leaderboard` → earnings22

**Full test set** — 2,741 clips, 325.7 min (5.43 h) of audio, `batch_size=64`:

| impl                          | wall        | RTFx       | WER         |
| ----------------------------- | ----------- | ---------- | ----------- |
| transformers 5.5.4 (native)   | 36.8 s      | 530.3×     | **10.82 %** |
| **nano-cohere-transcribe**    | **24.7 s**  | **791.4×** | **10.82 %** |

WER matches the [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) (10.86 %) within rounding for both impls, confirming the benchmark methodology. Nano is **1.49× faster** at bs=64 with **byte-identical aggregate WER**. (CUDA graphs are auto-disabled at `B > 16` since per-shape capture overhead outweighs the win when launch is already amortized across a large batch.)

### Long-form: `hf-audio/asr-leaderboard-longform` → earnings21

**Full test set** — 44 clips, 2355.8 min (39.3 h) of audio. Each clip is ~55 min; both impls pre-chunk at quiet points (35 s max), so here `batch_size` means chunks-per-generate (nano) / clips-per-generate (transformers native).

#### batch_size = 8

| impl                          | wall                 | RTFx       | WER        |
| ----------------------------- | -------------------- | ---------- | ---------- |
| transformers 5.5.4 (native)   | 720.8 s (12.0 min)   | 196.1×     | **8.68 %** |
| **nano-cohere-transcribe**    | **224.0 s (3.7 min)** | **631.5×** | 8.73 %     |

Nano is **3.22× faster** with **+0.05 pp WER** — within rounding. The win comes from CUDA-graph capture of the per-step decoder forward (one GPU dispatch per token instead of hundreds of small kernel launches).

#### batch_size = 1

| impl                          | wall                  | RTFx        | WER        |
| ----------------------------- | --------------------- | ----------- | ---------- |
| transformers 5.5.4 (native)   | 3681.1 s (61.4 min)   | 38.4×       | **8.68 %** |
| **nano-cohere-transcribe**    | **1017.8 s (17.0 min)** | **138.6×** | 8.72 %     |

Nano is **3.62× faster** at bs=1 — the autoregressive loop is dominated by per-step kernel launch overhead, exactly what CUDA graph replay erases. Both impls process chunks serially at bs=1, but nano replays a single graph dispatch per token while transformers issues hundreds of kernel launches per step. WER is unchanged between bs=1 and bs=8 (8.72% vs 8.73%) — batching is a throughput lever, not a quality one.

### Notes on the transformers path

- **Native transformers only** (`transformers==5.5.4`, no `trust_remote_code`). The model's shipped remote-code `CohereAsrTokenizer` character-BPEs the `<|...|>` control tokens on `transformers>=5.4`, and the remote-code `.transcribe()` has an O(n²) Python detokenization hot-path that caps throughput at ~9× RTFx.
- **External chunking for long audio** — the native Conformer encoder does not ship the remote-code version's `_conv_split_by_batch` workaround for PyTorch's int32 CUDA indexing limit. A single 55-min clip's encoder tensor blows past 2^31 elements. We pre-split with nano's energy chunker and feed chunk batches through the native processor/generate.
- vLLM (0.19.1) was also evaluated on short-form: RTFx 118.8× on 2 samples. Its shipped `cohere_asr` adapter has the same control-token bug (builds prompt as text → fast tokenizer → garbage); a one-line patch to return `TokensPrompt(prompt_token_ids=…)` fixes it (see `benchmark_vllm.py`). Removed from the main table since it's an additional ~2.5× slower than nano and requires a separate env.

### Reproducing

```bash
# Short-form full
python benchmark_datasets.py earnings22-open --num-samples 0 --batch-size 64

# Long-form full, bs=8 and bs=1
python benchmark_datasets.py earnings21-lb --num-samples 0 --batch-size 8
python benchmark_datasets.py earnings21-lb --num-samples 0 --batch-size 1

# Nano only (skip heavy transformers half)
python benchmark_datasets.py earnings21-lb --num-samples 0 --batch-size 8 --skip-transformers
```

### Batched transcription from Python

```python
from nano_cohere_transcribe import from_pretrained
from nano_cohere_transcribe.audio import load_audio_16k_mono

model = from_pretrained("CohereLabs/cohere-transcribe-03-2026", device="cuda")
wavs = [load_audio_16k_mono(p) for p in ["a.wav", "b.wav", "c.wav"]]
texts = model.transcribe_batch(wavs, language="en", batch_size=8)
```

Long clips in the batch are auto-chunked internally; all chunks across all inputs are sorted longest-first and packed into batches of `batch_size` for efficient padding.

### Fast vs. SentencePiece detokenizer

Nano ships two interchangeable detokenizer backends:

```python
from_pretrained(..., decoder_tokenizer="sentencepiece")   # default, C++ SP
from_pretrained(..., decoder_tokenizer="fast")            # HF Rust tokenizers
```

Both produce byte-identical output. SP is the bundled default (one fewer runtime dep); `"fast"` is useful when slotting nano into an HF pipeline. Encoding the prompt always goes through SentencePiece directly — the HF fast tokenizer character-BPEs the `<|...|>` control tokens on `transformers>=5.4`, so we can't use it there.

## Tests

```bash
pytest tests/ -v -m 'not slow'     # fast offline subset (<30 s)
pytest tests/ -v                   # also runs the short end-to-end test (~30 s model load)
pytest tests/ -v -m slow           # opt-in heavy tests: 36-min long-form + batched regression (~90 s on A100)
```

## License

Apache 2.0. Model weights are distributed under their own Apache 2.0 license by Cohere Labs.

[model]: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
