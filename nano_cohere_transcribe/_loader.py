"""Load a Cohere ASR checkpoint into our nano module tree.

The upstream state_dict keys already match our module naming one-for-one
(by design — see ``model.py``), so no remapping is needed. We just:

1. Download / resolve the snapshot directory via ``huggingface_hub``.
2. Read ``config.json`` and build a :class:`CohereAsrConfig`.
3. Load ``model.safetensors`` and pop:
   - ``log_softmax.mlp.layer0.weight`` (tied to decoder embedding; avoids a warn)
   - ``encoder.layers.*.conv.batch_norm.num_batches_tracked`` (unused at eval)
4. ``load_state_dict(state_dict, strict=False)`` to accept the
   ``preprocessor.featurizer.{fb,window}`` buffers even though they're
   non-persistent.

Returns the instantiated :class:`CohereAsr`, the :class:`CohereTokenizer`,
and the path to the ``tokenizer.model`` file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as safetensors_load_file

from .model import CohereAsr, CohereAsrConfig
from .tokenizer import CohereTokenizer

# Files we need from the HF repo.
_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.model",
)


def resolve_snapshot(repo_id_or_path: str) -> Path:
    """Return a local directory containing the required files.

    Accepts either a HuggingFace repo id or a local directory.
    """
    p = Path(repo_id_or_path)
    if p.is_dir():
        missing = [f for f in _REQUIRED_FILES if not (p / f).exists()]
        if not missing:
            return p
    local = snapshot_download(
        repo_id=repo_id_or_path,
        allow_patterns=list(_REQUIRED_FILES),
    )
    return Path(local)


def load_model_from_snapshot(
    snapshot_dir: Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    decoder_tokenizer: str = "sentencepiece",
) -> tuple[CohereAsr, CohereTokenizer]:
    with open(snapshot_dir / "config.json") as f:
        raw_cfg = json.load(f)
    cfg = CohereAsrConfig.from_hf_config(raw_cfg)

    state = safetensors_load_file((snapshot_dir / "model.safetensors").as_posix())

    # Split out preprocessor buffers; they're stored in the checkpoint but
    # their home module (`FilterbankFeatures`) keeps them as non-persistent
    # buffers so `load_state_dict` would ignore/error on them.
    fb = state.pop("preprocessor.featurizer.fb", None)
    window = state.pop("preprocessor.featurizer.window", None)

    # Drop BatchNorm running_batches_tracked (int64 scalars) — not needed at eval.
    for k in list(state.keys()):
        if k.endswith("num_batches_tracked"):
            state.pop(k)

    # Drop tied head weight redundancy. The checkpoint stores both
    # `log_softmax.mlp.layer0.weight` and `transf_decoder._embedding.token_embedding.weight`
    # with the same values; we keep only the embedding side and re-tie in __init__.
    state.pop("log_softmax.mlp.layer0.weight", None)

    model = CohereAsr(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # Re-tie after load (load_state_dict may have broken the alias on older torch).
    model.log_softmax.mlp.layer0.weight = model.transf_decoder._embedding.token_embedding.weight

    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:8]} (...)")
    # "missing" should only contain the two non-persistent mel buffers and the
    # re-tied log_softmax.mlp.layer0.weight; anything else is a bug.
    allowed_missing = {
        "preprocessor.featurizer.fb",
        "preprocessor.featurizer.window",
        "log_softmax.mlp.layer0.weight",
    }
    unexpected_missing = [k for k in missing if k not in allowed_missing]
    if unexpected_missing:
        raise RuntimeError(f"Missing keys not accounted for: {unexpected_missing[:8]} (...)")

    # Install preprocessor buffers.
    if fb is not None:
        model.preprocessor.featurizer.fb = fb.squeeze(0) if fb.dim() == 3 else fb
        # Reference stores [1, n_mels, n_fft//2+1]; matmul expects same layout.
        if model.preprocessor.featurizer.fb.dim() == 2:
            model.preprocessor.featurizer.fb = model.preprocessor.featurizer.fb.unsqueeze(0)
    if window is not None:
        model.preprocessor.featurizer.window = window

    # Dtype + device.
    model.eval()
    if dtype is None:
        dtype = _autoselect_dtype(device)
    model = model.to(device=torch.device(device), dtype=dtype)
    # Keep BatchNorm running stats in fp32 for numerical stability.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean = m.running_mean.float()
            m.running_var = m.running_var.float()

    tokenizer = CohereTokenizer(
        (snapshot_dir / "tokenizer.model").as_posix(),
        decoder=decoder_tokenizer,
        snapshot_dir=snapshot_dir,
    )
    return model, tokenizer


def _autoselect_dtype(device: str | torch.device) -> torch.dtype:
    dev = torch.device(device)
    if dev.type == "cuda":
        major = torch.cuda.get_device_capability(dev)[0]
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32
