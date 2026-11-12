"""Capture an owned-checkpoint smoke manifest without a Hugging Face model forward."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from pagedcore.cli import collect_environment_diagnostics, collect_validation_provenance
from pagedcore.config import MODEL_ID, MODEL_REVISION
from pagedcore.decoder import prefill
from pagedcore.oracle import reference_math
from pagedcore.weights import load_weights


def verify() -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("verification requires one CUDA-visible T4")
    if "t4" not in torch.cuda.get_device_name(0).casefold():
        raise ValueError("verification requires an NVIDIA T4")
    fixtures = json.loads(
        (Path(__file__).resolve().parents[1] / "tests/fixtures/reference_prompts.json").read_text()
    )
    settings = fixtures["tokenizer"]
    if (settings["model_id"], settings["revision"]) != (MODEL_ID, MODEL_REVISION):
        raise ValueError("tokenizer fixtures must use the pinned checkpoint")
    for name, expected in settings["source_sha256"].items():
        path = Path(hf_hub_download(MODEL_ID, name, revision=MODEL_REVISION))
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"pinned tokenizer file hash mismatch: {name}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if type(tokenizer).__name__ != settings["tokenizer_class"]:
        raise ValueError("pinned tokenizer class mismatch")
    for name in ("add_bos_token", "add_eos_token", "bos_token_id", "eos_token_id", "vocab_size"):
        if getattr(tokenizer, name) != settings[name]:
            raise ValueError(f"pinned tokenizer setting mismatch: {name}")
    for fixture in fixtures["prompts"]:
        encoded = tokenizer(fixture["prompt"], add_special_tokens=True, truncation=False)
        if encoded.input_ids != fixture["input_ids"]:
            raise ValueError(f"pinned tokenizer IDs mismatch: {fixture['name']}")
    ids = tokenizer("PagedCore target verification.", return_tensors="pt")["input_ids"].to("cuda:0")
    with reference_math() as math_settings, torch.inference_mode():
        weights = load_weights(device="cuda:0")
        logits, cache = prefill(ids, weights)
        if logits.shape != (*ids.shape, 32000) or logits.dtype != torch.float16:
            raise ValueError("owned checkpoint returned an invalid logit shape or dtype")
        if not bool(logits.isfinite().all()) or cache.length != ids.shape[-1]:
            raise ValueError("owned checkpoint returned non-finite logits or an invalid cache")
        torch.cuda.synchronize()
    return {
        "manifest_version": 3,
        "status": "passed",
        "scope": "Owned checkpoint and tokenizer smoke; no external correctness comparison",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer": settings,
        "math_settings": math_settings,
        "environment": collect_environment_diagnostics(),
        "provenance": collect_validation_provenance(),
        "forward_pass": {"input_tokens": ids.shape[-1], "logits_shape": list(logits.shape)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(verify(), indent=2, allow_nan=False) + "\n")
