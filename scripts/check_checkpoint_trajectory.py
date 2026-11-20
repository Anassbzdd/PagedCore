"""Diagnostic real-weight component checks and a fixed cached trajectory."""

import argparse
import json
import runpy
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from scripts.probe_two_token_decode import ROOT
from scripts.replay_one_page import round_to_fp16

from pagedcore import decoder
from pagedcore.accuracy import compare_logits
from pagedcore.determinism import seed_everything
from pagedcore.engine_types import CapacityCounts, KVReservation, RequestId
from pagedcore.kv_cache import KVAllocator
from pagedcore.oracle import reference_math
from pagedcore.weights import load_weights


def rounded(value):
    return round_to_fp16(value.cpu()).to(value.device)


def norm64(hidden, scale):
    value = hidden.double()
    normalized = value / (value.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    return rounded(rounded(normalized).double() * scale.double())


@contextmanager
def component_audit(records, directory, label):
    originals = {name: getattr(decoder, name) for name in ("rms_norm", "swiglu", "final_logits")}

    def record(name, actual, expected, hidden):
        error = actual.double() - expected.double()
        index = len(records)
        records.append(
            {
                "label": label,
                "component": name,
                "max_absolute_error": float(error.abs().max()),
                "rms_error": float(error.square().mean().sqrt()),
                "note": "Local staged-FP16 definition with FP64 reductions; diagnostic, not a new acceptance rule",
            }
        )
        torch.save(
            {"input": hidden.cpu(), "actual": actual.cpu(), "reference": expected.cpu()},
            directory / f"component-{index:04}.pt",
        )

    def norm(hidden, scale):
        result = originals["rms_norm"](hidden, scale)
        record("rms_norm", result, norm64(hidden, scale), hidden)
        return result

    def mlp(hidden, weights):
        result = originals["swiglu"](hidden, weights)
        gate = rounded(hidden.double() @ weights.gate_proj.double().T)
        up = rounded(hidden.double() @ weights.up_proj.double().T)
        activation = rounded(gate.double() / (1 + (-gate.double()).exp()))
        product = rounded(activation.double() * up.double())
        expected = product.double() @ weights.down_proj.double().T
        record("swiglu", result, expected, hidden)
        return result

    def logits(hidden, weights):
        result = originals["final_logits"](hidden, weights)
        normalized = norm64(hidden, weights.final_norm)
        record("final_logits", result, normalized.double() @ weights.lm_head.double().T, hidden)
        return result

    with (
        patch.object(decoder, "rms_norm", norm),
        patch.object(decoder, "swiglu", mlp),
        patch.object(decoder, "final_logits", logits),
    ):
        yield


@torch.inference_mode()
def trajectory(
    weights, guard, ids, directory, *, audit=False, baseline=None, attention_backend=None
):
    directory.mkdir(parents=True, exist_ok=True)
    pool = torch.full((22, 2, 32, 16, 4, 64), torch.nan, dtype=torch.float16, device="cuda:0")
    allocator, request = KVAllocator(pool), RequestId("trajectory")
    rows, components, outputs = [], [], []
    try:
        allocator.create(KVReservation(request, 3))
        prompt = torch.tensor([ids[:15]], device=pool.device)
        _, cache = decoder.prefill(prompt, weights)
        decoder.prefill_paged(prompt, weights, allocator, request)
        pointer = pool.data_ptr()
        for step in range(32):
            token = torch.tensor([[ids[15 + step]]], device=pool.device)
            selected = audit and step in (0, 15, 31)
            with (
                component_audit(components, directory, f"reference-step-{step}")
                if selected
                else empty_context()
            ):
                reference, cache = decoder.decode(token, cache, weights, fp32_attention=True)
            with (
                guard(pool),
                component_audit(components, directory, f"paged-step-{step}")
                if selected
                else empty_context(),
            ):
                actual = decoder.decode_paged(
                    token, weights, allocator, (request,), attention_backend=attention_backend
                )
            assert pool.data_ptr() == pointer and allocator.cached_length(request) == 16 + step
            pair = {"paged": actual.cpu(), "reference": reference.cpu()}
            outputs.append(pair)
            exact = baseline is None or all(
                torch.equal(pair[name], baseline[step][name]) for name in pair
            )
            kv_ratio = 0.0
            for logical, _ in enumerate(allocator.block_table(request)):
                for page, side in zip(allocator.read_block(request, logical), (0, 1), strict=True):
                    for layer in range(22):
                        expected = cache.layers[layer][side][
                            0, :, logical * 16 : logical * 16 + page.shape[1]
                        ].transpose(0, 1)
                        kv_ratio = max(
                            kv_ratio,
                            float(
                                (
                                    (page[layer].double() - expected.double()).abs()
                                    / (0.002 * (1 + expected.double().abs()))
                                ).max()
                            ),
                        )
            rows.append(
                {
                    "step": step,
                    "attended_tokens": 16 + step,
                    "logits": compare_logits(actual.cpu(), reference.cpu()),
                    "kv_tolerance_ratio": kv_ratio,
                    "instrumentation_logits_exact": exact,
                }
            )
            torch.save(pair, directory / f"logits-{step:02}.pt")
            assert exact, "component instrumentation changed native logits; stop interpretation"
    finally:
        torch.cuda.synchronize()
        allocator.release_all(request)
        assert allocator.capacity() == CapacityCounts(32, 32, 0, 0, 0, 0)
        (directory / "report.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "components": components,
                    "attention_backend": (
                        "pagedcore"
                        if attention_backend is None
                        else type(attention_backend).__name__
                    ),
                    "cleanup": asdict(allocator.capacity()),
                    "scope": "One frozen fixture continuation; not natural-language quality or full acceptance",
                },
                indent=2,
            )
        )
    return outputs


@contextmanager
def empty_context():
    yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--attention-backend", choices=("pagedcore", "flashinfer"), default="pagedcore"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.device_count() == 1 and "T4" in torch.cuda.get_device_name(0)
    assert torch.__version__ == "2.7.1+cu126"
    fixtures = json.loads((ROOT / "tests/fixtures/reference_prompts.json").read_text())
    ids = next(c["input_ids"] for c in fixtures["prompts"] if c["name"] == "maximum_output")
    assert len(ids) >= 47
    seed_everything(0)
    guard = runpy.run_path(str(ROOT / "tests/conftest.py"))["_BlockwiseKVGuard"]
    with reference_math():
        weights = load_weights(device="cuda:0")
        attention_backend = None
        if args.attention_backend == "flashinfer":
            from pagedcore.flashinfer_backend import FlashInferAttention

            attention_backend = FlashInferAttention(device="cuda:0")
        baseline = trajectory(
            weights, guard, ids, args.output_dir / "baseline", attention_backend=attention_backend
        )
        trajectory(
            weights,
            guard,
            ids,
            args.output_dir / "audited",
            audit=True,
            baseline=baseline,
            attention_backend=attention_backend,
        )


if __name__ == "__main__":
    main()
