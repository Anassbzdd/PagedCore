"""Replay saved attention operands only; no model load or serving changes."""

import hashlib
import json
import sys
from pathlib import Path

import torch

from pagedcore.oracle import reference_math


def restore(saved, device):
    value, m = saved["value"], saved["layout"]
    backing = torch.empty(
        m["storage_nbytes"] // value.element_size(), dtype=value.dtype, device=device
    )
    view = backing.as_strided(m["shape"], m["stride"], m["storage_offset"])
    view.copy_(value)
    return view


@torch.inference_mode()
def replay(path, device):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    rm, pm = raw["reference_attention_math"], raw["paged_attention_math"]

    def product(events):
        return [e for e in events if e["operation"].split(".")[1] in ("matmul", "bmm", "mm")][-1]

    def scores(events):
        return next(e["output"] for e in events if e.get("scalar") == 0.125)

    rs, ps = restore(scores(rm), device), restore(scores(pm), device)
    rv = restore(product(rm)["operands"][1], device)
    pv = restore(product(pm)["operands"][1], device)
    assert torch.equal(rs.reshape(4, 8, 2), ps)
    assert torch.equal(rv.reshape(4, 8, 2, 64), pv[:, None].expand(4, 8, 2, 64))
    e = (ps - ps.amax(-1, keepdim=True)).exp()
    d = e.sum(-1, keepdim=True)
    n = e @ pv
    q = raw["reference"]["q_rotated"]["value"].to(device).double()
    k = raw["reference"]["k_read"]["value"].to(device).double().repeat_interleave(8, 1)
    v = raw["reference"]["v_read"]["value"].to(device).double().repeat_interleave(8, 1)
    outputs = {
        "reference_normalize_first": rs.softmax(-1) @ rv,
        "grouped_softmax_first": ps.softmax(-1) @ pv,
        "exp_normalize_first": (e / d) @ pv,
        "normalize_last": n / d,
        "division_fp64_only": n.double() / d.double(),
        "pv_and_division_fp64": (e.double() @ pv.double()) / d.double(),
        "fp64_fixed_scores": rs.double().softmax(-1) @ v,
        "fp64_qkv": ((q @ k.transpose(-2, -1)) * 0.125).softmax(-1) @ v,
    }
    outputs = {name: value.reshape(1, 32, 1, 64).cpu() for name, value in outputs.items()}
    recorded_ref = product(rm)["output"]["value"].reshape(1, 32, 1, 64)
    recorded_paged = next(
        e["output"]["value"] for e in reversed(pm) if e["operation"] == "aten.div.Tensor"
    ).reshape(1, 32, 1, 64)
    report = {
        "device": device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_replays_exact": {
            "reference": torch.equal(outputs["reference_normalize_first"], recorded_ref),
            "paged": torch.equal(outputs["normalize_last"], recorded_paged),
        },
        "arms": {},
    }
    for name, value in outputs.items():
        report["arms"][name] = {
            "head0_channel3": float(value[0, 0, 0, 3]),
            "head0_channel3_fp16": float(value.half()[0, 0, 0, 3]),
            "max_error_fp64_qkv": float((value.double() - outputs["fp64_qkv"]).abs().max()),
            "max_error_fp64_fixed_scores": float(
                (value.double() - outputs["fp64_fixed_scores"]).abs().max()
            ),
            "fp16_mismatches_fp64_qkv": int((value.half() != outputs["fp64_qkv"].half()).sum()),
        }
    return report, outputs


if __name__ == "__main__":
    path, output = Path(sys.argv[1]), Path(sys.argv[2])
    device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
    output.mkdir(parents=True, exist_ok=True)
    if device.startswith("cuda"):
        assert torch.__version__ == "2.7.1+cu126" and "T4" in torch.cuda.get_device_name(0)
    with reference_math() as settings:
        report, tensors = replay(path, device)
        report["math_settings"] = settings
    report["scope"] = "Isolated saved layer-5 operands, not integrated acceptance"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(tensors, output / "outputs.pt")
    print(json.dumps(report, indent=2))
    if device.startswith("cuda") and not all(report["native_replays_exact"].values()):
        raise SystemExit("Native replay differs; inspect evidence before interpreting controls")
