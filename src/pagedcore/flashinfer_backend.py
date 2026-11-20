"""Experimental FlashInfer decode over PagedCore-owned pages (no dense gather).

Pinned candidate: flashinfer-python 0.2.5, PyTorch 2.7.1+cu126, Linux/T4.
JIT installation and numerical acceptance still require the T4 validation run.
"""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from threading import current_thread
from typing import Protocol, cast

import torch
from torch import Tensor

from pagedcore.engine_types import RequestId
from pagedcore.kv_cache import KVAllocator

FLASHINFER_VERSION = "0.2.5"


class _DecodeWrapper(Protocol):
    def plan(
        self,
        indptr: Tensor,
        indices: Tensor,
        last_page_len: Tensor,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        pos_encoding_mode: str,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        sm_scale: float,
    ) -> None: ...

    def run(self, query: Tensor, paged_kv_cache: tuple[Tensor, Tensor]) -> Tensor: ...


class _FlashInfer(Protocol):
    def BatchDecodeWithPagedKVCacheWrapper(
        self,
        workspace: Tensor,
        *,
        kv_layout: str,
        use_cuda_graph: bool,
        use_tensor_cores: bool,
    ) -> _DecodeWrapper: ...


def _load_wrapper(device: torch.device) -> tuple[Tensor, _DecodeWrapper]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FlashInfer decode requires CUDA on an NVIDIA T4")
    if torch.cuda.get_device_capability(device) != (7, 5):
        raise RuntimeError("this experimental backend is restricted to T4 / SM75")
    if torch.__version__ != "2.7.1+cu126":
        raise RuntimeError("FlashInfer candidate requires the frozen torch 2.7.1+cu126 environment")
    try:
        installed = version("flashinfer-python")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "install flashinfer-python==0.2.5; no fallback backend is selected"
        ) from error
    if installed != FLASHINFER_VERSION:
        raise RuntimeError(f"expected flashinfer-python=={FLASHINFER_VERSION}, got {installed}")
    module = cast(_FlashInfer, import_module("flashinfer"))
    with torch.cuda.device(device):
        workspace = torch.empty(128 * 2**20, dtype=torch.uint8, device=device)
        wrapper = module.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_cuda_graph=False, use_tensor_cores=False
        )
    return workspace, wrapper


class FlashInferAttention:
    """Worker/stream-bound experimental native decode; one request per call.

    Prefill, projections, RoPE, residuals and cache ownership stay in PagedCore.
    Rebuild small page metadata each call, including uncommitted page boundaries.
    Kernel planning is intentionally uncached until correctness is established.
    """

    def __init__(self, *, device: str | torch.device = "cuda:0") -> None:
        self._workspace, self._wrapper = _load_wrapper(torch.device(device))
        self._device = self._workspace.device
        self._worker = current_thread()
        self._stream = torch.cuda.current_stream(self._device) if self._workspace.is_cuda else None

    @torch.inference_mode()
    def __call__(
        self,
        query: Tensor,
        allocator: KVAllocator,
        request_id: RequestId,
        *,
        layer_index: int,
        eager_rounding: bool = False,
    ) -> Tensor:
        if current_thread() is not self._worker or (
            self._stream is not None and torch.cuda.current_stream(self._device) != self._stream
        ):
            raise RuntimeError("FlashInfer backend must stay on its owning worker and CUDA stream")
        if eager_rounding:
            raise ValueError("FlashInfer does not implement legacy eager rounding")
        if query.shape != (1, 32, 1, 64) or query.dtype != torch.float16:
            raise ValueError("query must be FP16 [1, 32, 1, 64]")
        view = allocator.paged_layer_view(request_id, layer_index=layer_index)
        if query.device != self._device or view.key.device != self._device:
            raise ValueError("query, KV pool and FlashInfer workspace must share one device")
        indptr = torch.tensor([0, len(view.blocks)], dtype=torch.int32, device=self._device)
        indices = torch.tensor(view.blocks, dtype=torch.int32, device=self._device)
        last_page_len = torch.tensor(
            [view.last_page_tokens], dtype=torch.int32, device=self._device
        )
        # Q and K are already rotated at absolute positions. Never rotate twice.
        with torch.autocast(device_type=self._device.type, enabled=False):
            self._wrapper.plan(
                indptr,
                indices,
                last_page_len,
                num_qo_heads=32,
                num_kv_heads=4,
                head_dim=64,
                page_size=16,
                pos_encoding_mode="NONE",
                q_data_type=torch.float16,
                kv_data_type=torch.float16,
                sm_scale=0.125,
            )
            output = self._wrapper.run(query.reshape(1, 32, 64), (view.key, view.value))
        if (
            output.shape != (1, 32, 64)
            or output.dtype != torch.float16
            or output.device != query.device
        ):
            raise RuntimeError("FlashInfer returned an invalid decode output")
        return output.reshape(1, 32, 1, 64)
