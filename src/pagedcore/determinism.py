"""Small controls for reproducible inference and tests."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager


def seed_everything(seed: int) -> None:
    """Seed Python and the available PyTorch CPU/CUDA generators.

    PyTorch is an optional package for the base installation, so it is imported
    only when this control is used. A missing PyTorch installation still gets a
    reproducible Python random stream.
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    random.seed(seed)

    try:
        import torch
    except (ImportError, OSError):
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def inference_mode() -> Iterator[None]:
    """Run an inference section without autograd bookkeeping."""
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError("PyTorch is required to enable inference mode") from error

    with torch.inference_mode():
        yield
