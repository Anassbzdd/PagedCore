import math

import pytest
import torch

from pagedcore.dense import apply_rope, rms_norm

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("scale", [0.0, 1e-4, 1.0, 1000.0])
def test_rms_norm_against_fp64_definition(scale):
    hidden = (torch.randn(2, 3, 2048) * scale).half()
    weight = torch.linspace(0.5, 1.5, 2048).half()
    values = hidden.double()
    expected = values / (values.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    # Normalized activations are rounded before the learned FP16 scale in this decoder.
    expected = expected.half() * weight
    actual = rms_norm(hidden, weight)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    assert torch.equal(hidden, values.half())


@pytest.mark.parametrize("position", [0, 15, 16, 17, 511, 2047])
def test_rope_all_half_pairs_against_scalar_rotation(position):
    query = torch.randn(1, 32, 1, 64).half()
    key = torch.randn(1, 4, 1, 64).half()
    actual = apply_rope(query, key, torch.tensor([[position]]))
    for source, result in zip((query, key), actual, strict=True):
        expected = torch.empty_like(source)
        for pair in range(32):
            angle = position / (10000.0 ** (2 * pair / 64))
            # Runtime casts sine/cosine to the activation dtype before the rotation.
            cosine = torch.tensor(math.cos(angle), dtype=torch.float16)
            sine = torch.tensor(math.sin(angle), dtype=torch.float16)
            a, b = source[..., pair], source[..., pair + 32]
            expected[..., pair] = a * cosine - b * sine
            expected[..., pair + 32] = b * cosine + a * sine
        torch.testing.assert_close(result, expected, atol=2e-3, rtol=2e-3)
