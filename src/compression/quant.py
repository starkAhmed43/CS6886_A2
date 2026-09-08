"""Uniform affine (fake) quantization math.

All functions operate on plain float tensors: quantization is simulated by rounding
values to an integer grid and immediately dequantizing them, so the rest of the
network keeps computing in float (fake quantization).
"""

from typing import Tuple

import torch


def get_qmin_qmax(num_bits: int, symmetric: bool) -> Tuple[int, int]:
    """Return the integer grid bounds for a given bit width.

    Symmetric grids are centered on zero (e.g. int8 -> [-127, 127]).
    Asymmetric grids use the full unsigned range [0, 2**num_bits - 1].
    """
    if symmetric:
        qmax = (2 ** (num_bits - 1)) - 1
        return -qmax, qmax
    return 0, (2**num_bits) - 1


def fake_quantize(
    x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, qmin: int, qmax: int
) -> torch.Tensor:
    """Quantize `x` to the integer grid [qmin, qmax] then immediately dequantize.

    `scale`/`zero_point` may be scalars or per-channel tensors broadcastable against `x`.
    """
    q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
    return (q - zero_point) * scale


def _reduce_dims(x: torch.Tensor, ch_axis: int):
    """Return all dims of `x` except `ch_axis`, for per-channel min/max reduction."""
    return [d for d in range(x.dim()) if d != ch_axis]


def _minmax_qparams(
    x_min: torch.Tensor, x_max: torch.Tensor, num_bits: int, symmetric: bool
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Derive (scale, zero_point, qmin, qmax) from an observed [x_min, x_max] range."""
    qmin, qmax = get_qmin_qmax(num_bits, symmetric)
    if symmetric:
        max_abs = torch.maximum(x_max.abs(), x_min.abs()).clamp(min=1e-8)
        scale = max_abs / qmax
        zero_point = torch.zeros_like(scale)
    else:
        scale = ((x_max - x_min) / (qmax - qmin)).clamp(min=1e-8)
        zero_point = torch.round(qmin - x_min / scale).clamp(qmin, qmax)
    return scale, zero_point, qmin, qmax


def compute_qparams(
    x: torch.Tensor,
    num_bits: int,
    symmetric: bool,
    per_channel: bool,
    ch_axis: int = 0,
    method: str = "minmax",
    num_mse_steps: int = 80,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute (scale, zero_point) for quantizing `x`.

    :param per_channel: reduce min/max over all dims except `ch_axis`; scale/zero_point are
        broadcastable against `x` (e.g. [O,1,1,1] for a conv weight with ch_axis=0).
    :param method: "minmax" (observed range) or "mse" (search clipping fractions of the
        observed range and keep the one minimizing quantization MSE).
    """
    if per_channel:
        dims = _reduce_dims(x, ch_axis)
        x_min = x.amin(dim=dims, keepdim=True)
        x_max = x.amax(dim=dims, keepdim=True)
    else:
        x_min = x.amin()
        x_max = x.amax()

    if method == "minmax":
        scale, zero_point, qmin, qmax = _minmax_qparams(x_min, x_max, num_bits, symmetric)
        return scale, zero_point

    if method != "mse":
        raise ValueError(f"Unknown calibration method: {method}")

    qmin, qmax = get_qmin_qmax(num_bits, symmetric)
    fractions = torch.linspace(1.0 / num_mse_steps, 1.0, num_mse_steps, device=x.device)

    if per_channel:
        dims = _reduce_dims(x, ch_axis)
        num_ch = x.shape[ch_axis]
        best_mse = torch.full((num_ch,) + (1,) * (x.dim() - 1), float("inf"), device=x.device)
        # move ch_axis to dim 0 for easy per-channel indexing
        best_scale = torch.zeros_like(x_min)
        best_zp = torch.zeros_like(x_min)
        for frac in fractions:
            clip_min = x_min * frac
            clip_max = x_max * frac
            scale, zero_point, _, _ = _minmax_qparams(clip_min, clip_max, num_bits, symmetric)
            xq = fake_quantize(x, scale, zero_point, qmin, qmax)
            mse = ((xq - x) ** 2).mean(dim=dims, keepdim=True)
            improve = mse < best_mse
            best_mse = torch.where(improve, mse, best_mse)
            best_scale = torch.where(improve, scale, best_scale)
            best_zp = torch.where(improve, zero_point, best_zp)
        return best_scale, best_zp
    else:
        best_mse = torch.tensor(float("inf"))
        best_scale, best_zp = None, None
        for frac in fractions:
            clip_min = x_min * frac
            clip_max = x_max * frac
            scale, zero_point, _, _ = _minmax_qparams(clip_min, clip_max, num_bits, symmetric)
            xq = fake_quantize(x, scale, zero_point, qmin, qmax)
            mse = ((xq - x) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_scale, best_zp = scale, zero_point
        return best_scale, best_zp
