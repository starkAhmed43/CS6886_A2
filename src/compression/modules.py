"""Quantized Conv2d/Linear wrappers built around the fake-quantization math in `quant.py`.

Each wrapper has two modes:
- "calibrate": runs the float op and feeds the input activation to an `ActObserver` so its
  range can be learned.
- "quantized": fake-quantizes both the (frozen) weight and the input activation, then runs
  the op on the dequantized values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.compression.quant import compute_qparams, fake_quantize, get_qmin_qmax


class ActObserver(nn.Module):
    """Tracks a running min/max over calibration batches and freezes to fixed qparams."""

    def __init__(self):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))
        self.scale = None
        self.zero_point = None
        self.qmin = None
        self.qmax = None

    def observe(self, x: torch.Tensor) -> None:
        self.min_val = torch.minimum(self.min_val, x.detach().amin())
        self.max_val = torch.maximum(self.max_val, x.detach().amax())

    def freeze(self, num_bits: int, method: str = "minmax") -> None:
        """Compute and cache asymmetric per-tensor activation qparams from the observed range."""
        x_min = self.min_val.reshape(1)
        x_max = self.max_val.reshape(1)
        # Build a tiny 2-value tensor spanning [min, max] so compute_qparams' minmax/mse
        # logic (which reduces over the tensor) reproduces the observed range exactly.
        span = torch.stack([x_min, x_max]).reshape(-1)
        scale, zero_point = compute_qparams(
            span, num_bits, symmetric=False, per_channel=False, method=method
        )
        self.scale = scale
        self.zero_point = zero_point
        self.qmin, self.qmax = get_qmin_qmax(num_bits, symmetric=False)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        return fake_quantize(x, self.scale, self.zero_point, self.qmin, self.qmax)


class _QuantModuleMixin:
    def _cache_weight_qparams(self):
        self.weight_scale, self.weight_zero_point = compute_qparams(
            self.weight,
            self.weight_bits,
            symmetric=True,
            per_channel=self.per_channel,
            ch_axis=0,
            method=self.calib_method,
        )
        self.weight_qmin, self.weight_qmax = get_qmin_qmax(self.weight_bits, symmetric=True)

    def freeze(self) -> None:
        self.act_obs.freeze(self.act_bits, method=self.calib_method)
        self._cache_weight_qparams()

    def _quantized_weight(self) -> torch.Tensor:
        return fake_quantize(
            self.weight, self.weight_scale, self.weight_zero_point, self.weight_qmin, self.weight_qmax
        )


class QuantConv2d(nn.Module, _QuantModuleMixin):
    """Drop-in replacement for `nn.Conv2d` with fake-quantized weights and activations."""

    def __init__(
        self,
        conv: nn.Conv2d,
        weight_bits: int,
        act_bits: int,
        per_channel: bool,
        calib_method: str = "minmax",
    ):
        super().__init__()
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone()) if conv.bias is not None else None
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.per_channel = per_channel
        self.calib_method = calib_method
        self.act_obs = ActObserver()
        self.mode = "calibrate"

        self.weight_scale = None
        self.weight_zero_point = None
        self.weight_qmin = None
        self.weight_qmax = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "calibrate":
            self.act_obs.observe(x)
            return F.conv2d(
                x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        xq = self.act_obs.quantize(x)
        wq = self._quantized_weight()
        return F.conv2d(
            xq, wq, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class QuantLinear(nn.Module, _QuantModuleMixin):
    """Drop-in replacement for `nn.Linear` with fake-quantized weights and activations."""

    def __init__(
        self,
        linear: nn.Linear,
        weight_bits: int,
        act_bits: int,
        per_channel: bool,
        calib_method: str = "minmax",
    ):
        super().__init__()
        self.weight = nn.Parameter(linear.weight.detach().clone())
        self.bias = nn.Parameter(linear.bias.detach().clone()) if linear.bias is not None else None

        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.per_channel = per_channel
        self.calib_method = calib_method
        self.act_obs = ActObserver()
        self.mode = "calibrate"

        self.weight_scale = None
        self.weight_zero_point = None
        self.weight_qmin = None
        self.weight_qmax = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "calibrate":
            self.act_obs.observe(x)
            return F.linear(x, self.weight, self.bias)
        xq = self.act_obs.quantize(x)
        wq = self._quantized_weight()
        return F.linear(xq, wq, self.bias)


def set_mode(net: nn.Module, mode: str) -> None:
    """Set `.mode` on every QuantConv2d/QuantLinear module in `net`."""
    for m in net.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            m.mode = mode
