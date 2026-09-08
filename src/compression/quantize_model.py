"""Apply quantized wrappers to a MobileNetV2 (or any Conv2d/Linear network) and calibrate."""

from typing import List

import torch
import torch.nn as nn

from src.compression.modules import QuantConv2d, QuantLinear, set_mode


def _get_parent_and_attr(net: nn.Module, dotted_name: str):
    """Return (parent_module, attr_name) for `dotted_name` so the child can be replaced in-place."""
    parts = dotted_name.split(".")
    parent = net
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def quantize_model(
    net: nn.Module,
    weight_bits: int,
    activation_bits: int,
    per_channel_weights: bool,
    calib_method: str,
    keep_first: bool,
    keep_last: bool,
) -> List[nn.Module]:
    """Replace all eligible Conv2d/Linear submodules with fake-quantized wrappers, in-place.

    :param keep_first: leave the very first Conv2d (in forward/definition order) as float.
    :param keep_last: leave the very last Linear (in forward/definition order) as float.
    :return: the list of newly created QuantConv2d/QuantLinear modules, in traversal order.
    """
    named = list(net.named_modules())
    conv_names = [n for n, m in named if isinstance(m, nn.Conv2d)]
    linear_names = [n for n, m in named if isinstance(m, nn.Linear)]

    first_conv = conv_names[0] if conv_names else None
    last_linear = linear_names[-1] if linear_names else None

    quant_modules: List[nn.Module] = []
    for name, module in named:
        if isinstance(module, nn.Conv2d):
            if keep_first and name == first_conv:
                continue
            parent, attr = _get_parent_and_attr(net, name)
            q = QuantConv2d(
                module, weight_bits, activation_bits, per_channel_weights, calib_method
            )
            setattr(parent, attr, q)
            quant_modules.append(q)
        elif isinstance(module, nn.Linear):
            if keep_last and name == last_linear:
                continue
            parent, attr = _get_parent_and_attr(net, name)
            q = QuantLinear(
                module, weight_bits, activation_bits, per_channel_weights, calib_method
            )
            setattr(parent, attr, q)
            quant_modules.append(q)

    return quant_modules


def calibrate(net: nn.Module, loader, num_batches: int, device) -> None:
    """Run `num_batches` calibration batches through `net` in float, then freeze qparams."""
    set_mode(net, "calibrate")
    net.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches:
                break
            net(x.to(device))

    for m in net.modules():
        if isinstance(m, (QuantConv2d, QuantLinear)):
            m.freeze()
    set_mode(net, "quantized")
