"""Storage/size accounting for a (partially) quantized model."""

from typing import Dict, List

import torch
import torch.nn as nn


def model_size_report(
    net: nn.Module,
    quant_modules: List[nn.Module],
    weight_bits: int,
    activation_bits: int,
    per_channel_weights: bool,
) -> Dict[str, float]:
    """Report the storage cost of `net` if quantized weights are packed at `weight_bits`.

    Every other parameter (BN affine params, biases, kept float layers) is counted at fp32.
    Adds per-channel weight scale overhead (symmetric weights need no zero_point) and
    per-tensor activation scale+zero_point overhead for each quantized module.
    """
    quant_weight_ptrs = {id(m.weight) for m in quant_modules}

    total_params = 0
    quantized_weight_params = 0
    other_param_bits = 0.0
    for p in net.parameters():
        n = p.numel()
        total_params += n
        if id(p) in quant_weight_ptrs:
            quantized_weight_params += n
        else:
            other_param_bits += n * 32

    quantized_weight_bits = quantized_weight_params * weight_bits

    weight_meta_bits = 0.0
    for m in quant_modules:
        out_channels = m.weight.shape[0]
        num_scales = out_channels if per_channel_weights else 1
        weight_meta_bits += num_scales * 32  # symmetric weights -> no zero_point to store

    act_meta_bits = len(quant_modules) * 64.0  # one fp32 scale + one int32 zero_point each

    baseline_bits = total_params * 32
    model_bits = quantized_weight_bits + other_param_bits + weight_meta_bits + act_meta_bits

    model_size_mb = model_bits / 8 / 1e6
    baseline_mb = baseline_bits / 8 / 1e6
    compression_ratio = baseline_bits / model_bits

    weight_compression_ratio = (quantized_weight_params * 32) / (
        quantized_weight_params * weight_bits + weight_meta_bits
    )

    return {
        "baseline_bits": baseline_bits,
        "model_bits": model_bits,
        "baseline_mb": baseline_mb,
        "model_size_mb": model_size_mb,
        "compression_ratio": compression_ratio,
        "weight_compression_ratio": weight_compression_ratio,
        "weight_meta_bytes": weight_meta_bits / 8,
        "act_meta_bytes": act_meta_bits / 8,
        "total_params": total_params,
        "quantized_weight_params": quantized_weight_params,
    }


def activation_compression(
    net: nn.Module,
    quant_modules: List[nn.Module],
    example_input: torch.Tensor,
    activation_bits: int,
) -> Dict[str, float]:
    """Estimate the activation storage savings from quantizing each quant module's input."""
    numels = []
    hooks = []

    def make_hook():
        def hook(module, inputs):
            numels.append(inputs[0].numel())

        return hook

    for m in quant_modules:
        hooks.append(m.register_forward_pre_hook(make_hook()))

    net.eval()
    with torch.no_grad():
        net(example_input)

    for h in hooks:
        h.remove()

    total_numel = sum(numels)
    activation_bits_total = total_numel * activation_bits
    baseline_bits_total = total_numel * 32
    activation_compression_ratio = (
        baseline_bits_total / activation_bits_total if activation_bits_total else 1.0
    )

    return {
        "activation_numel_total": total_numel,
        "activation_bits_total": activation_bits_total,
        "activation_baseline_bits_total": baseline_bits_total,
        "activation_compression_ratio": activation_compression_ratio,
    }
