"""MobileNet-v2 factory for CIFAR-10.

Torchvision's `mobilenet_v2` is designed for 224x224 ImageNet inputs and has 5 stride-2
downsampling convs, which together downsample a 32x32 CIFAR-10 input all the way to a 1x1
feature map before the network even sees much of it. The number of these downsampling
strides to relax to stride 1 is configurable via `stride_relax` (default 2, matching the
stem conv and the first inverted-residual downsample), which keeps more spatial detail
early on for small CIFAR-sized inputs.

stride_relax=0 => 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1 final feature map
stride_relax=1 => 32x32 -> 32x32 -> 16x16 -> 8x8 -> 4x4 -> 2x2 final feature map
stride_relax=2 => 32x32 -> 32x32 -> 32x32 -> 16x16 -> 8x8 -> 4x4 final feature map
stride_relax=3 => 32x32 -> 32x32 -> 32x32 -> 32x32 -> 16x16 -> 8x8 final feature map
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


def mobilenetv2_cifar(
    num_classes: int = 10,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    stride_relax: int = 2,
) -> torch.nn.Module:
    """Build a MobileNet-v2 adapted for CIFAR-sized (32x32) inputs.

    :param num_classes: Number of output classes.
    :param width_mult: Width multiplier for the network channels.
    :param dropout: Dropout probability before the final classifier.
    :param stride_relax: Number of the 5 stride-2 downsampling convs, from earliest, to set
        to stride 1; 2 keeps a 4x4 final map on 32x32 input, 3 keeps 8x8, 0 is the stock
        ImageNet network.
    :return: A `torch.nn.Module` instance of the adapted MobileNet-v2.
    """
    net = mobilenet_v2(
        weights=None, num_classes=num_classes, width_mult=width_mult, dropout=dropout
    )
    # Collect the stride-2 downsampling convs in forward order, then relax the earliest
    # `stride_relax` of them to stride 1. See module docstring for the CIFAR rationale.
    downsamplers = [
        m
        for blk in net.features
        for m in blk.modules()
        if isinstance(m, nn.Conv2d) and tuple(m.stride) == (2, 2)
    ]
    if not 0 <= stride_relax <= len(downsamplers):
        raise ValueError(f"stride_relax must be in [0, {len(downsamplers)}], got {stride_relax}")
    for conv in downsamplers[:stride_relax]:
        conv.stride = (1, 1)
    return net


if __name__ == "__main__":
    model = mobilenetv2_cifar()
    x = torch.randn(2, 3, 32, 32)
    print(model(x).shape)
