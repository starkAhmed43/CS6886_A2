import torch

from src.compression.quant import compute_qparams, fake_quantize, get_qmin_qmax
from src.compression.size import model_size_report
from src.compression.modules import QuantLinear
import torch.nn as nn


def test_fake_quantize_roundtrip_tight():
    torch.manual_seed(0)
    x = torch.randn(10000)
    scale, zero_point = compute_qparams(x, 8, symmetric=False, per_channel=False)
    qmin, qmax = get_qmin_qmax(8, symmetric=False)
    xq = fake_quantize(x, scale, zero_point, qmin, qmax)
    assert (xq - x).abs().max() <= 1.5 * scale


def test_per_channel_weight_qparams_shape():
    w = torch.randn(8, 3, 3, 3)
    scale, zero_point = compute_qparams(w, 8, symmetric=True, per_channel=True, ch_axis=0)
    assert scale.shape == (8, 1, 1, 1)
    assert zero_point.shape == (8, 1, 1, 1)
    # broadcast check
    _ = fake_quantize(w, scale, zero_point, *get_qmin_qmax(8, True))


def test_get_qmin_qmax():
    assert get_qmin_qmax(8, False) == (0, 255)
    assert get_qmin_qmax(8, True) == (-127, 127)


def test_model_size_report_linear_8bit():
    linear = nn.Linear(64, 10)
    ql = QuantLinear(linear, weight_bits=8, act_bits=8, per_channel=True, calib_method="minmax")
    ql.weight.data = linear.weight.data.clone()
    ql._cache_weight_qparams()
    net = nn.Sequential(ql)
    report = model_size_report(net, [ql], weight_bits=8, activation_bits=8, per_channel_weights=True)
    assert 3 < report["compression_ratio"] < 4.2
