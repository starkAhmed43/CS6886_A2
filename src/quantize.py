# Load full (non-weights-only) checkpoints; PyTorch 2.6+ defaults torch.load to weights_only=True.
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
# - setting up PROJECT_ROOT environment variable
# - loading environment variables from ".env" in root dir
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.compression.quantize_model import calibrate, quantize_model
from src.compression.size import activation_compression, model_size_report
from src.models.cifar_module import CIFARLitModule
from src.utils import RankedLogger, extras, instantiate_loggers

log = RankedLogger(__name__, rank_zero_only=True)


@torch.no_grad()
def _evaluate(net, loader, device) -> float:
    """Compute top-1 accuracy of `net` on `loader`."""
    net.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = net(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.numel()
    return correct / total


def quantize(cfg: DictConfig) -> None:
    """Post-training quantize a MobileNetV2 checkpoint and report accuracy/size."""
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    dm = hydra.utils.instantiate(cfg.data)
    dm.prepare_data()
    dm.setup()

    log.info(f"Loading checkpoint <{cfg.ckpt_path}>")
    model = CIFARLitModule.load_from_checkpoint(cfg.ckpt_path)
    net = model.net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device).eval()

    comp = cfg.compression
    log.info(
        f"Quantizing model: weight_bits={comp.weight_bits} act_bits={comp.activation_bits} "
        f"per_channel={comp.per_channel_weights} method={comp.calib_method}"
    )
    quant_modules = quantize_model(
        net,
        weight_bits=comp.weight_bits,
        activation_bits=comp.activation_bits,
        per_channel_weights=comp.per_channel_weights,
        calib_method=comp.calib_method,
        keep_first=comp.keep_first,
        keep_last=comp.keep_last,
    )

    log.info(f"Calibrating on {comp.num_calib_batches} batches...")
    calibrate(net, dm.train_dataloader(), comp.num_calib_batches, device)

    log.info("Evaluating quantized model on the test set...")
    test_loader = dm.test_dataloader()
    quantized_acc = _evaluate(net, test_loader, device)

    size_report = model_size_report(
        net, quant_modules, comp.weight_bits, comp.activation_bits, comp.per_channel_weights
    )
    example_input, _ = next(iter(test_loader))
    act_report = activation_compression(
        net, quant_modules, example_input.to(device), comp.activation_bits
    )

    log.info("=" * 60)
    log.info(f"Quantized top-1 accuracy: {quantized_acc:.4f}")
    log.info(
        f"Model size: {size_report['model_size_mb']:.3f} MB (baseline {size_report['baseline_mb']:.3f} MB)"
    )
    log.info(f"Weight compression ratio: {size_report['compression_ratio']:.3f}x")
    log.info(f"Weight-only compression ratio: {size_report['weight_compression_ratio']:.3f}x")
    log.info(f"Activation compression ratio: {act_report['activation_compression_ratio']:.3f}x")
    log.info("=" * 60)

    loggers = instantiate_loggers(cfg.get("logger"))
    if loggers:
        metrics = {
            "weight_quant_bits": comp.weight_bits,
            "activation_quant_bits": comp.activation_bits,
            "quantized_acc": quantized_acc,
            "model_size_mb": size_report["model_size_mb"],
            "compression_ratio": size_report["compression_ratio"],
            "weight_cr": size_report["weight_compression_ratio"],
            "activation_cr": act_report["activation_compression_ratio"],
        }
        for logger in loggers:
            logger.log_metrics(metrics)
            logger.save()
            if hasattr(logger, "finalize"):
                logger.finalize("success")


@hydra.main(version_base="1.3", config_path="../configs", config_name="quantize.yaml")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: apply run extras then quantize the configured checkpoint."""
    extras(cfg)
    quantize(cfg)


if __name__ == "__main__":
    main()
