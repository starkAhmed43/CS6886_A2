# CS6886 Assignment 2 — MobileNet-v2 on CIFAR-10

Train MobileNet-v2 on CIFAR-10 from scratch, then compress it with a custom quantization method.

Built with **PyTorch**, **Lightning**, and **Hydra**. Experiments log to **Weights & Biases**.

## Environment

A CUDA GPU is required. The code was developed and executed on an NVIDIA RTX PRO 4500 Blackwell, which needs a PyTorch build with CUDA 12.8 or newer.

### Dependency versions

| Package      | Version        |
| ------------ | -------------- |
| Python       | 3.12           |
| torch        | 2.14.0 (cu132) |
| torchvision  | 0.29.0 (cu132) |
| lightning    | 2.6.5          |
| torchmetrics | 1.9.0          |
| hydra-core   | 1.3.6          |
| wandb        | 0.29.0         |

### Setup

This project uses conda. If conda is not installed, install Miniconda first — see
https://docs.conda.io/en/latest/miniconda.html (or https://www.anaconda.com/docs/getting-started/miniconda/install).

Recreate the exact environment from the pinned spec:

```bash
# 1. Create and activate the environment (installs the Blackwell/CUDA 13.2 torch build)
conda env create -f environment.yaml
conda activate cs6886

# 2. Verify the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`environment.yaml` pins `torch==2.14.0+cu132` from the PyTorch CUDA 13.2 index. This build
targets NVIDIA GPUs from the Turing generation onwards. For a different GPU or CUDA version, change the
`--extra-index-url` and the torch/torchvision versions in `environment.yaml`.

### Weights & Biases

Log in once, then set your project and entity in `configs/logger/wandb.yaml`
(`project` and `entity`). To run without an account, append `logger=csv` to any command.

```bash
wandb login
```

## Repository structure

```
configs/                 # Hydra configuration (composed at runtime)
├── train.yaml           # main training config; lists the default config groups
├── eval.yaml            # evaluation config (needs a checkpoint path)
├── data/cifar10.yaml    # CIFAR-10 datamodule settings
├── model/mobilenetv2.yaml  # LightningModule + network + loss settings
├── optimizer/default.yaml  # SGD settings
├── scheduler/default.yaml  # warmup + cosine schedule settings
├── experiment/mobilenetv2_cifar10.yaml  # the Q1 baseline experiment
├── trainer/             # cpu / gpu / ddp trainer presets
├── callbacks/           # checkpoint, early stopping, progress bar
├── logger/              # wandb, csv, tensorboard, ...
└── ...
src/
├── train.py             # training entry point
├── eval.py              # evaluation entry point
├── data/
│   └── cifar10_datamodule.py   # CIFAR10DataModule (transforms, dataloaders)
├── models/
│   ├── cifar_module.py         # CIFARLitModule (train/val/test steps, metrics)
│   └── components/
│       ├── mobilenetv2.py      # MobileNet-v2 factory with the CIFAR stem fix
│       └── schedulers.py       # warmup_cosine learning-rate schedule
└── utils/               # logging, instantiation, and helper utilities
tests/                   # config, datamodule, and quantization tests
```

The code is split as follows: **data** (`src/data`), **model, training, and evaluation** (`src/models`, `src/train.py`, `src/eval.py`), and **compression** (`src/compression`, `src/quantize.py`).

## Hydra configuration system

Using the Hydra library, configuration is composed from small files ("config groups") instead of one large file.
`configs/train.yaml` lists the default choice for each group:

```yaml
defaults:
  - data: cifar10
  - model: mobilenetv2
  - optimizer: default
  - scheduler: default
  - trainer: default
  - logger: wandb
```

You can override any group or value by modifying the config files or directly from the command line:

```bash
# swap a whole group
python src/train.py trainer=cpu logger=csv

# override a single value (dotted path)
python src/train.py data.batch_size=256 optimizer.lr=0.05 trainer.max_epochs=100
```

In this repo, an **experiment** config file bundles a full set of overrides under one name. The Q1 config
lives in `configs/experiment/mobilenetv2_cifar10.yaml` and is selected with `experiment=mobilenetv2_cifar10`.

## Reproduce Q1 (baseline training)

```bash
conda activate cs6886
export PROJECT_ROOT=$(pwd) # PWD should be the root of this project. Otherwise provide the path to the project root
python src/train.py experiment=mobilenetv2_cifar10
```

This trains for 200 epochs on the GPU, evaluates on the CIFAR-10 test set, and logs
loss/accuracy curves to Weights & Biases. The best checkpoint (by `val/acc`) is saved
under `logs/train/runs/<timestamp>/checkpoints/`.

The first run downloads CIFAR-10 automatically (~170 MB): the archive to `data/raw/` and the
extracted set to `data/processed/`. It needs internet once, then caches locally. The dataset is
git-ignored, so it is never committed.

### Seed and reproducibility

The experiment sets `seed: 42`, which seeds Python, NumPy, and PyTorch through Lightning.
Override it with `seed=<n>`. For stricter determinism (slower), add `trainer.deterministic=true`.

### Evaluate a checkpoint

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

## Q1 configuration details

**Data (Q1a).** CIFAR-10, normalized with mean `(0.4914, 0.4822, 0.4465)` and std
`(0.2470, 0.2435, 0.2616)`. Training augmentation: `RandomCrop(32, padding=4)` and
`RandomHorizontalFlip`. The test set uses normalization only.

**Model (Q1b).** MobileNet-v2 (torchvision architecture, trained from scratch,
`width_mult=1.0`, `dropout=0.2`). The network relaxes the earliest 3 of MobileNet-v2's
5 stride-2 downsampling convs to stride 1 (controlled by `stride_relax` in `configs/model/mobilenetv2.yaml`, default 3). This
keeps an 8x8 final feature map on 32x32 CIFAR input, instead of the stock 1x1 map that
over-downsampled small images. The change adds no parameters (~2.2M total) and only changes where the downsampling happens.

**Training strategy (Q1b).** SGD (`lr=0.1`, `momentum=0.9`, `Nesterov`, `weight_decay=5e-4`),
a 5-epoch linear warmup followed by cosine annealing over 200 epochs, batch size 128,
label smoothing 0.1, and mixed precision (`bf16-mixed`).

## Results (Q1c)

**Final test top-1 accuracy: 93.98%** (`stride_relax=3`).

The downsampling procedure we choose for MobileNet-v2 matters for the results we get on the CIFAR-10 dataset.
A larger final feature map keeps more spatial detail and gives a clear accuracy gain:

| `stride_relax` | final map | test top-1 |
| -------------- | --------- | ---------- |
| 1              | 2x2       | 90.65%     |
| 2              | 4x4       | 92.47%     |
| 3              | 8x8       | **93.98%** |

Training/validation loss and accuracy curves: see `reports/q1_curves.png` (and the wandb
run `mnv2-relax3-8x8-94.0`). Per-class accuracy and confusion matrix: `reports/q1_confusion_matrix.png`.

## Compression — Post-Training Quantization (Q2–Q4)

We quantize the trained MobileNet-v2 with using post-training quantization (PTQ). No
external quantization library is used. Weights use per-channel symmetric quantization; activations
use per-tensor asymmetric quantization with static calibration (MSE clipping). The first
convolution, the final classifier and BatchNorms stay in float32.

### Run quantization

```bash
conda activate cs6886
export PROJECT_ROOT=$(pwd)

# Quantize the baseline at 8-bit weights + 8-bit activations
python src/quantize.py compression.weight_bits=8 compression.activation_bits=8

# A different level (4-bit weights, 8-bit activations)
python src/quantize.py compression.weight_bits=4 compression.activation_bits=8
```

### Bit-width sweep (Q3)

Use the sweep script:

```bash
# default grid: weights {2,3,4,6,8} x activations {4,6,8}
bash scripts/sweep.sh

# custom grid via env vars; extra Hydra overrides are forwarded
WEIGHT_BITS="4 8" ACT_BITS="8" bash scripts/sweep.sh compression.calib_method=minmax
```

The runs land in the wandb group `ptq-sweep`.

### Configurable options

`configs/compression/default.yaml` keys:

- `weight_bits` — bit width for weight quantization.
- `activation_bits` — bit width for activation quantization.
- `per_channel_weights` — quantize each output channel's weights separately.
- `calib_method` — `minmax` or `mse`, how calibration picks activation clipping ranges.
- `num_calib_batches` — number of training batches used for calibration.
- `keep_first` — keep the first layer in float.
- `keep_last` — keep the last (classifier) layer in float.

### Results

Baseline (Q1): 93.98% top-1, 8.95 MB.

| weights | activations | top-1  | size (MB) | model ratio |
| ------- | ----------- | ------ | --------- | ----------- |
| 8-bit   | 8-bit       | 93.86% | 2.45      | 3.65x       |
| 6-bit   | 8-bit       | 93.75% | 1.90      | 4.71x       |
| 4-bit   | 8-bit       | 90.80% | 1.35      | 6.61x       |

**Optimal quant chosen for Q4**: 4-bit weights, 8-bit activations — 90.80% top-1, 1.35 MB,
6.61x model compression (weight-only ~7.8x, activation 4.0x).
Full sweep: `reports/q2_sweep.csv` and `reports/q2_frontier.png`.

### Seed and reproducibility

Quantization calibration uses a few training batches; results vary by \<0.1% run to run.

## Tests

```bash
conda activate cs6886
export PROJECT_ROOT=$(pwd)
python -m pytest tests/test_configs.py tests/test_datamodules.py tests/test_quant.py -q
```

## AI usage declaration

I directed this project and used Claude Code as a coding assistant.

My work:

- Set the goals and drove every design decision.
- Ran the `stride_relax` experiments for Q1 and chose the final configuration (relax=3).
- Selected the quantization approach: PTQ, per-channel weights, the calibration method, and the final bit-width operating point.
- Reviewed all code and results, and verified the numbers.

The assistant helped with:

- Turning my design into the implementation (`src/compression/`, `src/quantize.py`)
  and making it Hydra-configurable.
- Boilerplate: the Lightning + Hydra wiring, comments/docstrings, and tests.
- Debugging (CI, Weights & Biases, checkpoint loading), and generating figures and tables.

All quantization logic is custom code. No external quantization library is used.
