# CS6886 Assignment 2 — MobileNet-v2 on CIFAR-10

Train MobileNet-v2 on CIFAR-10 from scratch, then compress it with a custom quantization
method. This repository holds the code, configs, and reproduction commands.

Built with **PyTorch**, **Lightning**, and **Hydra**. Experiments log to **Weights & Biases**.

## Status

- **Q1 — Training baseline: implemented.** Commands and configuration are below.
- Q2–Q4 (custom compression) and their analysis build on this baseline.

## Environment

A CUDA GPU is required. The code was developed and verified on an NVIDIA RTX PRO 4500
(Blackwell, sm_120), which needs a PyTorch build with CUDA 12.8 or newer.

### Dependency versions

| Package | Version |
|---|---|
| Python | 3.12 |
| torch | 2.14.0 (cu132) |
| torchvision | 0.29.0 (cu132) |
| lightning | 2.6.5 |
| torchmetrics | 1.9.0 |
| hydra-core | 1.3.6 |
| wandb | 0.29.0 |

### Setup

```bash
# 1. Create the environment
conda create -n cs6886 python=3.12 -y
conda activate cs6886

# 2. Install PyTorch with CUDA 13.2 wheels (Blackwell-capable)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 3. Install the remaining dependencies
pip install lightning torchmetrics "hydra-core>=1.3" hydra-colorlog wandb rootutils rich pytest

# 4. Verify the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`environment.yaml` records the same stack for reference.

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
tests/                   # config and datamodule tests
```

The code separates the three concerns Q5 asks for: **data** (`src/data`),
**model and training/evaluation logic** (`src/models`, `src/train.py`, `src/eval.py`),
and future **compression** work will live in its own module.

## Hydra configuration system

Configuration is composed from small files ("config groups") instead of one large file.
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

You override any group or value from the command line:

```bash
# swap a whole group
python src/train.py trainer=cpu logger=csv

# override a single value (dotted path)
python src/train.py data.batch_size=256 optimizer.lr=0.05 trainer.max_epochs=100
```

An **experiment** file bundles a full set of overrides under one name. The Q1 baseline
lives in `configs/experiment/mobilenetv2_cifar10.yaml` and is selected with
`experiment=mobilenetv2_cifar10`.

## Reproduce Q1 (baseline training)

```bash
conda activate cs6886
export PROJECT_ROOT=$(pwd)
python src/train.py experiment=mobilenetv2_cifar10
```

This trains for 200 epochs on the GPU, evaluates on the CIFAR-10 test set, and logs
loss/accuracy curves to Weights & Biases. The best checkpoint (by `val/acc`) is saved
under `logs/train/runs/<timestamp>/checkpoints/`.

### Seed and reproducibility

The experiment sets `seed: 42`, which seeds Python, NumPy, and PyTorch through Lightning.
Override it with `seed=<n>`. For stricter determinism (slower), add
`trainer.deterministic=true`.

### Evaluate a checkpoint

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

## Q1 configuration reference

**Data (Q1a).** CIFAR-10, normalized with mean `(0.4914, 0.4822, 0.4465)` and std
`(0.2470, 0.2435, 0.2616)`. Training augmentation: `RandomCrop(32, padding=4)` and
`RandomHorizontalFlip`. The test set uses normalization only.

**Model (Q1b).** MobileNet-v2 (torchvision architecture, trained from scratch,
`width_mult=1.0`, `dropout=0.2`). The network relaxes the earliest 3 of MobileNet-v2's
5 stride-2 downsampling convs to stride 1 (controlled by `stride_relax`, default 3). This
keeps an 8x8 final feature map on 32x32 CIFAR input, instead of the stock 1x1 map that
over-downsampled small images. The change adds no parameters (~2.2M total) and only
changes where downsampling happens.

**Training strategy (Q1b).** SGD (`lr=0.1`, `momentum=0.9`, Nesterov, `weight_decay=5e-4`),
a 5-epoch linear warmup followed by cosine annealing over 200 epochs, batch size 128,
label smoothing 0.1, and mixed precision (`bf16-mixed`).

## Results (Q1c)

**Final test top-1 accuracy: 93.98%** (`stride_relax=3`, the default).

The CIFAR downsampling adaptation matters. A larger final feature map keeps more spatial
detail and gives a clear accuracy gain:

| `stride_relax` | final map | test top-1 |
|---|---|---|
| 1 | 2x2 | 90.65% |
| 2 | 4x4 | 92.47% |
| 3 | 8x8 | **93.98%** |

Training/validation loss and accuracy curves: see `reports/q1_curves.png` (and the wandb
run `mnv2-relax3-8x8-94.0`). Per-class accuracy and confusion matrix:
`reports/q1_confusion_matrix.png`.

## Tests

```bash
conda activate cs6886
export PROJECT_ROOT=$(pwd)
python -m pytest tests/test_configs.py tests/test_datamodules.py -q
```

## License

MIT.
