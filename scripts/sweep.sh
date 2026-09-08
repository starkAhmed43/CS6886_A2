#!/usr/bin/env bash
# Sweep PTQ bit-widths. Runs each (weight_bits, activation_bits) config as its own
# process, so every config gets its own wandb run (a single process would reuse one run).
#
# Usage:
#   scripts/sweep.sh                                   # default grid
#   WEIGHT_BITS="4 8" ACT_BITS="8" scripts/sweep.sh    # custom grid via env vars
#   scripts/sweep.sh compression.calib_method=minmax   # extra Hydra overrides are forwarded
#   WEIGHT_BITS="8" ACT_BITS="8" scripts/sweep.sh logger=csv   # override the logger (wandb grouping skipped)
set -euo pipefail

WEIGHT_BITS=${WEIGHT_BITS:-"2 3 4 6 8"}
ACT_BITS=${ACT_BITS:-"4 6 8"}
export PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}

# If the caller overrides the logger (e.g. logger=csv), skip the wandb-only overrides.
user_sets_logger=0
for a in "$@"; do
  case "$a" in logger=*) user_sets_logger=1 ;; esac
done

for wb in $WEIGHT_BITS; do
  for ab in $ACT_BITS; do
    echo "=== w${wb}a${ab} ==="
    if [ "$user_sets_logger" -eq 0 ]; then
      wandb_args=(logger=wandb "logger.wandb.group=ptq-sweep" "+logger.wandb.name=w${wb}a${ab}")
    else
      wandb_args=()
    fi
    python src/quantize.py \
      compression.weight_bits="$wb" compression.activation_bits="$ab" \
      "${wandb_args[@]}" "$@"
  done
done
