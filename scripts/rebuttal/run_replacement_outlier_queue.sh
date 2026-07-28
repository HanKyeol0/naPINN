#!/usr/bin/env bash
# Replacement-outlier paired comparison.
#
# The manuscript says gross outliers replace an observation; the submitted code
# adds a large positive offset to it. This queue runs the replacement protocol
# on the same PDE, severity, methods and seeds as the completed additive core,
# with the same corrupted rows and the same magnitude draw, so the two are
# paired and the only difference is the protocol itself.
#
# The additive counterpart already exists in
# outputs/rebuttal/synthetic_recovery_20260726 and is not re-run.
set -u
GPU="${1:?usage: run_replacement_outlier_queue.sh <gpu>}"
ROOT=outputs/rebuttal/replacement_outlier_20260728

for SEED in 40 41 42; do
  for METHOD in mse lad orpinn_q19 orpinn_q29 pinn_ebm napinn; do
    echo "=== ${METHOD} seed=${SEED} ==="
    python -m scripts.rebuttal.run_synthetic \
      --experiment-name allencahn2d \
      --method "${METHOD}" \
      --seed "${SEED}" \
      --device "cuda:${GPU}" \
      --noise-kind 4G \
      --outlier-ratio 0.15 \
      --outlier-mode replacement \
      --experiment-config configs/rebuttal/allencahn2d.yaml \
      --output-root "${ROOT}"
  done
done
