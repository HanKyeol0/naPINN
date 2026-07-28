#!/usr/bin/env bash
# The exact ablation Reviewer aoJS asked for in Q2.
#
# aoJS asked for "warm-up plus residual-density estimation but without a
# trainable gate", to test whether the gains come from residual screening
# rather than from the trainable gate and rejection cost. Our existing
# selector ablations do not answer that: the fixed-quantile and
# learnable-threshold variants both drop the EBM as well, so they remove two
# components rather than one, and direct PINN-EBM removes the gate by changing
# the objective to the NLL.
#
# This queue removes exactly one component. Warm-up runs, the EBM still learns
# the residual density, and the density still produces a per-measurement
# weight -- but the cutoff and steepness stay at their initial values and the
# rejection cost is zero, so the weighting is a fixed likelihood rule instead
# of a learned decision. The matched trainable-gate cells already exist in
# outputs/rebuttal/synthetic_recovery_20260726 and are not re-run.
set -u
GPU="${1:?usage: run_frozen_gate_queue.sh <gpu>}"
ROOT=outputs/rebuttal/frozen_gate_20260728

for SEED in 40 41 42; do
  for PDE in allencahn2d burgers2d lambdaomega2d; do
    echo "=== ${PDE} seed=${SEED} frozen gate ==="
    python -m scripts.rebuttal.run_synthetic \
      --experiment-name "${PDE}" \
      --method napinn \
      --seed "${SEED}" \
      --device "cuda:${GPU}" \
      --noise-kind 4G \
      --outlier-ratio 0.15 \
      --freeze-gate \
      --experiment-config "configs/rebuttal/${PDE}.yaml" \
      --output-root "${ROOT}"
  done
done
