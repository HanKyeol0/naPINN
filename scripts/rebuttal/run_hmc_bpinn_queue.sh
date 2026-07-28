#!/usr/bin/env bash
# HMC B-PINN reporting campaign (Reviewer 6XZg).
#
# The sampler configuration below was fixed on calibration seed 39 using the
# acceptance rate only, before any error metric was inspected: with
# sigma_pde=1.0 the dual-averaging adaptation reached acceptance 0.560 at a
# step size of about 1.0e-5, whereas sigma_pde=0.1 collapsed the step size to
# ~1e-8 and could not move. Both sigma_data values are run and both are
# reported, so HMC is not represented by a single possibly-misspecified noise
# scale.
#
# Every run writes acceptance rate, split R-hat and effective sample size. If
# a run does not meet the predeclared thresholds, its error numbers must be
# reported as an inconclusive reproduction attempt, not as HMC performance.
set -u
GPU="${1:?usage: run_hmc_bpinn_queue.sh <gpu>}"
ROOT=outputs/rebuttal/hmc_bpinn_20260728

for SD in 0.1 1.0; do
  for SEED in 40 41 42; do
    echo "=== sigma_data=${SD} seed=${SEED} ==="
    python -m scripts.rebuttal.run_synthetic_hmc_bpinn \
      --seed "${SEED}" --device "cuda:${GPU}" \
      --sigma-data "${SD}" --sigma-pde 1.0 \
      --n-chains 2 --burn-in 1000 --n-samples 1000 --n-leapfrog 30 --thin 10 \
      --output-root "${ROOT}/sigma_data_${SD}"
  done
done
