#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 {mse|lad|pinn_ebm|pinn_ebm_equal_weight|napinn|napinn_scaled_rejection} SEED DEVICE [SMOKE_STEPS]" >&2
  exit 2
fi

method="$1"
seed="$2"
device="$3"
smoke_steps="${4:-}"
case "$method" in
  mse|lad|pinn_ebm|pinn_ebm_equal_weight|napinn|napinn_scaled_rejection) ;;
  *)
    echo "Unknown method: $method" >&2
    exit 2
    ;;
esac

args=(
  python -m scripts.rebuttal.run_realpdebench
  --common-config configs/experiment/realpdebench_cylinder_common.yaml
  --model-config configs/model/mlp.yaml
  --exp-config "configs/experiment/realpdebench_cylinder_${method}.yaml"
  --seed "$seed"
  --device "$device"
)
if [[ -n "$smoke_steps" ]]; then
  args+=(--smoke-steps "$smoke_steps")
fi
"${args[@]}"
