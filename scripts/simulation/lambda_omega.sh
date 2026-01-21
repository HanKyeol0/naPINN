# run: scripts/simulation/lambda_omega.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.lambda_omega_simulation \
  --config configs/experiment/lambda_omega2d.yaml