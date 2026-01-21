# run: scripts/simulation/lambda_omega_rd.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.lambda_omega_rd_simulation \
  --config configs/experiment/lambda_omega_rd2d.yaml
