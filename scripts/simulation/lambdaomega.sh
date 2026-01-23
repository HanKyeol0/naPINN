# run: scripts/simulation/lambdaomega.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.lambdaomega_simulation \
  --config configs/experiment/lambdaomega2d.yaml