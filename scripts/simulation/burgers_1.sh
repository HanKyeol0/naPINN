# run: scripts/simulation/burgers_1.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.burgers_simulation_1 \
  --config configs/experiment/burgers2d_1.yaml