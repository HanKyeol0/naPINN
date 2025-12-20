# run: scripts/simulation/burgers_2.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.burgers_simulation_2 \
  --config configs/experiment/burgers2d_2.yaml