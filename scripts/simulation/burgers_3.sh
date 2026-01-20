# run: scripts/simulation/burgers_3.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.burgers_simulation_3 \
  --config configs/experiment/burgers2d_3.yaml