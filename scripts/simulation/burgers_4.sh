# run: scripts/simulation/burgers_4.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.burgers_simulation_4 \
  --config configs/experiment/burgers2d_4.yaml