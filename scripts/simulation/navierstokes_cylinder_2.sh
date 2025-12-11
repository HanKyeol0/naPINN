# run: scripts/simulation/navierstokes_cylinder.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.navierstokes_cylinder_simulation_2 \
  --config configs/experiment/navierstokes2d_cylinder.yaml