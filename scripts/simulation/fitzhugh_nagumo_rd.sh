# run: scripts/simulation/fitzhugh_nagumo_rd.sh

#!/usr/bin/env bash
set -e

python -m pinnlab.simulation.fitzhugh_nagumo_simulation \
  --config configs/experiment/fitzhugh_nagumo_rd2d.yaml
