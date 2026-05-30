"""Train a battery-LESS + OCCUPANCY-aware policy (NEW FILE; existing files untouched).

Thin wrapper around comm_train.py: rebinds the env class it constructs
(make_env + probe reference the module-global ``ShapeFormationEnv``) to the
occupancy-augmented ``OccupancyShapeFormationEnv``, then runs comm_train.main()
unchanged. Every CLI flag of comm_train.py is accepted as-is.

Used for the no-battery cases of the unified training matrix. obs_dim grows by
n_agents (e.g. 14 agents: 71 -> 85); evaluate with comm_eval_strict_occ.py
(without --battery) or a base-env eval rebound to OccupancyShapeFormationEnv.
"""
from __future__ import annotations

import comm_train
from comm_env_occupancy import OccupancyShapeFormationEnv

comm_train.ShapeFormationEnv = OccupancyShapeFormationEnv

if __name__ == "__main__":
    comm_train.main()
