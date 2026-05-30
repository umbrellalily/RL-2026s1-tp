"""Train a battery + OCCUPANCY-aware policy (NEW FILE; existing files untouched).

Thin wrapper around comm_train_battery.py: it rebinds the env class that
comm_train_battery constructs (make_env + the probe both reference the
module-global ``BatteryShapeFormationEnv``) to the occupancy-augmented
``BatteryOccupancyShapeFormationEnv``, then runs comm_train_battery.main()
unchanged. Every CLI flag of comm_train_battery.py is accepted as-is.

Because the occupancy env adds n_agents observation dims, obs_dim grows
(e.g. 14 agents: 85 -> 99) and the model is sized from probe.obs_dim, so the
saved checkpoints are occupancy-shaped. Evaluate them with comm_eval_*_occ.py.
"""
from __future__ import annotations

import comm_train_battery
from comm_env_occupancy import BatteryOccupancyShapeFormationEnv

# Redirect every `BatteryShapeFormationEnv(...)` construction inside
# comm_train_battery (make_env + probe) to the occupancy env, without editing it.
comm_train_battery.BatteryShapeFormationEnv = BatteryOccupancyShapeFormationEnv

if __name__ == "__main__":
    comm_train_battery.main()
