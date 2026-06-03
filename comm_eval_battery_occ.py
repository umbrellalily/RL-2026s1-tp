"""Evaluate a battery + OCCUPANCY-aware policy (NEW FILE; existing files untouched).

Thin wrapper around comm_eval_battery.py: rebinds the env class it constructs
to the occupancy-augmented BatteryOccupancyShapeFormationEnv, then runs
comm_eval_battery.main() unchanged (same CLI, same GIF/stats output).

The checkpoint MUST be one trained with comm_train_battery_occ.py (occupancy
obs_dim). A base-env checkpoint will fail to load with a size mismatch.
"""
from __future__ import annotations

import comm_eval_battery
from comm_env_occupancy import BatteryOccupancyShapeFormationEnv

comm_eval_battery.BatteryShapeFormationEnv = BatteryOccupancyShapeFormationEnv

if __name__ == "__main__":
    comm_eval_battery.main()
