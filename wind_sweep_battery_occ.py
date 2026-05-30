"""Wind sweep for OCCUPANCY-aware policies (NEW FILE; existing files untouched).

wind_sweep_battery.py builds its envs through comm_eval_battery.make_env, which
references the module-global comm_eval_battery.BatteryShapeFormationEnv. We
rebind that to the occupancy env BEFORE running wind_sweep_battery.main(), so
the sweep loads occupancy-shaped checkpoints correctly. Same CLI as
wind_sweep_battery.py.
"""
from __future__ import annotations

import comm_eval_battery
from comm_env_occupancy import BatteryOccupancyShapeFormationEnv

# Patch the class that comm_eval_battery.make_env (used by wind_sweep_battery)
# constructs, before importing/calling the sweep's main().
comm_eval_battery.BatteryShapeFormationEnv = BatteryOccupancyShapeFormationEnv

import wind_sweep_battery  # noqa: E402  (import after patch on purpose)

if __name__ == "__main__":
    wind_sweep_battery.main()
