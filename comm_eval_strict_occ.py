"""Strict-termination eval of an OCCUPANCY-aware policy (NEW FILE; originals untouched).

Thin wrapper around comm_eval_strict.py: rebinds BOTH env classes it can use
(battery-less ShapeFormationEnv and battery BatteryShapeFormationEnv) to their
occupancy-augmented subclasses, then runs comm_eval_strict.main() unchanged.

Use exactly like comm_eval_strict.py:
  - battery-less occupancy policy:  python comm_eval_strict_occ.py --ckpt ...
  - battery occupancy policy:        python comm_eval_strict_occ.py --battery --ckpt ...

The checkpoint MUST be occupancy-shaped (trained via *_occ training).
"""
from __future__ import annotations

import comm_eval_strict
from comm_env_occupancy import (
    BatteryOccupancyShapeFormationEnv,
    OccupancyShapeFormationEnv,
)

comm_eval_strict.BatteryShapeFormationEnv = BatteryOccupancyShapeFormationEnv
comm_eval_strict.ShapeFormationEnv = OccupancyShapeFormationEnv

if __name__ == "__main__":
    comm_eval_strict.main()
