"""Evaluate a battery-LESS + OCCUPANCY-aware policy (NEW FILE; originals untouched).

Thin wrapper around comm_eval.py: rebinds the env class it constructs to the
occupancy-augmented OccupancyShapeFormationEnv, then runs comm_eval.main()
unchanged (same CLI, same GIF/stats output). For the no-battery cases of the
unified matrix. The checkpoint MUST be occupancy-shaped (trained via
comm_train_occ.py); a base-env checkpoint fails to load (size mismatch).
"""
from __future__ import annotations

import comm_eval
from comm_env_occupancy import OccupancyShapeFormationEnv

comm_eval.ShapeFormationEnv = OccupancyShapeFormationEnv

if __name__ == "__main__":
    comm_eval.main()
