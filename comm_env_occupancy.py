"""Occupancy-augmented formation environments (NEW FILE; comm_env.py untouched).

WHY
---
The "freeze one step short of a free target cell" / funnel-deadlock failure
(see demo_strict_91.gif) happens partly because a drone CANNOT directly tell
which formation slots are still empty. Its observation only contains:
  - its own position, all target-cell coordinates, OTHER drones' positions
    (droppable by comm_fail), its own assigned cell (+ battery).
It must *infer* occupancy from other drones' positions, which is noisy and is
hidden whenever a comm link drops. So a drone parked next to an empty cell has
no clean signal that the cell is free and worth entering -> it can settle into
a "stay" local optimum and the formation never completes.

WHAT
----
These subclasses append a per-target-cell OCCUPANCY vector to the observation:
  occupancy[i] = 1.0 if target_cells[i] is occupied by ANY drone right now,
                 else 0.0   (same ordering as env.target_cells)
appended at the END of the base observation (existing feature positions stay
put). Now the policy can directly see "which slots are filled / which are
free" and learn to head for / hold the remaining empty cells.

DESIGN CHOICES
--------------
1. GLOBAL signal: occupancy is always visible to every drone and is NOT subject
   to comm_fail_prob (think: a ground station broadcasts which formation slots
   are filled). The point is to give the policy the coordination info it lacked.
   A comm-fail-aware variant (count a cell only if its occupant has a live link)
   is possible but intentionally NOT used here -- see OCCUPANCY_CHANGES.txt.
2. obs_dim grows by n_agents (every formation has exactly n_agents target
   cells). => Networks trained on the base env are NOT load-compatible; train
   occupancy policies FROM SCRATCH.
3. Pure subclasses: all reward logic (incl. the new coverage_step_reward) and
   battery dynamics are inherited unchanged. Only the observation is extended.
"""
from __future__ import annotations

import numpy as np
from gymnasium import spaces

from comm_env import BatteryShapeFormationEnv, ShapeFormationEnv


def occupancy_vector(env) -> np.ndarray:
    """1.0 for each currently-occupied target cell, else 0.0 (target_cells order)."""
    occupied = set(env.agent_pos.values())
    return np.array(
        [1.0 if cell in occupied else 0.0 for cell in env.target_cells],
        dtype=np.float32,
    )


def _extend_obs_space(env) -> None:
    """Grow obs_dim by n_agents (one occupancy slot per target cell) and rebuild
    the observation Box. Call AFTER the base __init__ has set obs_dim/_obs_space."""
    env.obs_dim = env.obs_dim + env.n_agents
    env._obs_space = spaces.Box(
        low=0.0, high=1.0, shape=(env.obs_dim,), dtype=np.float32
    )


class OccupancyShapeFormationEnv(ShapeFormationEnv):
    """ShapeFormationEnv + global target-occupancy observation (battery-less)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _extend_obs_space(self)

    def _get_obs(self, agent: str) -> np.ndarray:
        return np.concatenate([super()._get_obs(agent), occupancy_vector(self)])


class BatteryOccupancyShapeFormationEnv(BatteryShapeFormationEnv):
    """BatteryShapeFormationEnv + global target-occupancy observation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _extend_obs_space(self)

    def _get_obs(self, agent: str) -> np.ndarray:
        return np.concatenate([super()._get_obs(agent), occupancy_vector(self)])
