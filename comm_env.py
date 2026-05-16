"""Dec-POMDP shape-transition grid world with inter-agent position broadcast.

Setup
-----
- 15x15 grid, 10 drones, actions = {stay, up, down, left, right}.
- Formation coordinates are defined outside the environment in formation_seq.py.
- A comma-separated ``shapes`` path such as ``GROUND,X,I,-`` means:
      start at GROUND -> move to X -> move to I -> move to -
- GROUND is a bottom-row line, so with grid_size=15 and n_agents=10 the
  default start cells are (14, 2), ..., (14, 11).
- Observation:
    [ own (row, col) / grid_size                         ]  2
    [ current target cells (row, col) / grid_size         ]  2*n_agents
    [ for each other drone: (row, col, valid_flag)        ]  3*(n_agents-1)
    [ my assigned target cell (row, col) / grid_size      ]  2
- Drone<->target assignment is computed at the start of each stage by
  Hungarian algorithm using the current drone positions and current target.
- Reward: step penalty, on-target reward, stage completion reward, collision
  penalty, and per-drone shaping toward the assigned target.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from gymnasium import spaces
from pettingzoo import ParallelEnv

from formation_seq import Formation, FormationPath, SHAPES, build_shapes

# 0:stay, 1:up, 2:down, 3:left, 4:right
MOVES: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


class ShapeFormationEnv(ParallelEnv):
    metadata = {"name": "shape_transition_comm_v0", "is_parallelizable": True}

    def __init__(
        self,
        grid_size: int = 15,
        n_agents: int = 10,
        max_steps: int = 150,
        shapes: list[str] | None = None,
        target_shapes: list[str] | None = None,
        formation_path: FormationPath | None = None,
        comm_fail_prob: float = 0.0,
        completion_reward: float = 10.0,
        on_target_reward: float = 0.1,
        collision_penalty: float = 0.2,
        step_penalty: float = 0.01,
        shaping_coef: float = 0.3,
        ground_row: int | None = None,
        ground_start_col: int | None = None,
    ):
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.comm_fail_prob = comm_fail_prob
        self.completion_reward = completion_reward
        self.on_target_reward = on_target_reward
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.shaping_coef = shaping_coef
        self.ground_row = ground_row
        self.ground_start_col = ground_start_col

        # Backward-compatible naming: target_shapes is treated as the shapes path.
        if shapes is None and target_shapes is not None:
            shapes = target_shapes
        if shapes is None:
            shapes = ["GROUND", "X"]

        self.formation_path: FormationPath = formation_path or build_shapes(
            names=shapes,
            grid_size=grid_size,
            n_agents=n_agents,
            ground_row=ground_row,
            ground_start_col=ground_start_col,
        )
        self.shapes: list[str] = self.formation_path.names
        # Compatibility with older train/eval code that read target_shapes.
        self.target_shapes: list[str] = self.shapes

        self.possible_agents: list[str] = [f"drone_{i}" for i in range(n_agents)]
        self.agents: list[str] = list(self.possible_agents)

        # obs = own_pos(2) + target_cells(2*n_agents) + comm(3*(n_agents-1)) + assigned_target(2)
        self.obs_dim: int = 2 + 2 * n_agents + 3 * (n_agents - 1) + 2
        self._obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self._act_space = spaces.Discrete(5)

        self.agent_pos: dict[str, tuple[int, int]] = {}
        self.target_cells: list[tuple[int, int]] = []
        self._target_set: set[tuple[int, int]] = set()
        self.target_shape_name: str = ""
        self.stage_idx: int = 0
        self.stage_done_count: int = 0
        self.last_collision_count: int = 0
        self.step_count: int = 0
        self.assigned_target_cell: dict[str, tuple[int, int]] = {}
        self.prev_per_drone_dists: dict[str, float] = {}
        self.np_random: np.random.Generator = np.random.default_rng()

    # PettingZoo API ---------------------------------------------------------
    def observation_space(self, agent: str) -> spaces.Box:
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:
        return self._act_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.step_count = 0
        self.last_collision_count = 0
        self.stage_idx = 0
        self.stage_done_count = 0

        # Start from the first formation, normally GROUND.
        self.agent_pos = {
            agent: cell
            for agent, cell in zip(self.possible_agents, self.formation_path.start.cells)
        }

        # First target formation.
        self._set_target_formation(self.formation_path.targets[self.stage_idx])
        self._reset_assignment()

        return self._all_obs(), {a: {} for a in self.agents}

    def step(self, actions: dict[str, int]):
        self.step_count += 1

        # 1) Propose next positions (walls clip the move)
        proposed: dict[str, tuple[int, int]] = {}
        for a in self.possible_agents:
            r, c = self.agent_pos[a]
            dr, dc = MOVES[int(actions[a])]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                proposed[a] = (nr, nc)
            else:
                proposed[a] = (r, c)

        # 2) Resolve collisions (same-cell + swap conflicts -> both stay)
        counts = Counter(proposed.values())
        collisions: set[str] = set()
        new_pos: dict[str, tuple[int, int]] = {}
        for a, p in proposed.items():
            if counts[p] > 1:
                new_pos[a] = self.agent_pos[a]
                collisions.add(a)
                continue
            swap = False
            for b, pb in proposed.items():
                if b == a:
                    continue
                if p == self.agent_pos[b] and pb == self.agent_pos[a]:
                    swap = True
                    collisions.add(a)
                    collisions.add(b)
                    break
            new_pos[a] = self.agent_pos[a] if swap else p
        self.agent_pos = new_pos
        self.last_collision_count = len(collisions)

        # 3) Base rewards
        rewards = {a: -self.step_penalty for a in self.possible_agents}
        for a in collisions:
            rewards[a] -= self.collision_penalty

        occupied: set[tuple[int, int]] = set()
        for a, p in self.agent_pos.items():
            if p in self._target_set:
                rewards[a] += self.on_target_reward
                occupied.add(p)
        shape_done = len(occupied) == len(self.target_cells)

        # 4) Per-drone potential-based shaping toward the current assigned target.
        # This is computed before a possible stage transition, so the reward term
        # belongs to the stage that was active during this step.
        for a in self.possible_agents:
            tr, tc = self.assigned_target_cell[a]
            r, c = self.agent_pos[a]
            new_dist = float(abs(r - tr) + abs(c - tc))
            if self.shaping_coef != 0.0:
                rewards[a] += self.shaping_coef * (self.prev_per_drone_dists[a] - new_dist)
            self.prev_per_drone_dists[a] = new_dist

        # 5) Stage transition. Completing an intermediate shape advances the
        # target instead of ending the episode. Only the final target terminates.
        final_done = self._advance_stage_if_needed(shape_done, rewards)

        truncated = self.step_count >= self.max_steps
        terminations = {a: final_done for a in self.possible_agents}
        truncations = {a: (truncated and not final_done) for a in self.possible_agents}

        obs = self._all_obs()
        infos = {
            a: {
                "collision": (a in collisions),
                "shape_done": shape_done,
                "final_done": final_done,
                "stage_idx": self.stage_idx,
                "stages_completed": self.stage_done_count,
                "target_shape": self.target_shape_name,
                "shapes_path": self.formation_path.label,
            }
            for a in self.possible_agents
        }

        if final_done or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # Helpers ----------------------------------------------------------------
    def _set_target_formation(self, formation: Formation) -> None:
        self.target_shape_name = formation.name
        self.target_cells = list(formation.cells)
        self._target_set = set(self.target_cells)

    def _reset_assignment(self) -> None:
        self.assigned_target_cell = self._compute_assignment()
        self.prev_per_drone_dists = {
            a: float(
                abs(self.agent_pos[a][0] - self.assigned_target_cell[a][0])
                + abs(self.agent_pos[a][1] - self.assigned_target_cell[a][1])
            )
            for a in self.possible_agents
        }

    def _advance_stage_if_needed(
        self,
        shape_done: bool,
        rewards: dict[str, float],
    ) -> bool:
        """Advance to the next target formation.

        Returns True only when the final target in the shapes path is completed.
        """
        if not shape_done:
            return False

        self.stage_done_count += 1
        for a in self.possible_agents:
            rewards[a] += self.completion_reward

        is_final_stage = self.stage_idx >= len(self.formation_path.targets) - 1
        if is_final_stage:
            return True

        self.stage_idx += 1
        self._set_target_formation(self.formation_path.targets[self.stage_idx])
        self._reset_assignment()
        return False

    def _compute_assignment(self) -> dict[str, tuple[int, int]]:
        """Optimal 1:1 drone<->target assignment via Hungarian algorithm."""
        drones = [self.agent_pos[a] for a in self.possible_agents]
        targets = self.target_cells
        n = len(drones)

        cost_matrix = np.zeros((n, n), dtype=np.float32)
        for i, (dr, dc) in enumerate(drones):
            for j, (tr, tc) in enumerate(targets):
                cost_matrix[i, j] = abs(dr - tr) + abs(dc - tc)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return {
            self.possible_agents[int(row)]: targets[int(col)]
            for row, col in zip(row_ind, col_ind)
        }

    def _all_obs(self) -> dict[str, np.ndarray]:
        return {a: self._get_obs(a) for a in self.possible_agents}

    def _get_obs(self, agent: str) -> np.ndarray:
        gs = float(self.grid_size)
        r, c = self.agent_pos[agent]

        # (a) Own GPS position
        own = np.array([r / gs, c / gs], dtype=np.float32)

        # (b) Current target cells, canonical order, length = 2 * n_agents
        target_vec = np.empty(2 * self.n_agents, dtype=np.float32)
        for i, (tr, tc) in enumerate(self.target_cells):
            target_vec[2 * i] = tr / gs
            target_vec[2 * i + 1] = tc / gs

        # (c) Comm: receive other drones' positions w/ per-link Bernoulli loss
        comm = np.zeros(3 * (self.n_agents - 1), dtype=np.float32)
        i = 0
        for other in self.possible_agents:
            if other == agent:
                continue
            if self.np_random.random() >= self.comm_fail_prob:
                ro, co = self.agent_pos[other]
                comm[3 * i] = ro / gs
                comm[3 * i + 1] = co / gs
                comm[3 * i + 2] = 1.0
            i += 1

        # (d) My assigned target cell for the current stage
        atr, atc = self.assigned_target_cell[agent]
        assigned = np.array([atr / gs, atc / gs], dtype=np.float32)

        return np.concatenate([own, target_vec, comm, assigned])

    def render(self) -> None:
        grid = [["."] * self.grid_size for _ in range(self.grid_size)]
        for (r, c) in self.target_cells:
            grid[r][c] = "x"
        for a, (r, c) in self.agent_pos.items():
            grid[r][c] = a[-1]
        print(
            f"shapes='{self.formation_path.label}'  "
            f"stage={self.stage_idx + 1}/{len(self.formation_path.targets)}  "
            f"target='{self.target_shape_name}'"
        )
        print("\n".join("".join(row) for row in grid))
        print()
