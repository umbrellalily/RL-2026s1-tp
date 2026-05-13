"""Dec-POMDP shape-formation grid world with multi-shape targets and inter-agent
position broadcast ("night mode" drone show).

Setup
-----
- 15x15 grid, 5 drones, actions = {stay, up, down, left, right}.
- Each episode samples a random target shape from `target_shapes`. Five-cell
  shapes are defined in `SHAPES` below. Cells are spaced 2 apart (one empty
  cell between adjacent shape cells) so drones don't have to crowd at
  targets.
- Drones cannot see each other visually (night). They share their own GPS
  positions over a broadcast channel; each link may fail with prob
  `comm_fail_prob` (per receiver, per sender, per step).
- Observation (26-dim flat vector for n_agents=5):
    [ own (row, col) / grid_size                                    ]  2
    [ target cells (row, col) / grid_size, sorted, 5 cells          ] 10
    [ for each of 4 other drones: (row, col, valid_flag)            ] 12
    [ my assigned target cell (row, col) / grid_size                ]  2
- Drone<->target assignment: computed once at reset via Hungarian (brute
  force over 5! perms) using initial positions. Stable for the episode.
  Future-extensible: weight Hungarian cost by battery etc. to make
  low-battery drones get nearer cells automatically.
- Reward: step -0.01, on-target +0.1, completion +10, collision -0.2,
  plus per-drone shaping = shaping_coef * (prev_d_i - new_d_i) where d_i
  is drone i's Manhattan distance to its assigned target cell.
"""
from __future__ import annotations

from collections import Counter
from itertools import permutations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# 0:stay, 1:up, 2:down, 3:left, 4:right
MOVES: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


# Five-cell target shapes on a 15x15 grid, centered around (7, 7). Cells are
# spaced 2 apart (1 empty cell between adjacent shape cells) so drones don't
# have to crowd. All shapes use exactly 5 cells, pre-sorted (row, col)
# ascending so the observation vector has a canonical ordering across shapes.
SHAPES: dict[str, list[tuple[int, int]]] = {
    "I": [(3, 7), (5, 7), (7, 7), (9, 7), (11, 7)],   # vertical bar
    "-": [(7, 3), (7, 5), (7, 7), (7, 9), (7, 11)],   # horizontal bar
    "L": [(5, 7), (7, 7), (9, 7), (11, 7), (11, 9)],
    "T": [(5, 5), (5, 7), (5, 9), (7, 7), (9, 7)],
    "+": [(5, 7), (7, 5), (7, 7), (7, 9), (9, 7)],
    "X": [(5, 5), (5, 9), (7, 7), (9, 5), (9, 9)],
}


class ShapeFormationEnv(ParallelEnv):
    metadata = {"name": "shape_formation_comm_v0", "is_parallelizable": True}

    def __init__(
        self,
        grid_size: int = 15,
        n_agents: int = 5,
        max_steps: int = 150,
        target_shapes: list[str] | None = None,
        comm_fail_prob: float = 0.0,
        completion_reward: float = 10.0,
        on_target_reward: float = 0.1,
        collision_penalty: float = 0.2,
        step_penalty: float = 0.01,
        shaping_coef: float = 0.3,
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

        # Validate / default target shapes
        if target_shapes is None:
            target_shapes = list(SHAPES.keys())
        for s in target_shapes:
            if s not in SHAPES:
                raise ValueError(f"Unknown shape {s!r}; available: {list(SHAPES)}")
            if len(SHAPES[s]) != n_agents:
                raise ValueError(
                    f"Shape {s!r} has {len(SHAPES[s])} cells but n_agents={n_agents}"
                )
        self.target_shapes: list[str] = list(target_shapes)

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

        # Sample a target shape for this episode
        idx = int(self.np_random.integers(0, len(self.target_shapes)))
        self.target_shape_name = self.target_shapes[idx]
        self.target_cells = list(SHAPES[self.target_shape_name])
        self._target_set = set(self.target_cells)

        # Random initial positions (distinct cells)
        cells = self.np_random.choice(
            self.grid_size * self.grid_size, size=self.n_agents, replace=False
        )
        self.agent_pos = {
            a: (int(c // self.grid_size), int(c % self.grid_size))
            for a, c in zip(self.possible_agents, cells)
        }
        self.assigned_target_cell = self._compute_assignment()
        self.prev_per_drone_dists = {
            a: float(
                abs(self.agent_pos[a][0] - self.assigned_target_cell[a][0])
                + abs(self.agent_pos[a][1] - self.assigned_target_cell[a][1])
            )
            for a in self.possible_agents
        }
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

        # 3) Rewards
        rewards = {a: -self.step_penalty for a in self.possible_agents}
        for a in collisions:
            rewards[a] -= self.collision_penalty

        occupied: set[tuple[int, int]] = set()
        for a, p in self.agent_pos.items():
            if p in self._target_set:
                rewards[a] += self.on_target_reward
                occupied.add(p)
        shape_done = len(occupied) == len(self.target_cells)
        if shape_done:
            for a in self.possible_agents:
                rewards[a] += self.completion_reward

        # Per-drone potential-based shaping: each drone gets credit for
        # closing the L1 gap to ITS assigned target cell (assignment was
        # fixed at reset by Hungarian on initial positions).
        for a in self.possible_agents:
            tr, tc = self.assigned_target_cell[a]
            r, c = self.agent_pos[a]
            new_dist = float(abs(r - tr) + abs(c - tc))
            if self.shaping_coef != 0.0:
                rewards[a] += self.shaping_coef * (self.prev_per_drone_dists[a] - new_dist)
            self.prev_per_drone_dists[a] = new_dist

        truncated = self.step_count >= self.max_steps
        terminations = {a: shape_done for a in self.possible_agents}
        truncations = {a: (truncated and not shape_done) for a in self.possible_agents}

        obs = self._all_obs()
        infos = {
            a: {
                "collision": (a in collisions),
                "shape_done": shape_done,
            }
            for a in self.possible_agents
        }

        if shape_done or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # Helpers ----------------------------------------------------------------
    def _compute_assignment(self) -> dict[str, tuple[int, int]]:
        """Optimal 1:1 drone<->target assignment via brute-force Hungarian.

        Picks the permutation of target_cells that minimizes total L1 cost
        from current agent_pos. Future extension point: weight the cost by
        battery state, urgency, etc.
        """
        drones = [self.agent_pos[a] for a in self.possible_agents]
        targets = self.target_cells
        n = len(drones)
        best_cost = float("inf")
        best_perm: tuple[int, ...] = tuple(range(n))
        for perm in permutations(range(n)):
            cost = 0
            for i, j in enumerate(perm):
                dr, dc = drones[i]
                tr, tc = targets[j]
                cost += abs(dr - tr) + abs(dc - tc)
                if cost >= best_cost:
                    break
            else:
                # only reached if inner loop didn't break early
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm
        return {
            self.possible_agents[i]: targets[best_perm[i]] for i in range(n)
        }

    def _all_obs(self) -> dict[str, np.ndarray]:
        return {a: self._get_obs(a) for a in self.possible_agents}

    def _get_obs(self, agent: str) -> np.ndarray:
        gs = float(self.grid_size)
        r, c = self.agent_pos[agent]

        # (a) Own GPS position
        own = np.array([r / gs, c / gs], dtype=np.float32)

        # (b) Target cells (sorted canonical order, length = 2 * n_agents)
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
            # else: failed link -> leave (0, 0, 0), valid flag = 0
            i += 1

        # (d) My assigned target cell (dynamic per-episode via Hungarian)
        atr, atc = self.assigned_target_cell[agent]
        assigned = np.array([atr / gs, atc / gs], dtype=np.float32)

        return np.concatenate([own, target_vec, comm, assigned])

    def render(self) -> None:
        grid = [["."] * self.grid_size for _ in range(self.grid_size)]
        for (r, c) in self.target_cells:
            grid[r][c] = "x"
        for a, (r, c) in self.agent_pos.items():
            grid[r][c] = a[-1]
        print(f"target = '{self.target_shape_name}'")
        print("\n".join("".join(row) for row in grid))
        print()
