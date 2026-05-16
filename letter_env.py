"""Dec-POMDP letter-formation grid world with VARIABLE letter sizes (v5).

Key difference from v4
----------------------
v4: every letter exactly 12 cells, drone fleet = 12.
v5: each letter uses its NATURAL cell count (10..20 currently); drone fleet
    = MAX_CELLS (20), and the remaining drones are sent to "wait cells" along
    the grid edge each episode.

How wait cells work
-------------------
For a letter with k lit cells, the env creates (MAX_CELLS - k) wait cells.
Wait cells are placed along the grid border, deterministically spaced so they
do not interfere with the letter's interior. The full target set
(letter cells + wait cells) has exactly MAX_CELLS cells, so the rest of the
pipeline (Hungarian 1:1 assignment, per-drone shaping, completion bonus on
"all targets occupied") works without changes.

A drone is considered "useful" if it ends up on a letter cell. A drone on a
wait cell counts toward completion but does not visually contribute to the
letter -- this is fine for learning purposes and matches the spec of having
a fixed fleet that scales to the largest letter.

Observation per agent:
    own (row, col) / G                                   ->  2
    target cells (row, col) / G, MAX_CELLS of them        -> 2*MAX_CELLS
    each other agent: (row, col, valid_flag)              -> 3*(MAX_CELLS-1)
    my assigned target cell (row, col) / G                ->  2
    total = 2 + 2*MAX_CELLS + 3*(MAX_CELLS-1) + 2
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from gymnasium import spaces
from pettingzoo import ParallelEnv

from letters import LETTERS, MAX_CELLS, letter_bbox, letter_cells

# 0:stay, 1:up, 2:down, 3:left, 4:right
MOVES: dict[int, tuple[int, int]] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


def _border_wait_cells(grid_size: int, k: int, letter_set: set) -> list[tuple[int, int]]:
    """Generate k wait cells along the grid border, skipping letter cells.

    Cells are picked deterministically: start from corner (0,0), walk along
    the border clockwise, taking every nth empty border cell so they spread
    out. With MAX_CELLS=20 and typical letter k<=20, we need <=10 wait cells
    which fit comfortably on a 4*(grid_size-1) border.
    """
    if k <= 0:
        return []

    # Enumerate border cells clockwise: top row, right col, bottom row, left col
    g = grid_size
    border = []
    border.extend((0, c) for c in range(g))            # top
    border.extend((r, g - 1) for r in range(1, g))     # right
    border.extend((g - 1, c) for c in range(g - 2, -1, -1))  # bottom
    border.extend((r, 0) for r in range(g - 2, 0, -1))  # left

    available = [p for p in border if p not in letter_set]
    if k > len(available):
        # Should not happen for reasonable grid_size; fall back to whatever exists
        k = len(available)

    # Evenly spaced selection
    stride = max(1, len(available) // k)
    chosen = []
    seen = set()
    idx = 0
    while len(chosen) < k and idx < len(available) * 2:
        cell = available[idx % len(available)]
        if cell not in seen:
            chosen.append(cell)
            seen.add(cell)
        idx += stride
    # Fallback: fill remaining if stride missed some
    if len(chosen) < k:
        for cell in available:
            if cell not in seen:
                chosen.append(cell)
                seen.add(cell)
                if len(chosen) >= k:
                    break
    return chosen[:k]


def place_letter(name: str, origin: tuple[int, int]) -> list[tuple[int, int]]:
    """Absolute (row, col) cells for `name` placed at `origin` (top-left)."""
    orow, ocol = origin
    return sorted([(r + orow, c + ocol) for (r, c) in letter_cells(name)])


def build_targets(
    name: str, origin: tuple[int, int], grid_size: int
) -> tuple[list[tuple[int, int]], int]:
    """Return (target_cells, n_letter_cells).

    target_cells has length MAX_CELLS = letter cells + wait cells.
    n_letter_cells is the count of actual glyph cells (the rest are wait).
    """
    letter_abs = place_letter(name, origin)
    n_letter = len(letter_abs)
    n_wait = MAX_CELLS - n_letter
    wait = _border_wait_cells(grid_size, n_wait, set(letter_abs))
    all_targets = letter_abs + wait
    return sorted(all_targets), n_letter


class LetterFormationEnv(ParallelEnv):
    metadata = {"name": "letter_formation_v5", "is_parallelizable": True}

    def __init__(
        self,
        grid_size: int = 24,
        max_steps: int = 250,
        target_letters: list[str] | None = None,
        letter_origin: tuple[int, int] | None = None,
        comm_fail_prob: float = 0.0,
        completion_reward: float = 10.0,
        on_target_reward: float = 0.1,
        collision_penalty: float = 0.2,
        step_penalty: float = 0.01,
        shaping_coef: float = 0.3,
    ):
        self.grid_size = grid_size
        self.n_agents = MAX_CELLS
        self.max_steps = max_steps
        self.comm_fail_prob = comm_fail_prob
        self.completion_reward = completion_reward
        self.on_target_reward = on_target_reward
        self.collision_penalty = collision_penalty
        self.step_penalty = step_penalty
        self.shaping_coef = shaping_coef

        if target_letters is None:
            target_letters = list(LETTERS.keys())
        for s in target_letters:
            if s not in LETTERS:
                raise ValueError(f"Unknown letter {s!r}; available: {list(LETTERS)}")
        self.target_letters: list[str] = list(target_letters)

        self.letter_origin = letter_origin
        self._check_origin_fits()

        self.possible_agents: list[str] = [
            f"drone_{i}" for i in range(self.n_agents)
        ]
        self.agents: list[str] = list(self.possible_agents)

        # obs = own(2) + targets(2*n) + comm(3*(n-1)) + assigned(2)
        n = self.n_agents
        self.obs_dim: int = 2 + 2 * n + 3 * (n - 1) + 2
        self._obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        self._act_space = spaces.Discrete(5)

        # Episode state
        self.agent_pos: dict[str, tuple[int, int]] = {}
        self.target_cells: list[tuple[int, int]] = []
        self._target_set: set[tuple[int, int]] = set()
        self.target_letter_name: str = ""
        self.target_origin: tuple[int, int] = (0, 0)
        self.n_letter_cells: int = 0
        self.last_collision_count: int = 0
        self.step_count: int = 0
        self.assigned_target_cell: dict[str, tuple[int, int]] = {}
        self.prev_per_drone_dists: dict[str, float] = {}
        self.np_random: np.random.Generator = np.random.default_rng()

    def _check_origin_fits(self) -> None:
        if self.letter_origin is None:
            return
        orow, ocol = self.letter_origin
        for name in (self.target_letters or list(LETTERS)):
            h, w = letter_bbox(name)
            if not (0 <= orow and orow + h <= self.grid_size
                    and 0 <= ocol and ocol + w <= self.grid_size):
                raise ValueError(
                    f"letter_origin={self.letter_origin} places letter "
                    f"'{name}' (bbox {h}x{w}) out of a "
                    f"{self.grid_size}x{self.grid_size} grid."
                )

    def _sample_origin(self, name: str) -> tuple[int, int]:
        if self.letter_origin is not None:
            return self.letter_origin
        h, w = letter_bbox(name)
        # Keep a 1-cell margin from the border so wait cells have room
        max_orow = self.grid_size - h - 1
        max_ocol = self.grid_size - w - 1
        orow = int(self.np_random.integers(1, max(2, max_orow + 1)))
        ocol = int(self.np_random.integers(1, max(2, max_ocol + 1)))
        return (orow, ocol)

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

        idx = int(self.np_random.integers(0, len(self.target_letters)))
        self.target_letter_name = self.target_letters[idx]
        self.target_origin = self._sample_origin(self.target_letter_name)
        self.target_cells, self.n_letter_cells = build_targets(
            self.target_letter_name, self.target_origin, self.grid_size
        )
        self._target_set = set(self.target_cells)

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

        proposed: dict[str, tuple[int, int]] = {}
        for a in self.possible_agents:
            r, c = self.agent_pos[a]
            dr, dc = MOVES[int(actions[a])]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                proposed[a] = (nr, nc)
            else:
                proposed[a] = (r, c)

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

        for a in self.possible_agents:
            tr, tc = self.assigned_target_cell[a]
            r, c = self.agent_pos[a]
            new_dist = float(abs(r - tr) + abs(c - tc))
            if self.shaping_coef != 0.0:
                rewards[a] += self.shaping_coef * (
                    self.prev_per_drone_dists[a] - new_dist
                )
            self.prev_per_drone_dists[a] = new_dist

        truncated = self.step_count >= self.max_steps
        terminations = {a: shape_done for a in self.possible_agents}
        truncations = {
            a: (truncated and not shape_done) for a in self.possible_agents
        }

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

    def _compute_assignment(self) -> dict[str, tuple[int, int]]:
        drones = [self.agent_pos[a] for a in self.possible_agents]
        targets = self.target_cells
        n = len(drones)
        cost_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            dr, dc = drones[i]
            for j in range(n):
                tr, tc = targets[j]
                cost_matrix[i, j] = abs(dr - tr) + abs(dc - tc)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return {
            self.possible_agents[i]: targets[col_ind[i]] for i in range(n)
        }

    def _all_obs(self) -> dict[str, np.ndarray]:
        return {a: self._get_obs(a) for a in self.possible_agents}

    def _get_obs(self, agent: str) -> np.ndarray:
        gs = float(self.grid_size)
        r, c = self.agent_pos[agent]

        own = np.array([r / gs, c / gs], dtype=np.float32)

        target_vec = np.empty(2 * self.n_agents, dtype=np.float32)
        for i, (tr, tc) in enumerate(self.target_cells):
            target_vec[2 * i] = tr / gs
            target_vec[2 * i + 1] = tc / gs

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

        atr, atc = self.assigned_target_cell[agent]
        assigned = np.array([atr / gs, atc / gs], dtype=np.float32)

        return np.concatenate([own, target_vec, comm, assigned])

    def render(self) -> None:
        grid = [["."] * self.grid_size for _ in range(self.grid_size)]
        # letter cells = 'x', wait cells = 'o'
        letter_cells_abs = set(place_letter(self.target_letter_name, self.target_origin))
        for (r, c) in self.target_cells:
            grid[r][c] = "x" if (r, c) in letter_cells_abs else "o"
        for a, (r, c) in self.agent_pos.items():
            grid[r][c] = a.split("_")[-1][-1]
        print(
            f"letter='{self.target_letter_name}'  origin={self.target_origin}  "
            f"glyph_cells={self.n_letter_cells}  "
            f"wait_cells={self.n_agents - self.n_letter_cells}  "
            f"step={self.step_count}"
        )
        print("(x = letter cell, o = wait cell)")
        print("\n".join("".join(row) for row in grid))
        print()
