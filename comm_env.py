"""Dec-POMDP shape-transition grid world with inter-agent position broadcast.

Default setup
-------------
- 25x25 grid, 14 drones, actions = {stay, up, down, left, right}.
- Formation coordinates are defined outside the environment in formation_seq.py.
- A comma-separated ``shapes`` path such as ``GROUND,X,I,-`` means:
      start at GROUND -> move to X -> move to I -> move to -
- GROUND is a bottom-row line. With grid_size=25 and n_agents=14, the
  default start cells are (24, 5), ..., (24, 18).
- Observation:
    [ own (row, col) / grid_size                         ]  2
    [ current target cells (row, col) / grid_size         ]  2*n_agents
    [ for each other drone: (row, col, valid_flag)        ]  3*(n_agents-1)
    [ my assigned target cell (row, col) / grid_size      ]  2
- Drone<->target assignment is computed at the start of each stage by
  Hungarian algorithm using the current drone positions and current target.
- Reward includes:
    step penalty, collision penalty, on-target reward, exact assigned-target
    reward, coverage-delta team reward, stage completion reward, and shaping
    toward the assigned target.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from scipy.optimize import linear_sum_assignment

from formation_seq import Formation, FormationPath, build_shapes

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
        grid_size: int = 25,
        n_agents: int = 14,
        max_steps: int = 250,
        shapes: list[str] | None = None,
        target_shapes: list[str] | None = None,
        formation_path: FormationPath | None = None,
        comm_fail_prob: float = 0.0,
        completion_reward: float = 30.0,
        on_target_reward: float = 0.1,
        collision_penalty: float = 0.2,
        step_penalty: float = 0.01,
        shaping_coef: float = 0.3,
        assigned_target_reward: float = 0.3,
        coverage_delta_reward: float = 0.2,
        coverage_step_reward: float = 0.0,
        hover_penalty: float = 0.02,
        ground_row: int | None = None,
        ground_start_col: int | None = None,
        wind_prob: float = 0.0,
        wind_strength: int = 1,
        wind_dir: tuple[int, int] | None = None,
        randomize_wind: bool = False,
        randomize_comm_fail: bool = False,
        random_shape_pool: list[str] | None = None,
        random_path_length: int | tuple[int, int] | str = 3,
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
        self.assigned_target_reward = assigned_target_reward
        self.coverage_delta_reward = coverage_delta_reward
        self.coverage_step_reward = coverage_step_reward
        self.hover_penalty = hover_penalty
        self.ground_row = ground_row
        self.ground_start_col = ground_start_col

        # Wind disturbance (external robustness factor). wind_prob == 0 disables
        # it entirely, so the default behaviour is unchanged. wind_dir is a
        # (row, col) unit vector; None means a random cardinal direction per
        # episode. randomize_wind samples this episode's severity from
        # [0, wind_prob] for domain randomization.
        self.wind_prob = wind_prob
        self.wind_strength = wind_strength
        self.wind_dir = wind_dir
        self.randomize_wind = randomize_wind
        self._cur_wind_prob: float = 0.0
        self._cur_wind_dir: tuple[int, int] = (0, 0)

        # Communication-failure domain randomization. When randomize_comm_fail
        # is True, each episode samples its drop rate from [0, comm_fail_prob];
        # otherwise the env behaves exactly as before (cur == comm_fail_prob).
        self.randomize_comm_fail = randomize_comm_fail
        self._cur_comm_fail_prob: float = float(comm_fail_prob)

        # Per-step symmetric link state: one Bernoulli per unordered drone pair
        # decides whether the link is alive this step, so a dropped link hides
        # traffic in both directions (radio reciprocity). Refilled in _all_obs.
        self._cur_link_alive: dict[frozenset[str], bool] = {}

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
        self.target_shapes: list[str] = self.shapes  # compatibility

        # Per-episode random shape sequence (for generalization training).
        # If random_shape_pool is given, reset() rebuilds formation_path each
        # episode by sampling random_path_length targets from the pool, prepended
        # with GROUND. random_path_length may be an int, (lo, hi) tuple, or
        # "lo-hi" / "N" string.
        self.random_shape_pool: list[str] | None = (
            list(random_shape_pool) if random_shape_pool else None
        )
        if isinstance(random_path_length, str):
            if "-" in random_path_length:
                lo_s, hi_s = random_path_length.split("-")
                self._rand_len_range = (int(lo_s), int(hi_s))
            else:
                n = int(random_path_length)
                self._rand_len_range = (n, n)
        elif isinstance(random_path_length, tuple):
            self._rand_len_range = (int(random_path_length[0]), int(random_path_length[1]))
        else:
            n = int(random_path_length)
            self._rand_len_range = (n, n)

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
        # Agents that collided on the most recent step (same-cell or swap).
        # Exposed for visualization/eval; training does not read it.
        self.last_collision_agents: set[str] = set()
        self.step_count: int = 0
        self.assigned_target_cell: dict[str, tuple[int, int]] = {}
        self.prev_per_drone_dists: dict[str, float] = {}
        self.best_occupied_count: int = 0
        self.last_occupied_count: int = 0
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
        self.last_collision_agents = set()
        self.stage_idx = 0
        self.stage_done_count = 0

        # Sample this episode's wind condition (direction fixed for the episode,
        # occurrence drawn per step in step()).
        if self.wind_prob > 0.0:
            self._cur_wind_prob = (
                float(self.np_random.uniform(0.0, self.wind_prob))
                if self.randomize_wind
                else self.wind_prob
            )
            if self.wind_dir is not None:
                self._cur_wind_dir = self.wind_dir
            else:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                self._cur_wind_dir = dirs[int(self.np_random.integers(4))]
        else:
            self._cur_wind_prob = 0.0
            self._cur_wind_dir = (0, 0)

        # Sample this episode's communication failure rate. With
        # randomize_comm_fail, drop rate is drawn from [0, comm_fail_prob];
        # otherwise it is fixed to comm_fail_prob (original behaviour).
        if self.comm_fail_prob > 0.0 and self.randomize_comm_fail:
            self._cur_comm_fail_prob = float(
                self.np_random.uniform(0.0, self.comm_fail_prob)
            )
        else:
            self._cur_comm_fail_prob = float(self.comm_fail_prob)

        # If a random shape pool is configured, sample a fresh formation path
        # for this episode. Episode always starts at GROUND so the start cells
        # match self.n_agents.
        if self.random_shape_pool:
            lo, hi = self._rand_len_range
            k = int(self.np_random.integers(lo, hi + 1))
            sampled = [
                self.random_shape_pool[int(self.np_random.integers(len(self.random_shape_pool)))]
                for _ in range(k)
            ]
            names = ["GROUND"] + sampled
            self.formation_path = build_shapes(
                names=names,
                grid_size=self.grid_size,
                n_agents=self.n_agents,
                ground_row=self.ground_row,
                ground_start_col=self.ground_start_col,
            )
            self.shapes = self.formation_path.names
            self.target_shapes = self.shapes

        # Start from the first formation, normally GROUND.
        if len(self.formation_path.start.cells) != self.n_agents:
            raise ValueError("start formation size must equal n_agents")
        self.agent_pos = {
            agent: cell
            for agent, cell in zip(self.possible_agents, self.formation_path.start.cells)
        }

        # First target formation.
        self._set_target_formation(self.formation_path.targets[self.stage_idx])
        self._reset_assignment()

        return self._all_obs(), {agent: {} for agent in self.agents}

    def step(self, actions: dict[str, int]):
        self.step_count += 1

        # 1) Propose next positions: action + wind gust, walls clip the move.
        proposed: dict[str, tuple[int, int]] = {}
        wind_dr, wind_dc = self._cur_wind_dir
        for agent in self.possible_agents:
            r, c = self.agent_pos[agent]
            dr, dc = MOVES[int(actions[agent])]
            # Wind perturbs the actual displacement (per-drone Bernoulli draw).
            if self._cur_wind_prob > 0.0 and self.np_random.random() < self._cur_wind_prob:
                dr += wind_dr * self.wind_strength
                dc += wind_dc * self.wind_strength
            nr, nc = r + dr, c + dc
            proposed[agent] = (
                (nr, nc) if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size else (r, c)
            )

        # 2) Resolve collisions (same-cell + swap conflicts -> both stay).
        counts = Counter(proposed.values())
        collisions: set[str] = set()
        new_pos: dict[str, tuple[int, int]] = {}
        for agent, pos in proposed.items():
            if counts[pos] > 1:
                new_pos[agent] = self.agent_pos[agent]
                collisions.add(agent)
                continue
            swap = False
            for other, other_pos in proposed.items():
                if other == agent:
                    continue
                if pos == self.agent_pos[other] and other_pos == self.agent_pos[agent]:
                    swap = True
                    collisions.add(agent)
                    collisions.add(other)
                    break
            new_pos[agent] = self.agent_pos[agent] if swap else pos
        self.agent_pos = new_pos
        self.last_collision_count = len(collisions)
        self.last_collision_agents = set(collisions)

        # 3) Base rewards.
        rewards = {agent: -self.step_penalty for agent in self.possible_agents}
        for agent in collisions:
            rewards[agent] -= self.collision_penalty

        occupied: set[tuple[int, int]] = set()
        for agent, pos in self.agent_pos.items():
            if pos in self._target_set:
                rewards[agent] += self.on_target_reward
                occupied.add(pos)
            if pos == self.assigned_target_cell[agent]:
                rewards[agent] += self.assigned_target_reward

        covered_count = len(occupied)
        self.last_occupied_count = covered_count

        # Dense per-step coverage reward: a small shared reward proportional to
        # how many target cells are occupied RIGHT NOW. Unlike coverage_delta_reward
        # (paid once per new best), this is paid every step, so an unfilled cell is
        # a persistent, ongoing loss for the whole team -- this pressures drones to
        # fill AND hold the last cells instead of freezing one step short.
        if self.coverage_step_reward != 0.0:
            for agent in self.possible_agents:
                rewards[agent] += self.coverage_step_reward * covered_count

        coverage_delta = max(0, covered_count - self.best_occupied_count)
        if coverage_delta > 0:
            for agent in self.possible_agents:
                rewards[agent] += self.coverage_delta_reward * coverage_delta
            self.best_occupied_count = covered_count

        # Penalize hovering outside the target set to avoid a safe-but-stuck policy.
        for agent in self.possible_agents:
            if int(actions[agent]) == 0 and self.agent_pos[agent] not in self._target_set:
                rewards[agent] -= self.hover_penalty

        shape_done = covered_count == len(self.target_cells)

        # 4) Per-drone shaping toward current assigned target.
        for agent in self.possible_agents:
            target_r, target_c = self.assigned_target_cell[agent]
            r, c = self.agent_pos[agent]
            new_dist = float(abs(r - target_r) + abs(c - target_c))
            if self.shaping_coef != 0.0:
                rewards[agent] += self.shaping_coef * (self.prev_per_drone_dists[agent] - new_dist)
            self.prev_per_drone_dists[agent] = new_dist

        # 5) Stage transition. Only final target terminates the episode.
        final_done = self._advance_stage_if_needed(shape_done, rewards)

        truncated = self.step_count >= self.max_steps
        terminations = {agent: final_done for agent in self.possible_agents}
        truncations = {agent: (truncated and not final_done) for agent in self.possible_agents}

        obs = self._all_obs()
        infos = {
            agent: {
                "collision": (agent in collisions),
                "shape_done": shape_done,
                "final_done": final_done,
                "stage_idx": self.stage_idx,
                "stages_completed": self.stage_done_count,
                "target_shape": self.target_shape_name,
                "shapes_path": self.formation_path.label,
                "occupied_count": self.last_occupied_count,
                "best_occupied_count": self.best_occupied_count,
                "coverage": self.last_occupied_count / max(1, len(self.target_cells)),
                "wind_prob": self._cur_wind_prob,
                "wind_dir": self._cur_wind_dir,
                "comm_fail_prob": self._cur_comm_fail_prob,
            }
            for agent in self.possible_agents
        }

        if final_done or truncated:
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    # Helpers ----------------------------------------------------------------
    def _set_target_formation(self, formation: Formation) -> None:
        if len(formation.cells) != self.n_agents:
            raise ValueError(f"target formation {formation.name!r} size must equal n_agents")
        self.target_shape_name = formation.name
        self.target_cells = list(formation.cells)
        self._target_set = set(self.target_cells)

    def _count_occupied_targets(self) -> int:
        return len({pos for pos in self.agent_pos.values() if pos in self._target_set})

    def _reset_assignment(self) -> None:
        self.assigned_target_cell = self._compute_assignment()
        self.prev_per_drone_dists = {
            agent: float(
                abs(self.agent_pos[agent][0] - self.assigned_target_cell[agent][0])
                + abs(self.agent_pos[agent][1] - self.assigned_target_cell[agent][1])
            )
            for agent in self.possible_agents
        }
        # Start coverage accounting from the current overlap with the new target.
        self.last_occupied_count = self._count_occupied_targets()
        self.best_occupied_count = self.last_occupied_count

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
        for agent in self.possible_agents:
            rewards[agent] += self.completion_reward

        is_final_stage = self.stage_idx >= len(self.formation_path.targets) - 1
        if is_final_stage:
            return True

        self.stage_idx += 1
        self._set_target_formation(self.formation_path.targets[self.stage_idx])
        self._reset_assignment()
        return False

    def _compute_assignment(self) -> dict[str, tuple[int, int]]:
        """Optimal 1:1 drone<->target assignment via Hungarian algorithm."""
        drones = [self.agent_pos[agent] for agent in self.possible_agents]
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
        # Draw this step's symmetric link mask once: one Bernoulli per unordered
        # (a, b) pair, used for both (a -> b) and (b -> a) packet delivery.
        agents = self.possible_agents
        self._cur_link_alive = {}
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                self._cur_link_alive[frozenset({a, b})] = (
                    self.np_random.random() >= self._cur_comm_fail_prob
                )
        return {agent: self._get_obs(agent) for agent in agents}

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
            if self._cur_link_alive.get(frozenset({agent, other}), True):
                ro, co = self.agent_pos[other]
                comm[3 * i] = ro / gs
                comm[3 * i + 1] = co / gs
                comm[3 * i + 2] = 1.0
            i += 1

        assigned_r, assigned_c = self.assigned_target_cell[agent]
        assigned = np.array([assigned_r / gs, assigned_c / gs], dtype=np.float32)

        return np.concatenate([own, target_vec, comm, assigned])

    def render(self) -> None:
        grid = [["."] * self.grid_size for _ in range(self.grid_size)]
        for r, c in self.target_cells:
            grid[r][c] = "x"
        for agent, (r, c) in self.agent_pos.items():
            grid[r][c] = agent.split("_")[-1][-1]
        print(
            f"shapes='{self.formation_path.label}'  "
            f"stage={self.stage_idx + 1}/{len(self.formation_path.targets)}  "
            f"target='{self.target_shape_name}'  "
            f"coverage={self.last_occupied_count}/{len(self.target_cells)}"
        )
        print("\n".join("".join(row) for row in grid))
        print()


class BatteryShapeFormationEnv(ShapeFormationEnv):
    """Same as ShapeFormationEnv plus a per-drone battery state.

    Goal of this variant: encourage the swarm to spread movement across drones
    rather than letting a single drone do all the work. Each drone has a
    battery level in [0, 1] that drains every step. Hovering (stay) drains by
    ``hover_battery_cost`` and moving drains by ``move_battery_cost`` (the
    latter is normally larger). Moving a drone with a low battery incurs an
    extra penalty proportional to ``low_battery_move_penalty * (1 - battery)``,
    so the policy is pushed to let depleted drones rest while fuller drones
    move.

    Observation appends:
        own_battery (1) + others_battery (n_agents - 1, masked by comm_fail_prob)

    All reward shaping from the parent class is preserved so the battery
    variant only differs from the baseline by the new state, observation,
    and the battery-weighted move penalty.
    """

    DEFAULT_INITIAL_BATTERY: float = 1.0
    DEFAULT_HOVER_BATTERY_COST: float = 0.002
    DEFAULT_MOVE_BATTERY_COST: float = 0.005
    DEFAULT_LOW_BATTERY_MOVE_PENALTY: float = 0.1

    metadata = {"name": "shape_transition_comm_battery_v0", "is_parallelizable": True}

    def __init__(
        self,
        *args,
        initial_battery: float = DEFAULT_INITIAL_BATTERY,
        hover_battery_cost: float = DEFAULT_HOVER_BATTERY_COST,
        move_battery_cost: float = DEFAULT_MOVE_BATTERY_COST,
        low_battery_move_penalty: float = DEFAULT_LOW_BATTERY_MOVE_PENALTY,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not 0.0 < initial_battery <= 1.0:
            raise ValueError("initial_battery must be in (0, 1]")
        self.initial_battery = float(initial_battery)
        self.hover_battery_cost = float(hover_battery_cost)
        self.move_battery_cost = float(move_battery_cost)
        self.low_battery_move_penalty = float(low_battery_move_penalty)

        # Extend observation: + own_battery (1) + others_battery (n_agents - 1)
        self.obs_dim = self.obs_dim + 1 + (self.n_agents - 1)
        self._obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self.battery: dict[str, float] = {
            agent: self.initial_battery for agent in self.possible_agents
        }

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, infos = super().reset(seed=seed, options=options)
        self.battery = {agent: self.initial_battery for agent in self.possible_agents}
        # Re-emit observations because parent's were built without battery slots.
        obs = self._all_obs()
        # NOTE: do not write battery into ``infos`` — torchrl's PettingZooWrapper
        # locks an info schema based on what's present at reset time, and then
        # fails on any extra keys appearing at step time. The parent step adds
        # several info keys (collision, shape_done, ...) that aren't in reset,
        # so the only safe schema is the parent's empty one. Battery is already
        # in the observation, so the policy can still see it.
        return obs, infos

    def step(self, actions: dict[str, int]):
        # Snapshot pre-step positions so we can detect actual movement after
        # collision resolution (a drone that bumped into another stays put,
        # so we drain hover cost, not move cost).
        prev_pos = dict(self.agent_pos)

        obs, rewards, terminations, truncations, infos = super().step(actions)

        for agent in self.possible_agents:
            act = int(actions[agent])
            moved = self.agent_pos[agent] != prev_pos[agent]
            # Drain: attempted move (action != 0) costs move_battery_cost even
            # if a collision blocked the move; pure stay costs hover_battery_cost.
            if act == 0 and not moved:
                drain = self.hover_battery_cost
            else:
                drain = self.move_battery_cost
            self.battery[agent] = max(0.0, self.battery[agent] - drain)

            # Battery-weighted move penalty: as battery -> 0, moving costs more.
            # No extra penalty for staying.
            if act != 0 and self.low_battery_move_penalty != 0.0:
                rewards[agent] -= self.low_battery_move_penalty * (1.0 - self.battery[agent])

        obs = self._all_obs()
        return obs, rewards, terminations, truncations, infos

    def _get_obs(self, agent: str) -> np.ndarray:
        gs = float(self.grid_size)
        r, c = self.agent_pos[agent]
        own = np.array([r / gs, c / gs], dtype=np.float32)

        target_vec = np.empty(2 * self.n_agents, dtype=np.float32)
        for i, (tr, tc) in enumerate(self.target_cells):
            target_vec[2 * i] = tr / gs
            target_vec[2 * i + 1] = tc / gs

        # Same per-step symmetric link mask drives BOTH position and battery,
        # so a dropped packet hides both fields together AND the drop is
        # mirrored between (a -> b) and (b -> a).
        comm = np.zeros(3 * (self.n_agents - 1), dtype=np.float32)
        comm_batt = np.zeros(self.n_agents - 1, dtype=np.float32)
        i = 0
        for other in self.possible_agents:
            if other == agent:
                continue
            if self._cur_link_alive.get(frozenset({agent, other}), True):
                ro, co = self.agent_pos[other]
                comm[3 * i] = ro / gs
                comm[3 * i + 1] = co / gs
                comm[3 * i + 2] = 1.0
                comm_batt[i] = self.battery[other]
            i += 1

        assigned_r, assigned_c = self.assigned_target_cell[agent]
        assigned = np.array([assigned_r / gs, assigned_c / gs], dtype=np.float32)

        own_batt = np.array([self.battery[agent]], dtype=np.float32)

        return np.concatenate([own, target_vec, comm, assigned, own_batt, comm_batt])
