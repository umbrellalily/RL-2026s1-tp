"""Evaluate a MAPPO policy trained on the formation-path communication environment.

Default setup is 20 drones on a 25x25 grid.

Examples:
    python comm_eval.py --ckpt checkpoints_comm20/ckpt_60.pt --shapes GROUND,X --greedy
    python comm_eval.py --ckpt checkpoints_comm20/ckpt_60.pt --save-gif demo.gif --shapes GROUND,+,X,I,-
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

from comm_env import ShapeFormationEnv
from formation_seq import available_shape_names

GROUP = "drone"


class _Tee:
    """Write to multiple file-like streams simultaneously (console + file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def build_actor(obs_dim, n_actions, n_agents, hidden, device):
    actor_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=n_actions,
        n_agents=n_agents,
        centralized=False,
        share_params=True,
        depth=2,
        num_cells=hidden,
        device=device,
    )
    actor_module = TensorDictModule(
        actor_net,
        in_keys=[(GROUP, "observation")],
        out_keys=[(GROUP, "logits")],
    )
    return ProbabilisticActor(
        module=actor_module,
        in_keys=[(GROUP, "logits")],
        out_keys=[(GROUP, "action")],
        distribution_class=Categorical,
        return_log_prob=False,
        log_prob_key=(GROUP, "sample_log_prob"),
    )


def make_env(seed, device, grid_size, n_agents, max_steps, shapes, comm_fail_prob):
    base = ShapeFormationEnv(
        grid_size=grid_size,
        n_agents=n_agents,
        max_steps=max_steps,
        shapes=shapes,
        comm_fail_prob=comm_fail_prob,
        shaping_coef=0.0,
    )
    env = PettingZooWrapper(
        env=base,
        group_map={GROUP: list(base.possible_agents)},
        categorical_actions=True,
        use_mask=False,
        device=device,
        seed=seed,
    )
    return env, base


def _snapshot_frame(base):
    return {
        "positions": dict(base.agent_pos),
        "target_cells": list(base.target_cells),
        "target_shape": base.target_shape_name,
        "stage_idx": base.stage_iddex,
        "num_stages": len(base.formation_path.targets),
        "stages_completed": base.stage_done_count,
        "occupied_count": getattr(base, "last_occupied_count", 0),
        "best_occupied_count": getattr(base, "best_occupied_count", 0),
        "target_count": len(base.target_cells),
        "coverage": getattr(base, "last_occupied_count", 0) / max(1, len(base.target_cells)),
        "shapes_path": base.formation_path.label,
    }


def rollout(env, base, actor, exploration, max_steps, record=False):
    td = env.reset()
    history = [_snapshot_frame(base)] if record else None
    total_reward = 0.0
    steps = 0
    collisions = 0
    success = False

    for _ in range(max_steps):
        with set_exploration_type(exploration), torch.no_grad():
            actor(td)
        td = env.step(td)
        steps += 1
        collisions += base.last_collision_count
        if record:
            history.append(_snapshot_frame(base))
        total_reward += float(td.get(("next", GROUP, "reward")).mean().item())

        if bool(td.get(("next", GROUP, "done")).all().item()):
            success = bool(td.get(("next", GROUP, "terminated")).any().item())
            break
        td = step_mdp(td)

    return {
        "success": success,
        "total_reward": total_reward,
        "steps": steps,
        "collisions": collisions,
        "stages_completed": base.stage_done_count,
        "num_stages": len(base.formation_path.targets),
        "shapes": base.formation_path.label,
        "final_target": base.target_shape_name,
        "occupied_count": getattr(base, "last_occupied_count", 0),
        "best_occupied_count": getattr(base, "best_occupied_count", 0),
        "target_count": len(base.target_cells),
        "coverage": getattr(base, "last_occupied_count", 0) / max(1, len(base.target_cells)),
        "best_coverage": getattr(base, "best_occupied_count", 0) / max(1, len(base.target_cells)),
        "history": history,
    }


def render_ascii(grid_size, target_cells, positions):
    grid = [["."] * grid_size for _ in range(grid_size)]
    for r, c in target_cells:
        grid[r][c] = "x"
    for agent, (r, c) in positions.items():
        grid[r][c] = agent.split("_")[-1][-1]
    return "\n".join("".join(row) for row in grid)


def save_gif(grid_size: int, history, path: Path, fps: int = 4) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab20")

    def draw(step: int) -> None:
        frame = history[step]
        ax.clear()
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, color="lightgray", linewidth=0.4)
        ax.set_aspect("equal")
        ax.set_title(
            f"{frame['shapes_path']} | stage "
            f"{frame['stage_idx'] + 1}/{frame['num_stages']} "
            f"target='{frame['target_shape']}' | "
            f"cover={frame['occupied_count']}/{frame['target_count']} | "
            f"step {step}/{len(history) - 1}"
        )
        for (r, c) in frame["target_cells"]:
            ax.add_patch(
                patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, facecolor="#ffe0e0", edgecolor="none"
                )
            )
        for i, (agent, (r, c)) in enumerate(frame["positions"].items()):
            ax.add_patch(
                patches.Circle(
                    (c, r), 0.35, facecolor=cmap(i % 20), edgecolor="black", linewidth=1.0
                )
            )
            ax.text(
                c, r, agent.split("_")[-1],
                color="white", ha="center", va="center",
                fontsize=7, fontweight="bold",
            )

    anim = FuncAnimation(fig, draw, frames=len(history), interval=1000 // fps)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--greedy", action="store_true", help="argmax action selection (default: sample)")
    parser.add_argument(
        "--shapes",
        type=str,
        default="GROUND,X",
        help=(
            "Comma-separated formation path, e.g. 'GROUND,X' or 'GROUND,+,X,I,-'. "
            f"Available names: {available_shape_names()}"
        ),
    )
    parser.add_argument(
        "--comm-fail-prob", type=float, default=0.0,
        help="evaluate under simulated communication loss",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.3)
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Path to save eval output text (default: eval_<ckpt_stem>.txt)",
    )
    args = parser.parse_args()

    shapes = [shape.strip() for shape in args.shapes.split(",") if shape.strip()]

    out_path = Path(args.out) if args.out else Path(f"eval_{Path(args.ckpt).stem}.txt")
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log = open(out_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, eval_log)

    try:
        device = torch.device(args.device)
        torch.manual_seed(args.seed)

        env, base = make_env(
            seed=args.seed,
            device=device,
            grid_size=args.grid_size,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
            shapes=shapes,
            comm_fail_prob=args.comm_fail_prob,
        )
        actor = build_actor(base.obs_dim, 5, base.n_agents, args.hidden, device)
        with torch.no_grad():
            actor(env.reset())
        state = torch.load(args.ckpt, map_location=device)
        actor.load_state_dict(state["actor"])
        actor.eval()

        exploration = ExplorationType.MODE if args.greedy else ExplorationType.RANDOM

        runs = []
        for _ in range(args.n_episodes):
            runs.append(rollout(env, base, actor, exploration, max_steps=base.max_steps))

        successes = sum(run["success"] for run in runs)
        print(
            f"=== Eval over {args.n_episodes} episodes "
            f"({'greedy' if args.greedy else 'stochastic'}, "
            f"comm_fail_prob={args.comm_fail_prob}) ==="
        )
        print(f"  Grid / agents          : {args.grid_size}x{args.grid_size}, n_agents={args.n_agents}")
        print(f"  Shapes path            : {base.formation_path.label}")
        print(f"  Overall success rate   : {successes / len(runs):.1%}")
        print(f"  Mean stages completed  : {np.mean([r['stages_completed'] for r in runs]):.2f} / {runs[0]['num_stages']}")
        print(f"  Mean final coverage    : {np.mean([r['coverage'] for r in runs]):.1%}")
        print(f"  Mean best coverage     : {np.mean([r['best_coverage'] for r in runs]):.1%}")
        print(f"  Overall mean reward    : {np.mean([r['total_reward'] for r in runs]):+.3f}")
        print(f"  Overall mean ep length : {np.mean([r['steps'] for r in runs]):.1f}")
        print(f"  Overall mean collisions: {np.mean([r['collisions'] for r in runs]):.2f}")

        if args.render or args.save_gif:
            demo_env, demo_base = make_env(
                seed=args.seed + 1,
                device=device,
                grid_size=args.grid_size,
                n_agents=args.n_agents,
                max_steps=args.max_steps,
                shapes=shapes,
                comm_fail_prob=args.comm_fail_prob,
            )
            demo = rollout(
                demo_env, demo_base, actor, exploration,
                max_steps=demo_base.max_steps, record=True,
            )
            print(
                f"\n=== Demo episode: shapes='{demo['shapes']}', "
                f"success={demo['success']}, stages={demo['stages_completed']}/{demo['num_stages']}, "
                f"coverage={demo['occupied_count']}/{demo['target_count']} "
                f"best={demo['best_occupied_count']}/{demo['target_count']}, "
                f"reward={demo['total_reward']:+.2f}, steps={demo['steps']} ==="
            )
            if args.render:
                for step, frame in enumerate(demo["history"]):
                    print(
                        f"\nstep {step}/{len(demo['history']) - 1} | "
                        f"stage {frame['stage_idx'] + 1}/{frame['num_stages']} | "
                        f"target='{frame['target_shape']}'"
                    )
                    print(render_ascii(args.grid_size, frame["target_cells"], frame["positions"]))
                    time.sleep(args.render_delay)
            if args.save_gif:
                gif_path = Path(args.save_gif)
                if gif_path.parent != Path(""):
                    gif_path.parent.mkdir(parents=True, exist_ok=True)
                save_gif(args.grid_size, demo["history"], gif_path)
                print(f"\nSaved GIF -> {gif_path}")
    finally:
        sys.stdout = orig_stdout
        eval_log.close()
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
