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


def make_env(seed, device, grid_size, n_agents, max_steps, shapes, comm_fail_prob, completion_reward):
    base = ShapeFormationEnv(
        grid_size=grid_size,
        n_agents=n_agents,
        max_steps=max_steps,
        shapes=shapes,
        comm_fail_prob=comm_fail_prob,
        completion_reward=completion_reward,
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
        "stage_idx": base.stage_idx,
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
    """Render an episode as a GIF with LED on/off drone visualization.

    LED ON/OFF semantics
    --------------------
    Each drone has a single LED. The LED is ON iff the drone currently sits
    on a target cell of the active stage; otherwise it is OFF and the drone
    is interpreted as hovering off-target.

      LED ON  -> warm amber (uniform across all drones) + glow halo
      LED OFF -> small dim gray dot (hovering / in transit)

    All drones share the same LED color when ON so the forming letter/shape
    reads as a single coherent light, like a real drone show.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Warm-amber palette for ON LEDs (mimics drone-show LED at night)
    LED_CORE = "#ffd24a"      # bright warm yellow core
    LED_RIM = "#fff4b3"       # pale yellow rim
    LED_GLOW = "#ffc94a"      # halo color
    LED_OFF_FACE = "#3a3a44"  # dim gray off-state
    LED_OFF_EDGE = "#555"

    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0a0a14")
    ax.set_facecolor("#0a0a14")

    def draw(step: int) -> None:
        frame = history[step]
        ax.clear()
        ax.set_facecolor("#0a0a14")
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(0, grid_size, 2))
        ax.set_yticks(range(0, grid_size, 2))
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.grid(True, color="#222", linewidth=0.4)
        ax.set_aspect("equal")
        target_set = set(frame["target_cells"])
        n_on = sum(1 for pos in frame["positions"].values() if pos in target_set)
        ax.set_title(
            f"{frame['shapes_path']} | stage "
            f"{frame['stage_idx'] + 1}/{frame['num_stages']} "
            f"target='{frame['target_shape']}' | "
            f"LED on={n_on}/{frame['target_count']} | "
            f"step {step}/{len(history) - 1}",
            color="#dddddd",
            fontsize=10,
        )
        # Target cells: faint dotted outline so the desired formation is
        # visible but doesn't outshine lit drones.
        for (r, c) in frame["target_cells"]:
            ax.add_patch(patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor="#1c1c2e", edgecolor="#3a3a55",
                linewidth=0.6, linestyle=(0, (2, 2)),
            ))
        # Drones with LED on/off rendering -- ALL share the same LED color.
        for agent, (r, c) in frame["positions"].items():
            led_on = (r, c) in target_set
            if led_on:
                # Outer warm-amber glow (3 expanding translucent disks)
                for glow_r, glow_alpha in [(0.85, 0.10), (0.65, 0.20), (0.48, 0.35)]:
                    ax.add_patch(patches.Circle(
                        (c, r), glow_r,
                        facecolor=LED_GLOW, edgecolor="none", alpha=glow_alpha,
                    ))
                # Bright core with pale rim
                ax.add_patch(patches.Circle(
                    (c, r), 0.34,
                    facecolor=LED_CORE, edgecolor=LED_RIM, linewidth=1.6,
                ))
            else:
                # Dim, neutral drone (LED off / hovering / in transit)
                ax.add_patch(patches.Circle(
                    (c, r), 0.22,
                    facecolor=LED_OFF_FACE, edgecolor=LED_OFF_EDGE,
                    linewidth=0.5, alpha=0.85,
                ))

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
    parser.add_argument(
        "--completion-reward",
        type=float,
        default=30.0,
        help="Reward scale used only for reported evaluation reward; success/GIF are unaffected.",
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
