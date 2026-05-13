"""Evaluate a MAPPO policy trained on the comm-based env (multi-shape, night mode).

Differences from eval.py:
  - imports ShapeFormationEnv from env_comm
  - reports per-shape success rate (since training is multi-shape)
  - --shape flag lets you pick a fixed target shape for the demo episode
  - --comm-fail-prob lets you simulate communication loss at eval time

Examples:
    python eval_comm.py --ckpt checkpoints_comm/ckpt_60.pt
    python eval_comm.py --ckpt checkpoints_comm/ckpt_60.pt --render
    python eval_comm.py --ckpt checkpoints_comm/ckpt_60.pt --save-gif demo_comm.gif --shape I
    python eval_comm.py --ckpt checkpoints_comm/ckpt_60.pt --comm-fail-prob 0.2
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

from comm_env import SHAPES, ShapeFormationEnv

GROUP = "drone"


class _Tee:
    """Write to multiple file-like streams simultaneously (console + file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


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


def make_env(seed, device, target_shapes, comm_fail_prob):
    base = ShapeFormationEnv(
        target_shapes=target_shapes,
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


def rollout(env, base, actor, exploration, max_steps, record=False):
    td = env.reset()
    history = [dict(base.agent_pos)] if record else None
    total_reward = 0.0
    steps = 0
    collisions = 0
    success = False
    shape_name = base.target_shape_name

    for _ in range(max_steps):
        with set_exploration_type(exploration), torch.no_grad():
            actor(td)
        td = env.step(td)
        steps += 1
        collisions += base.last_collision_count
        if record:
            history.append(dict(base.agent_pos))
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
        "shape": shape_name,
        "history": history,
    }


def render_ascii(base, positions):
    grid = [["."] * base.grid_size for _ in range(base.grid_size)]
    for r, c in base.target_cells:
        grid[r][c] = "x"
    for a, (r, c) in positions.items():
        grid[r][c] = a[-1]
    return "\n".join("".join(row) for row in grid)


def save_gif(base, history, path: Path, fps: int = 4) -> None:
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    g = base.grid_size
    fig, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab10")
    shape_name = base.target_shape_name

    def draw(step: int) -> None:
        ax.clear()
        ax.set_xlim(-0.5, g - 0.5)
        ax.set_ylim(-0.5, g - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(g))
        ax.set_yticks(range(g))
        ax.grid(True, color="lightgray", linewidth=0.5)
        ax.set_aspect("equal")
        ax.set_title(f"target='{shape_name}'   step {step}/{len(history) - 1}")
        for (r, c) in base.target_cells:
            ax.add_patch(
                patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1, facecolor="#ffe0e0", edgecolor="none"
                )
            )
        for i, (_agent, (r, c)) in enumerate(history[step].items()):
            ax.add_patch(
                patches.Circle(
                    (c, r), 0.35, facecolor=cmap(i), edgecolor="black", linewidth=1.0
                )
            )
            ax.text(
                c, r, str(i),
                color="white", ha="center", va="center",
                fontsize=10, fontweight="bold",
            )

    anim = FuncAnimation(fig, draw, frames=len(history), interval=1000 // fps)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--greedy", action="store_true", help="argmax (default: sample)"
    )
    parser.add_argument(
        "--shape",
        type=str,
        default=None,
        help=f"force target shape for the demo episode "
        f"(one of {list(SHAPES)}); default: random",
    )
    parser.add_argument(
        "--comm-fail-prob", type=float, default=0.0,
        help="evaluate under simulated communication loss"
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

    # Set up stdout tee so all eval output is mirrored to a text file.
    out_path = Path(args.out) if args.out else Path(f"eval_{Path(args.ckpt).stem}.txt")
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _eval_log = open(out_path, "w", encoding="utf-8")
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _eval_log)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    # ---- env for quantitative eval: all shapes ----
    env, base = make_env(
        seed=args.seed,
        device=device,
        target_shapes=list(SHAPES),
        comm_fail_prob=args.comm_fail_prob,
    )
    actor = build_actor(base.obs_dim, 5, base.n_agents, args.hidden, device)
    with torch.no_grad():
        actor(env.reset())
    state = torch.load(args.ckpt, map_location=device)
    actor.load_state_dict(state["actor"])
    actor.eval()

    exploration = ExplorationType.MODE if args.greedy else ExplorationType.RANDOM

    # ---- Quantitative evaluation, broken down per shape ----
    by_shape: dict[str, list[dict]] = defaultdict(list)
    for _ in range(args.n_episodes):
        r = rollout(env, base, actor, exploration, max_steps=base.max_steps)
        by_shape[r["shape"]].append(r)

    print(
        f"=== Eval over {args.n_episodes} episodes "
        f"({'greedy' if args.greedy else 'stochastic'}, "
        f"comm_fail_prob={args.comm_fail_prob}) ==="
    )
    all_runs = [r for runs in by_shape.values() for r in runs]
    successes_total = sum(r["success"] for r in all_runs)
    print(f"  Overall success rate     : {successes_total / len(all_runs):.1%}")
    print(f"  Overall mean reward      : "
          f"{np.mean([r['total_reward'] for r in all_runs]):+.3f}")
    print(f"  Overall mean ep length   : "
          f"{np.mean([r['steps'] for r in all_runs]):.1f}")
    print(f"  Overall mean collisions  : "
          f"{np.mean([r['collisions'] for r in all_runs]):.2f}")
    print()
    print("  Per-shape breakdown:")
    print(f"  {'shape':<7}{'N':>5}{'success':>10}{'reward':>10}{'len':>7}{'coll':>7}")
    for shape in sorted(by_shape):
        runs = by_shape[shape]
        n = len(runs)
        sr = sum(r["success"] for r in runs) / n
        mr = np.mean([r["total_reward"] for r in runs])
        ml = np.mean([r["steps"] for r in runs])
        mc = np.mean([r["collisions"] for r in runs])
        print(f"  {shape:<7}{n:>5}{sr:>9.1%}{mr:>+10.2f}{ml:>7.1f}{mc:>7.2f}")

    # ---- Visualization (one episode) ----
    if args.render or args.save_gif:
        # Optionally pin demo to a chosen shape
        if args.shape is not None:
            if args.shape not in SHAPES:
                raise SystemExit(f"--shape must be one of {list(SHAPES)}")
            demo_env, demo_base = make_env(
                seed=args.seed + 1,
                device=device,
                target_shapes=[args.shape],
                comm_fail_prob=args.comm_fail_prob,
            )
        else:
            demo_env, demo_base = env, base
        demo = rollout(
            demo_env, demo_base, actor, exploration,
            max_steps=demo_base.max_steps, record=True,
        )
        print(
            f"\n=== Demo episode: target='{demo['shape']}', "
            f"success={demo['success']}, reward={demo['total_reward']:+.2f}, "
            f"steps={demo['steps']} ==="
        )
        if args.render:
            for step, positions in enumerate(demo["history"]):
                print(f"\nstep {step}/{len(demo['history']) - 1}  "
                      f"target='{demo['shape']}'")
                print(render_ascii(demo_base, positions))
                time.sleep(args.render_delay)
        if args.save_gif:
            out = Path(args.save_gif)
            if out.parent != Path(""):
                out.parent.mkdir(parents=True, exist_ok=True)
            save_gif(demo_base, demo["history"], out)
            print(f"\nSaved GIF -> {out}")

    sys.stdout = _orig_stdout
    _eval_log.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
