"""Communication-failure robustness sweep for two MAPPO formation policies.

Mirrors wind_sweep.py but the swept axis is the per-link packet drop rate
(--comm-fail-levels). Wind is held at 0 during the sweep so the experiment
isolates communication-loss robustness. Drop rate is FIXED per level
(randomize_comm_fail=False); link symmetry and pos/battery sharing follow
whatever is currently implemented in comm_env.py.

Examples:
    python comm_fail_sweep.py --ckpt-a ckpt_dg_base/ckpt_60.pt \\
        --ckpt-b ckpt_dg_comm_robust/ckpt_60.pt --shapes "GROUND,D,G" \\
        --max-steps 220

    python comm_fail_sweep.py --ckpt-a base.pt --ckpt-b robust.pt \\
        --comm-fail-levels 0.0,0.05,0.1,0.15,0.2,0.3 --n-episodes 200 --greedy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType

from comm_eval import build_actor, make_env, rollout

N_ACTIONS = 5


def evaluate_policy(ckpt: str, shapes, comm_fail_prob: float, n_episodes: int, args, device):
    """Run n_episodes for one checkpoint at one fixed comm-fail level."""
    env, base = make_env(
        seed=args.seed,
        device=device,
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        shapes=shapes,
        comm_fail_prob=comm_fail_prob,
        wind_prob=0.0,
        wind_strength=1,
        randomize_wind=False,
        randomize_comm_fail=False,
    )
    actor = build_actor(base.obs_dim, N_ACTIONS, base.n_agents, args.hidden, device)
    with torch.no_grad():
        actor(env.reset())
    state = torch.load(ckpt, map_location=device)
    actor.load_state_dict(state["actor"])
    actor.eval()

    exploration = ExplorationType.MODE if args.greedy else ExplorationType.RANDOM
    runs = [
        rollout(env, base, actor, exploration, max_steps=base.max_steps)
        for _ in range(n_episodes)
    ]
    return {
        "success": float(np.mean([r["success"] for r in runs])),
        "coverage": float(np.mean([r["coverage"] for r in runs])),
        "collisions": float(np.mean([r["collisions"] for r in runs])),
        "stages": float(np.mean([r["stages_completed"] for r in runs])),
        "num_stages": runs[0]["num_stages"],
    }


def plot_curves(levels, results_a, results_b, args, shapes) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title, ylabel in (
        (ax1, "success", "Success rate vs comm fail", "success rate"),
        (ax2, "coverage", "Mean coverage vs comm fail", "mean coverage"),
    ):
        ax.plot(levels, [r[key] for r in results_a],
                "o--", color="#d9534f", label=args.label_a)
        ax.plot(levels, [r[key] for r in results_b],
                "o-", color="#5cb85c", label=args.label_b)
        ax.set_title(title)
        ax.set_xlabel("comm fail probability")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Communication-failure robustness sweep  -  shapes='{','.join(shapes)}'")
    fig.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True, help="baseline checkpoint (no comm-fail policy)")
    parser.add_argument("--ckpt-b", required=True, help="robust checkpoint (comm-fail DR policy)")
    parser.add_argument("--shapes", type=str, default="GROUND,X")
    parser.add_argument(
        "--comm-fail-levels", type=str, default="0.0,0.05,0.1,0.15,0.2,0.3",
        help="comma-separated comm-fail probabilities to sweep",
    )
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--greedy", action="store_true", help="argmax actions (default: sample)")
    parser.add_argument("--label-a", type=str, default="baseline (no comm fail)")
    parser.add_argument("--label-b", type=str, default="robust (comm-fail DR)")
    parser.add_argument("--out", type=str, default="comm_fail_sweep.png")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    levels = [float(p) for p in args.comm_fail_levels.split(",") if p.strip()]

    print(
        f"Comm-fail sweep | shapes='{','.join(shapes)}' | levels={levels} | "
        f"{args.n_episodes} episodes/point | {'greedy' if args.greedy else 'stochastic'}\n"
    )
    header = (
        f"{'comm':>5} | {'A succ':>7} {'A cov':>6} {'A coll':>7} "
        f"| {'B succ':>7} {'B cov':>6} {'B coll':>7}"
    )
    print(header)
    print("-" * len(header))

    results_a, results_b = [], []
    for p in levels:
        a = evaluate_policy(args.ckpt_a, shapes, p, args.n_episodes, args, device)
        b = evaluate_policy(args.ckpt_b, shapes, p, args.n_episodes, args, device)
        results_a.append(a)
        results_b.append(b)
        print(
            f"{p:>5.2f} | {a['success']:>6.1%} {a['coverage']:>6.1%} {a['collisions']:>7.2f} "
            f"| {b['success']:>6.1%} {b['coverage']:>6.1%} {b['collisions']:>7.2f}"
        )

    out_path = plot_curves(levels, results_a, results_b, args, shapes)
    print(f"\nSaved curve -> {out_path}")

    gap = results_b[-1]["success"] - results_a[-1]["success"]
    print(
        f"At comm_fail={levels[-1]:.2f}: baseline success={results_a[-1]['success']:.1%}, "
        f"robust success={results_b[-1]['success']:.1%}  (gap {gap:+.1%})"
    )


if __name__ == "__main__":
    main()
