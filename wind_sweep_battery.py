"""Wind-disturbance robustness sweep for two MAPPO **battery-aware** policies.

Mirrors wind_sweep.py but loads BatteryShapeFormationEnv (via comm_eval_battery)
so it can compare a baseline battery policy vs a wind-robust battery policy
under different wind levels.

Example:
    python wind_sweep_battery.py \\
        --ckpt-a ckpt_dg_batt_base/ckpt_70.pt \\
        --ckpt-b ckpt_dg_batt_robust/ckpt_70.pt \\
        --shapes "GROUND,D,G" --wind-levels 0.0,0.1,0.2,0.3,0.4 \\
        --out wind_sweep_batt.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchrl.envs.utils import ExplorationType

from comm_eval_battery import build_actor, make_env, rollout

N_ACTIONS = 5


def evaluate_policy(ckpt: str, shapes, wind_prob: float, n_episodes: int, args, device):
    env, base = make_env(
        seed=args.seed,
        device=device,
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        shapes=shapes,
        comm_fail_prob=0.0,
        completion_reward=args.completion_reward,
        wind_prob=wind_prob,
        wind_strength=args.wind_strength,
        randomize_wind=False,
        initial_battery=args.initial_battery,
        hover_battery_cost=args.hover_battery_cost,
        move_battery_cost=args.move_battery_cost,
        low_battery_move_penalty=args.low_battery_move_penalty,
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


def plot_curves(wind_levels, results_a, results_b, args, shapes) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title, ylabel in (
        (ax1, "success", "Success rate vs wind (battery)", "success rate"),
        (ax2, "coverage", "Mean coverage vs wind (battery)", "mean coverage"),
    ):
        ax.plot(wind_levels, [r[key] for r in results_a],
                "o--", color="#d9534f", label=args.label_a)
        ax.plot(wind_levels, [r[key] for r in results_b],
                "o-", color="#5cb85c", label=args.label_b)
        ax.set_title(title)
        ax.set_xlabel("wind probability")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"Wind robustness sweep (battery)  -  shapes='{','.join(shapes)}'")
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True, help="baseline checkpoint (no-wind battery policy)")
    parser.add_argument("--ckpt-b", required=True, help="robust checkpoint (wind-DR battery policy)")
    parser.add_argument("--shapes", type=str, default="GROUND,X")
    parser.add_argument("--wind-levels", type=str, default="0.0,0.1,0.2,0.3,0.4")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--wind-strength", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--label-a", type=str, default="baseline (no wind)")
    parser.add_argument("--label-b", type=str, default="robust (wind DR)")
    parser.add_argument("--out", type=str, default="wind_sweep_batt.png")
    parser.add_argument("--completion-reward", type=float, default=30.0)
    parser.add_argument("--initial-battery", type=float, default=1.0)
    parser.add_argument("--hover-battery-cost", type=float, default=0.002)
    parser.add_argument("--move-battery-cost", type=float, default=0.005)
    parser.add_argument("--low-battery-move-penalty", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    wind_levels = [float(w) for w in args.wind_levels.split(",") if w.strip()]

    print(
        f"Wind sweep (battery) | shapes='{','.join(shapes)}' | levels={wind_levels} | "
        f"{args.n_episodes} eps/point | {'greedy' if args.greedy else 'stochastic'}\n"
    )
    header = (
        f"{'wind':>5} | {'A succ':>7} {'A cov':>6} {'A coll':>7} "
        f"| {'B succ':>7} {'B cov':>6} {'B coll':>7}"
    )
    print(header)
    print("-" * len(header))

    results_a, results_b = [], []
    for w in wind_levels:
        a = evaluate_policy(args.ckpt_a, shapes, w, args.n_episodes, args, device)
        b = evaluate_policy(args.ckpt_b, shapes, w, args.n_episodes, args, device)
        results_a.append(a)
        results_b.append(b)
        print(
            f"{w:>5.2f} | {a['success']:>6.1%} {a['coverage']:>6.1%} {a['collisions']:>7.2f} "
            f"| {b['success']:>6.1%} {b['coverage']:>6.1%} {b['collisions']:>7.2f}"
        )

    out_path = plot_curves(wind_levels, results_a, results_b, args, shapes)
    print(f"\nSaved curve -> {out_path}")

    gap = results_b[-1]["success"] - results_a[-1]["success"]
    print(
        f"At wind={wind_levels[-1]:.2f}: baseline success={results_a[-1]['success']:.1%}, "
        f"robust success={results_b[-1]['success']:.1%}  (gap {gap:+.1%})"
    )


if __name__ == "__main__":
    main()
