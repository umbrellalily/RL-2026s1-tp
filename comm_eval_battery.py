"""Evaluate a battery-aware MAPPO policy and render a GIF that shows each
drone's battery level at every frame.

This mirrors comm_eval.py but uses BatteryShapeFormationEnv so observations
include battery state and the GIF visualisation overlays a per-drone battery
bar at each step.

Examples:
    python comm_eval_battery.py --ckpt checkpoints_comm20_battery/ckpt_60.pt \\
        --shapes GROUND,X --greedy --save-gif demo_battery.gif
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

from comm_env import BatteryShapeFormationEnv
from formation_seq import available_shape_names

GROUP = "drone"


class _Tee:
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


def make_env(
    seed, device, grid_size, n_agents, max_steps, shapes, comm_fail_prob,
    completion_reward, wind_prob, wind_strength, randomize_wind,
    initial_battery, hover_battery_cost, move_battery_cost, low_battery_move_penalty,
    random_shape_pool=None, random_path_length="3",
    randomize_comm_fail=False,
):
    base = BatteryShapeFormationEnv(
        grid_size=grid_size,
        n_agents=n_agents,
        max_steps=max_steps,
        shapes=shapes,
        comm_fail_prob=comm_fail_prob,
        completion_reward=completion_reward,
        shaping_coef=0.0,
        wind_prob=wind_prob,
        wind_strength=wind_strength,
        randomize_wind=randomize_wind,
        randomize_comm_fail=randomize_comm_fail,
        initial_battery=initial_battery,
        hover_battery_cost=hover_battery_cost,
        move_battery_cost=move_battery_cost,
        low_battery_move_penalty=low_battery_move_penalty,
        random_shape_pool=random_shape_pool,
        random_path_length=random_path_length,
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


def _target_snapshot(base):
    """Capture the formation the drones are *currently* forming.

    Must be read BEFORE ``env.step()``. A step that completes a letter advances
    the stage in-place (``comm_env`` sets ``target_cells`` to the next letter
    within the same step), so reading targets after the step makes a just-
    completed formation render with its LEDs OFF -- the drones sit on the old
    letter while ``target_cells`` already points at the next one.
    """
    return {
        "target_cells": list(base.target_cells),
        "target_shape": base.target_shape_name,
        "stage_idx": base.stage_idx,
        "num_stages": len(base.formation_path.targets),
        "target_count": len(base.target_cells),
        "shapes_path": base.formation_path.label,
    }


def _snapshot_frame(base, target=None):
    # ``target`` is the pre-step formation (see _target_snapshot). When omitted
    # (e.g. the initial reset frame) the env's current target is correct.
    if target is None:
        target = _target_snapshot(base)
    occupied = getattr(base, "last_occupied_count", 0)
    return {
        "positions": dict(base.agent_pos),
        "batteries": dict(base.battery),
        "stages_completed": base.stage_done_count,
        "occupied_count": occupied,
        "best_occupied_count": getattr(base, "best_occupied_count", 0),
        "coverage": occupied / max(1, target["target_count"]),
        **target,
    }


def rollout(env, base, actor, exploration, max_steps, record=False):
    td = env.reset()
    history = [_snapshot_frame(base)] if record else None
    total_reward = 0.0
    steps = 0
    collisions = 0
    success = False

    for _ in range(max_steps):
        # Snapshot the formation being formed BEFORE stepping; env.step() may
        # complete the letter and advance the stage in-place, which would
        # otherwise leave the completed letter's drones rendered as LED-off.
        active_target = _target_snapshot(base) if record else None
        with set_exploration_type(exploration), torch.no_grad():
            actor(td)
        td = env.step(td)
        steps += 1
        collisions += base.last_collision_count
        if record:
            history.append(_snapshot_frame(base, target=active_target))
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
        "final_batteries": dict(base.battery),
        "history": history,
    }


def _battery_color(level: float) -> str:
    """Green (full) -> yellow (mid) -> red (low)."""
    level = max(0.0, min(1.0, float(level)))
    if level > 0.5:
        # green -> yellow as it falls from 1.0 to 0.5
        t = (1.0 - level) / 0.5  # 0..1
        r = int(255 * t)
        g = 220
        b = 60
    else:
        # yellow -> red as it falls from 0.5 to 0.0
        t = (0.5 - level) / 0.5
        r = 255
        g = int(220 * (1.0 - t))
        b = 60
    return f"#{r:02x}{g:02x}{b:02x}"


def save_gif(grid_size: int, history, path: Path, fps: int = 4) -> None:
    """Render a battery-aware GIF.

    On top of the LED on/off visualisation, each drone gets a small horizontal
    battery bar above its dot. The bar fill width is proportional to the
    battery level and colour-coded green -> yellow -> red. A numeric percent
    is printed next to each drone for the exact value.
    """
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    LED_CORE = "#ffd24a"
    LED_RIM = "#fff4b3"
    LED_GLOW = "#ffc94a"
    LED_OFF_FACE = "#3a3a44"
    LED_OFF_EDGE = "#555"
    BAR_BG = "#222230"
    BAR_EDGE = "#777"

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="#0a0a14")
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
        batteries = frame.get("batteries", {})
        if batteries:
            batt_vals = list(batteries.values())
            mean_b = sum(batt_vals) / len(batt_vals)
            min_b = min(batt_vals)
            max_b = max(batt_vals)
            batt_title = (
                f" | batt mean={mean_b * 100:5.1f}% "
                f"min={min_b * 100:5.1f}% max={max_b * 100:5.1f}%"
            )
        else:
            batt_title = ""
        ax.set_title(
            f"{frame['shapes_path']} | stage "
            f"{frame['stage_idx'] + 1}/{frame['num_stages']} "
            f"target='{frame['target_shape']}' | "
            f"LED on={n_on}/{frame['target_count']} | "
            f"step {step}/{len(history) - 1}"
            f"{batt_title}",
            color="#dddddd",
            fontsize=9,
        )
        for (r, c) in frame["target_cells"]:
            ax.add_patch(patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor="#1c1c2e", edgecolor="#3a3a55",
                linewidth=0.6, linestyle=(0, (2, 2)),
            ))
        # Bar geometry (in data units, i.e. grid cells).
        bar_w = 0.9
        bar_h = 0.16
        bar_y_offset = 0.55  # bar sits this far above the drone centre
        for agent, (r, c) in frame["positions"].items():
            led_on = (r, c) in target_set
            if led_on:
                for glow_r, glow_alpha in [(0.85, 0.10), (0.65, 0.20), (0.48, 0.35)]:
                    ax.add_patch(patches.Circle(
                        (c, r), glow_r,
                        facecolor=LED_GLOW, edgecolor="none", alpha=glow_alpha,
                    ))
                ax.add_patch(patches.Circle(
                    (c, r), 0.34,
                    facecolor=LED_CORE, edgecolor=LED_RIM, linewidth=1.6,
                ))
            else:
                ax.add_patch(patches.Circle(
                    (c, r), 0.22,
                    facecolor=LED_OFF_FACE, edgecolor=LED_OFF_EDGE,
                    linewidth=0.5, alpha=0.85,
                ))

            # Battery bar above the drone.
            batt = float(batteries.get(agent, 0.0)) if batteries else 0.0
            bar_x = c - bar_w / 2.0
            bar_y = r - bar_y_offset - bar_h
            # Background
            ax.add_patch(patches.Rectangle(
                (bar_x, bar_y), bar_w, bar_h,
                facecolor=BAR_BG, edgecolor=BAR_EDGE, linewidth=0.4,
            ))
            # Fill
            fill_w = bar_w * max(0.0, min(1.0, batt))
            if fill_w > 0:
                ax.add_patch(patches.Rectangle(
                    (bar_x, bar_y), fill_w, bar_h,
                    facecolor=_battery_color(batt), edgecolor="none",
                ))
            # Percentage label to the right of the bar.
            ax.text(
                c + bar_w / 2.0 + 0.05, bar_y + bar_h / 2.0,
                f"{batt * 100:.0f}%",
                color="#dddddd", fontsize=5.5, va="center", ha="left",
            )

    anim = FuncAnimation(fig, draw, frames=len(history), interval=1000 // fps)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument(
        "--shapes", type=str, default="GROUND,X",
        help=(
            "Comma-separated formation path, e.g. 'GROUND,X' or 'GROUND,+,X,I,-'. "
            f"Available names: {available_shape_names()}"
        ),
    )
    parser.add_argument("--comm-fail-prob", type=float, default=0.0)
    parser.add_argument("--wind-prob", type=float, default=0.0)
    parser.add_argument("--wind-strength", type=int, default=1)
    parser.add_argument("--randomize-wind", action="store_true")
    parser.add_argument(
        "--randomize-comm-fail", action="store_true",
        help="Sample comm drop rate from [0, comm_fail_prob] each episode.",
    )
    parser.add_argument("--completion-reward", type=float, default=30.0)

    parser.add_argument(
        "--initial-battery", type=float,
        default=BatteryShapeFormationEnv.DEFAULT_INITIAL_BATTERY,
    )
    parser.add_argument(
        "--hover-battery-cost", type=float,
        default=BatteryShapeFormationEnv.DEFAULT_HOVER_BATTERY_COST,
    )
    parser.add_argument(
        "--move-battery-cost", type=float,
        default=BatteryShapeFormationEnv.DEFAULT_MOVE_BATTERY_COST,
    )
    parser.add_argument(
        "--low-battery-move-penalty", type=float,
        default=BatteryShapeFormationEnv.DEFAULT_LOW_BATTERY_MOVE_PENALTY,
    )

    # Random-sequence eval: each episode samples a fresh random target sequence
    # from --random-shape-pool. Use to measure generalization across many unseen
    # sequences instead of one fixed sequence.
    parser.add_argument(
        "--random-shape-pool",
        type=str,
        default="",
        help=(
            "Comma-separated shape pool for per-episode random sequence sampling "
            "during eval (e.g. 'A,B,C,D'). 'ALL_LETTERS' = A-Z. Empty disables "
            "(uses --shapes deterministically)."
        ),
    )
    parser.add_argument(
        "--random-path-length",
        type=str,
        default="3",
        help="Targets per random episode. 'N' fixed, 'lo-hi' range.",
    )

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay", type=float, default=0.3)
    parser.add_argument("--save-gif", type=str, default=None)
    parser.add_argument(
        "--demo-tries",
        type=int,
        default=1,
        help=(
            "Episodes to try when picking the demo to render/GIF. 1 = the original "
            "single greedy demo. >1 = search for a SUCCESSFUL episode (greedy first, "
            "then stochastic with varied seeds); falls back to the best-coverage one."
        ),
    )
    parser.add_argument(
        "--out", type=str, default="",
        help="Path to save eval output text (default: eval_battery_<ckpt_stem>.txt)",
    )
    args = parser.parse_args()

    shapes = [shape.strip() for shape in args.shapes.split(",") if shape.strip()]

    if args.random_shape_pool:
        if args.random_shape_pool.strip().upper() == "ALL_LETTERS":
            random_shape_pool: list[str] | None = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        else:
            random_shape_pool = [
                s.strip() for s in args.random_shape_pool.split(",") if s.strip()
            ]
    else:
        random_shape_pool = None

    out_path = Path(args.out) if args.out else Path(f"eval_battery_{Path(args.ckpt).stem}.txt")
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    eval_log = open(out_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    sys.stdout = _Tee(orig_stdout, eval_log)

    try:
        device = torch.device(args.device)
        torch.manual_seed(args.seed)

        env_kwargs = dict(
            device=device,
            grid_size=args.grid_size,
            n_agents=args.n_agents,
            max_steps=args.max_steps,
            shapes=shapes,
            comm_fail_prob=args.comm_fail_prob,
            completion_reward=args.completion_reward,
            wind_prob=args.wind_prob,
            wind_strength=args.wind_strength,
            randomize_wind=args.randomize_wind,
            randomize_comm_fail=args.randomize_comm_fail,
            initial_battery=args.initial_battery,
            hover_battery_cost=args.hover_battery_cost,
            move_battery_cost=args.move_battery_cost,
            low_battery_move_penalty=args.low_battery_move_penalty,
            random_shape_pool=random_shape_pool,
            random_path_length=args.random_path_length,
        )

        env, base = make_env(seed=args.seed, **env_kwargs)
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
        # Battery summary across episodes: how evenly batteries were drained.
        mean_batt = np.mean(
            [np.mean(list(r["final_batteries"].values())) for r in runs]
        )
        std_batt = np.mean(
            [np.std(list(r["final_batteries"].values())) for r in runs]
        )
        min_batt = np.mean(
            [np.min(list(r["final_batteries"].values())) for r in runs]
        )
        print(
            f"=== Battery-aware eval over {args.n_episodes} episodes "
            f"({'greedy' if args.greedy else 'stochastic'}, "
            f"comm_fail_prob={args.comm_fail_prob}, wind_prob={args.wind_prob}) ==="
        )
        print(f"  Grid / agents          : {args.grid_size}x{args.grid_size}, n_agents={args.n_agents}")
        print(f"  Shapes path            : {base.formation_path.label}")
        print(
            f"  Battery cfg            : init={args.initial_battery}, "
            f"hover={args.hover_battery_cost}, move={args.move_battery_cost}, "
            f"low_move_penalty={args.low_battery_move_penalty}"
        )
        print(f"  Overall success rate   : {successes / len(runs):.1%}")
        print(f"  Mean stages completed  : {np.mean([r['stages_completed'] for r in runs]):.2f}")
        print(f"  Mean final coverage    : {np.mean([r['coverage'] for r in runs]):.1%}")
        print(f"  Mean best coverage     : {np.mean([r['best_coverage'] for r in runs]):.1%}")
        print(f"  Overall mean reward    : {np.mean([r['total_reward'] for r in runs]):+.3f}")
        print(f"  Overall mean ep length : {np.mean([r['steps'] for r in runs]):.1f}")
        print(f"  Overall mean collisions: {np.mean([r['collisions'] for r in runs]):.2f}")
        print(f"  Final battery mean     : {mean_batt * 100:.1f}%")
        print(f"  Final battery min      : {min_batt * 100:.1f}%")
        print(f"  Final battery std-dev  : {std_batt * 100:.1f}% (lower = more even drain)")

        # ---- Random-pool generalization breakdown ------------------------
        # If random pool was active, episodes saw different sequences. Break
        # down success by sequence length and by individual letter.
        if random_shape_pool is not None:
            print()
            print(f"  === Random-sequence generalization breakdown ===")
            print(f"  Random pool             : {random_shape_pool}")
            print(f"  Random path length      : {args.random_path_length}")
            print(f"  Unique sequences seen   : {len(set(r['shapes'] for r in runs))} / {len(runs)} episodes")

            # Per-length success
            from collections import defaultdict
            by_len: dict[int, list[bool]] = defaultdict(list)
            for r in runs:
                # number of targets = stages in path
                n_tgt = r["num_stages"]
                by_len[n_tgt].append(r["success"])
            print(f"  Per-length success:")
            for L in sorted(by_len.keys()):
                s = by_len[L]
                print(f"    length {L} ({len(s):>3d} eps): success {sum(s) / len(s):.1%}")

            # Per-letter success: did episodes containing each letter succeed?
            letter_success: dict[str, list[bool]] = defaultdict(list)
            for r in runs:
                # extract letters from shapes label, e.g. "GROUND->B->A->Y->C"
                tokens = r["shapes"].split("->")[1:]  # drop GROUND
                for tok in tokens:
                    letter_success[tok].append(r["success"])
            print(f"  Per-letter success rate (episodes containing the letter):")
            for letter in sorted(letter_success.keys()):
                s = letter_success[letter]
                print(f"    {letter} ({len(s):>3d} eps): {sum(s) / len(s):.1%}")

            # Per-sequence detail for top failure cases
            seq_results: dict[str, list[dict]] = defaultdict(list)
            for r in runs:
                seq_results[r["shapes"]].append(r)
            failed_seqs = [
                (label, rs) for label, rs in seq_results.items()
                if any(not r["success"] for r in rs)
            ]
            if failed_seqs:
                print(f"  Sample failed sequences (first 5):")
                for label, rs in failed_seqs[:5]:
                    n_fail = sum(1 for r in rs if not r["success"])
                    avg_cov = np.mean([r["coverage"] for r in rs])
                    print(f"    {label}  failed {n_fail}/{len(rs)} times, mean coverage {avg_cov:.1%}")

        if args.render or args.save_gif:
            # Pick the episode to record. demo-tries=1 keeps the original single
            # greedy demo; >1 searches for a SUCCESSFUL episode (greedy first, then
            # stochastic with varied seeds), else falls back to best coverage.
            demo = best = None
            for t in range(max(1, args.demo_tries)):
                expl_t = exploration if t == 0 else ExplorationType.RANDOM
                demo_env, demo_base = make_env(seed=args.seed + 1 + t, **env_kwargs)
                d = rollout(demo_env, demo_base, actor, expl_t,
                            max_steps=demo_base.max_steps, record=True)
                if best is None or d["coverage"] > best["coverage"]:
                    best = d
                if d["success"]:
                    demo = d
                    print(f"[demo] success episode found (try {t + 1}, {'greedy' if t == 0 else 'stochastic'})")
                    break
            if demo is None:
                demo = best
                if args.demo_tries > 1:
                    print(f"[demo] no success in {args.demo_tries} tries -> saving best-coverage ({demo['coverage']:.0%}) episode")
            print(
                f"\n=== Demo episode: shapes='{demo['shapes']}', "
                f"success={demo['success']}, stages={demo['stages_completed']}/{demo['num_stages']}, "
                f"coverage={demo['occupied_count']}/{demo['target_count']} "
                f"best={demo['best_occupied_count']}/{demo['target_count']}, "
                f"reward={demo['total_reward']:+.2f}, steps={demo['steps']} ==="
            )
            if args.render:
                for step, frame in enumerate(demo["history"]):
                    batts = frame.get("batteries", {})
                    batt_line = "  ".join(
                        f"{a.split('_')[-1]}:{batts[a] * 100:5.1f}%"
                        for a in sorted(batts)
                    )
                    print(
                        f"\nstep {step}/{len(demo['history']) - 1} | "
                        f"stage {frame['stage_idx'] + 1}/{frame['num_stages']} | "
                        f"target='{frame['target_shape']}'\n"
                        f"  batt: {batt_line}"
                    )
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
