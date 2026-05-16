"""Evaluate a MAPPO policy on the v5 letter env (variable letter sizes).

GIF distinguishes letter cells (pink) from wait cells (light gray) so you can
visually see how many drones formed the actual letter vs went to the border.

Examples:
    python eval_letter.py --ckpt checkpoints_letter/ckpt_final.pt
    python eval_letter.py --ckpt checkpoints_letter/ckpt_final.pt --letter A --save-gif demo.gif
    python eval_letter.py --ckpt checkpoints_letter/ckpt_final.pt --letter all --save-gif demo.gif
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

from letter_env import LetterFormationEnv, place_letter
from letters import LETTERS

GROUP = "drone"


class _Tee:
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
        n_agent_inputs=obs_dim, n_agent_outputs=n_actions,
        n_agents=n_agents, centralized=False, share_params=True,
        depth=2, num_cells=hidden, device=device,
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


def parse_origin(s):
    s = s.strip()
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        raise SystemExit("--letter-origin must be 'row,col' or empty")
    return (int(parts[0]), int(parts[1]))


def make_env(seed, device, target_letters, letter_origin, comm_fail_prob):
    base = LetterFormationEnv(
        target_letters=target_letters,
        letter_origin=letter_origin,
        comm_fail_prob=comm_fail_prob,
        shaping_coef=0.0,
    )
    env = PettingZooWrapper(
        env=base,
        group_map={GROUP: list(base.possible_agents)},
        categorical_actions=True, use_mask=False,
        device=device, seed=seed,
    )
    return env, base


def rollout(env, base, actor, exploration, max_steps, record=False):
    td = env.reset()
    history = [dict(base.agent_pos)] if record else None
    total_reward = 0.0
    steps = 0
    collisions = 0
    success = False
    letter_name = base.target_letter_name
    origin = base.target_origin
    target_cells = list(base.target_cells)
    letter_cells_abs = set(place_letter(letter_name, origin))

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
        "letter": letter_name,
        "origin": origin,
        "target_cells": target_cells,
        "letter_cells": list(letter_cells_abs),
        "grid_size": base.grid_size,
        "history": history,
        "n_letter_cells": base.n_letter_cells,
    }


def save_gif(result, path: Path, fps: int = 4, trail_len: int = 12) -> None:
    """Letter cells = pink, wait cells = light gray, drones = colored circles."""
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    g = result["grid_size"]
    history = result["history"]
    target_cells = result["target_cells"]
    letter_cells = set(result["letter_cells"])
    letter_name = result["letter"]
    origin = result["origin"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    cmap = plt.get_cmap("tab20")

    def draw(step):
        ax.clear()
        ax.set_xlim(-0.5, g - 0.5)
        ax.set_ylim(-0.5, g - 0.5)
        ax.invert_yaxis()
        ax.set_xticks(range(0, g, 2))
        ax.set_yticks(range(0, g, 2))
        ax.grid(True, color="lightgray", linewidth=0.4)
        ax.set_aspect("equal")
        done = step >= len(history) - 1
        color = "green" if (done and result["success"]) else (
            "red" if done else "black"
        )
        ax.set_title(
            f"letter='{letter_name}'  origin={origin}  "
            f"glyph={result['n_letter_cells']} cells  "
            f"step {step}/{len(history) - 1}",
            color=color,
        )
        # target cells: differentiate letter (pink) vs wait (light gray)
        for (r, c) in target_cells:
            if (r, c) in letter_cells:
                ax.add_patch(patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#ffd0d0", edgecolor="none",
                ))
            else:
                ax.add_patch(patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#e8e8e8", edgecolor="#bbbbbb",
                    linewidth=0.4, linestyle="dashed",
                ))
        # drones with fading trails
        agents = list(history[0].keys())
        start = max(0, step - trail_len + 1)
        for i, agent in enumerate(agents):
            col_i = cmap(i % 20)
            xs, ys = [], []
            for k in range(start, step + 1):
                r, c = history[k][agent]
                xs.append(c)
                ys.append(r)
            if len(xs) > 1:
                ax.plot(xs, ys, color=col_i, linewidth=1.3, alpha=0.5)
            r, c = history[step][agent]
            ax.add_patch(patches.Circle(
                (c, r), 0.34, facecolor=col_i, edgecolor="black", linewidth=0.8,
            ))

    anim = FuncAnimation(fig, draw, frames=len(history), interval=1000 // fps)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n-episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--letter", type=str, default=None,
                   help=f"force letter (one of {list(LETTERS)} or 'all')")
    p.add_argument("--letter-origin", type=str, default="")
    p.add_argument("--comm-fail-prob", type=float, default=0.0)
    p.add_argument("--save-gif", type=str, default=None)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    out_path = Path(args.out) if args.out else Path(f"eval_{Path(args.ckpt).stem}.txt")
    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    _log = open(out_path, "w", encoding="utf-8")
    _orig = sys.stdout
    sys.stdout = _Tee(_orig, _log)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    demo_origin = parse_origin(args.letter_origin)

    env, base = make_env(
        args.seed, device, list(LETTERS), None, args.comm_fail_prob,
    )
    actor = build_actor(base.obs_dim, 5, base.n_agents, args.hidden, device)
    with torch.no_grad():
        actor(env.reset())
    state = torch.load(args.ckpt, map_location=device)
    actor.load_state_dict(state["actor"])
    actor.eval()

    exploration = ExplorationType.MODE if args.greedy else ExplorationType.RANDOM

    by_letter: dict[str, list[dict]] = defaultdict(list)
    for _ in range(args.n_episodes):
        r = rollout(env, base, actor, exploration, max_steps=base.max_steps)
        by_letter[r["letter"]].append(r)

    print(f"=== Eval over {args.n_episodes} episodes "
          f"({'greedy' if args.greedy else 'stochastic'}, "
          f"comm_fail_prob={args.comm_fail_prob}, placement=RANDOM) ===")
    all_runs = [r for runs in by_letter.values() for r in runs]
    successes = sum(r["success"] for r in all_runs)
    print(f"  Overall success rate     : {successes / len(all_runs):.1%}")
    print(f"  Overall mean reward      : "
          f"{np.mean([r['total_reward'] for r in all_runs]):+.3f}")
    print(f"  Overall mean ep length   : "
          f"{np.mean([r['steps'] for r in all_runs]):.1f}")
    print(f"  Overall mean collisions  : "
          f"{np.mean([r['collisions'] for r in all_runs]):.2f}")
    print()
    print("  Per-letter breakdown:")
    print(f"  {'letter':<8}{'N':>5}{'glyph':>7}{'success':>10}{'reward':>10}{'len':>7}{'coll':>7}")
    for letter in sorted(by_letter):
        runs = by_letter[letter]
        n = len(runs)
        glyph_n = runs[0]["n_letter_cells"]
        sr = sum(r["success"] for r in runs) / n
        mr = np.mean([r["total_reward"] for r in runs])
        ml = np.mean([r["steps"] for r in runs])
        mc = np.mean([r["collisions"] for r in runs])
        print(f"  {letter:<8}{n:>5}{glyph_n:>7}{sr:>9.1%}{mr:>+10.2f}{ml:>7.1f}{mc:>7.2f}")

    if args.save_gif:
        if args.letter == "all":
            demo_letters = list(LETTERS)
        elif args.letter is not None:
            if args.letter not in LETTERS:
                raise SystemExit(f"--letter must be one of {list(LETTERS)} or 'all'")
            demo_letters = [args.letter]
        else:
            demo_letters = [None]

        for idx, lt in enumerate(demo_letters):
            d_env, d_base = make_env(
                args.seed + idx + 1, device,
                [lt] if lt else list(LETTERS),
                demo_origin, args.comm_fail_prob,
            )
            demo = rollout(
                d_env, d_base, actor, exploration,
                max_steps=d_base.max_steps, record=True,
            )
            print(f"\n=== Demo: letter='{demo['letter']}' "
                  f"(glyph={demo['n_letter_cells']} cells), origin={demo['origin']}, "
                  f"success={demo['success']}, reward={demo['total_reward']:+.2f}, "
                  f"steps={demo['steps']} ===")
            out = Path(args.save_gif)
            if args.letter == "all":
                out = out.parent / f"{out.stem}_{demo['letter']}{out.suffix}"
            if out.parent != Path(""):
                out.parent.mkdir(parents=True, exist_ok=True)
            save_gif(demo, out)
            print(f"Saved GIF -> {out}")

    sys.stdout = _orig
    _log.close()
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
