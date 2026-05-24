"""MAPPO training for the BATTERY-aware formation environment.

Mirrors comm_train.py but uses BatteryShapeFormationEnv so the policy sees
per-drone battery levels and is penalised for moving low-battery drones. The
baseline (no battery) training in comm_train.py is left untouched so the two
setups can be compared directly.

Examples:
    python comm_train_battery.py --shapes GROUND,X --total-frames 300000
    python comm_train_battery.py --shapes GROUND,+,X,I \\
        --hover-battery-cost 0.003 --move-battery-cost 0.008 \\
        --low-battery-move-penalty 0.2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore

from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from comm_env import BatteryShapeFormationEnv


GROUP = "drone"


def make_env(
    seed: int,
    device: torch.device,
    grid_size: int,
    n_agents: int,
    max_steps: int,
    comm_fail_prob: float,
    shaping_coef: float,
    completion_reward: float,
    assigned_target_reward: float,
    coverage_delta_reward: float,
    hover_penalty: float,
    shapes: list[str] | None,
    wind_prob: float,
    wind_strength: int,
    randomize_wind: bool,
    initial_battery: float,
    hover_battery_cost: float,
    move_battery_cost: float,
    low_battery_move_penalty: float,
    random_shape_pool: list[str] | None = None,
    random_path_length: str = "3",
) -> TransformedEnv:
    base = BatteryShapeFormationEnv(
        grid_size=grid_size,
        n_agents=n_agents,
        max_steps=max_steps,
        comm_fail_prob=comm_fail_prob,
        shaping_coef=shaping_coef,
        completion_reward=completion_reward,
        assigned_target_reward=assigned_target_reward,
        coverage_delta_reward=coverage_delta_reward,
        hover_penalty=hover_penalty,
        shapes=shapes,
        wind_prob=wind_prob,
        wind_strength=wind_strength,
        randomize_wind=randomize_wind,
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
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[(GROUP, "reward")], out_keys=[(GROUP, "episode_reward")]),
    )
    return env


def build_models(obs_dim, n_actions, n_agents, hidden, device):
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
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys=[(GROUP, "logits")],
        out_keys=[(GROUP, "action")],
        distribution_class=Categorical,
        return_log_prob=True,
        log_prob_key=(GROUP, "sample_log_prob"),
    )

    critic_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=1,
        n_agents=n_agents,
        centralized=True,
        share_params=True,
        depth=2,
        num_cells=hidden,
        device=device,
    )
    critic = TensorDictModule(
        critic_net,
        in_keys=[(GROUP, "observation")],
        out_keys=[(GROUP, "state_value")],
    )
    return actor, critic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=25)
    parser.add_argument("--n-agents", type=int, default=14)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--total-frames", type=int, default=300_000)
    parser.add_argument("--frames-per-batch", type=int, default=4096)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lmbda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints_comm20_battery")
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--comm-fail-prob", type=float, default=0.0)
    parser.add_argument("--shaping-coef", type=float, default=0.3)
    parser.add_argument("--completion-reward", type=float, default=30.0)
    parser.add_argument("--assigned-target-reward", type=float, default=0.3)
    parser.add_argument("--coverage-delta-reward", type=float, default=0.2)
    parser.add_argument("--hover-penalty", type=float, default=0.02)
    parser.add_argument(
        "--shapes",
        type=str,
        default="GROUND,X",
        help="Comma-separated formation path, e.g. 'GROUND,X' or 'GROUND,+,X,I,-'.",
    )
    parser.add_argument("--wind-prob", type=float, default=0.0)
    parser.add_argument("--wind-strength", type=int, default=1)
    parser.add_argument("--randomize-wind", action="store_true")

    # Battery-specific knobs (defaults match BatteryShapeFormationEnv defaults).
    parser.add_argument(
        "--initial-battery",
        type=float,
        default=BatteryShapeFormationEnv.DEFAULT_INITIAL_BATTERY,
        help="Per-drone battery at episode start, in (0, 1]. Default 1.0 (full).",
    )
    parser.add_argument(
        "--hover-battery-cost",
        type=float,
        default=BatteryShapeFormationEnv.DEFAULT_HOVER_BATTERY_COST,
        help="Battery drained per hover (stay) step.",
    )
    parser.add_argument(
        "--move-battery-cost",
        type=float,
        default=BatteryShapeFormationEnv.DEFAULT_MOVE_BATTERY_COST,
        help="Battery drained per attempted-move step (larger than hover).",
    )
    parser.add_argument(
        "--low-battery-move-penalty",
        type=float,
        default=BatteryShapeFormationEnv.DEFAULT_LOW_BATTERY_MOVE_PENALTY,
        help=(
            "Reward penalty applied when a drone moves, scaled by (1 - battery). "
            "Set 0 to disable the battery-weighted penalty (drain still happens)."
        ),
    )

    # Per-episode random shape sequence (for generalization across arbitrary
    # sequences at test time). When --random-shape-pool is set, --shapes is
    # ignored for path construction; only the start (GROUND) is fixed and the
    # rest of the path is sampled per reset.
    parser.add_argument(
        "--random-shape-pool",
        type=str,
        default="",
        help=(
            "Comma-separated pool of shape names to randomly sample target sequences "
            "from each episode (e.g. 'A,B,C,D,E'). 'ALL_LETTERS' expands to A-Z. "
            "Empty disables (uses --shapes deterministically)."
        ),
    )
    parser.add_argument(
        "--random-path-length",
        type=str,
        default="3",
        help="Number of target shapes per episode. 'N' for fixed, 'lo-hi' for range.",
    )

    parser.add_argument("--load-ckpt", type=str, default="")
    parser.add_argument("--tb-logdir", type=str, default="runs_comm20_battery")
    args = parser.parse_args()

    # Resolve random shape pool.
    if args.random_shape_pool:
        if args.random_shape_pool.strip().upper() == "ALL_LETTERS":
            random_shape_pool: list[str] | None = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        else:
            random_shape_pool = [
                s.strip() for s in args.random_shape_pool.split(",") if s.strip()
            ]
    else:
        random_shape_pool = None

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    shapes = [shape.strip() for shape in args.shapes.split(",") if shape.strip()]

    env_kwargs = dict(
        seed=args.seed,
        device=device,
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        comm_fail_prob=args.comm_fail_prob,
        shaping_coef=args.shaping_coef,
        completion_reward=args.completion_reward,
        assigned_target_reward=args.assigned_target_reward,
        coverage_delta_reward=args.coverage_delta_reward,
        hover_penalty=args.hover_penalty,
        shapes=shapes,
        wind_prob=args.wind_prob,
        wind_strength=args.wind_strength,
        randomize_wind=args.randomize_wind,
        initial_battery=args.initial_battery,
        hover_battery_cost=args.hover_battery_cost,
        move_battery_cost=args.move_battery_cost,
        low_battery_move_penalty=args.low_battery_move_penalty,
        random_shape_pool=random_shape_pool,
        random_path_length=args.random_path_length,
    )
    env = make_env(**env_kwargs)

    probe = BatteryShapeFormationEnv(
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        comm_fail_prob=args.comm_fail_prob,
        shaping_coef=args.shaping_coef,
        completion_reward=args.completion_reward,
        assigned_target_reward=args.assigned_target_reward,
        coverage_delta_reward=args.coverage_delta_reward,
        hover_penalty=args.hover_penalty,
        shapes=shapes,
        wind_prob=args.wind_prob,
        wind_strength=args.wind_strength,
        randomize_wind=args.randomize_wind,
        initial_battery=args.initial_battery,
        hover_battery_cost=args.hover_battery_cost,
        move_battery_cost=args.move_battery_cost,
        low_battery_move_penalty=args.low_battery_move_penalty,
        random_shape_pool=random_shape_pool,
        random_path_length=args.random_path_length,
    )
    obs_dim = probe.obs_dim
    n_actions = 5
    n_agents = probe.n_agents

    print(
        f"Battery training: grid_size={args.grid_size}, n_agents={n_agents}, "
        f"max_steps={args.max_steps}, obs_dim={obs_dim}, "
        f"shapes='{probe.formation_path.label}'"
    )
    print(
        f"Battery: initial={args.initial_battery}, hover_cost={args.hover_battery_cost}, "
        f"move_cost={args.move_battery_cost}, "
        f"low_battery_move_penalty={args.low_battery_move_penalty}"
    )
    if args.wind_prob > 0.0:
        print(
            f"Wind: prob={args.wind_prob}, strength={args.wind_strength}, "
            f"randomize={args.randomize_wind}"
        )

    actor, critic = build_models(obs_dim, n_actions, n_agents, args.hidden, device)

    with torch.no_grad():
        td = env.reset()
        actor(td)
        critic(td)

    if args.load_ckpt:
        state = torch.load(args.load_ckpt, map_location=device)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        print(f"Warm-started from {args.load_ckpt}")

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=args.clip_eps,
        entropy_bonus=True,
        entropy_coeff=args.ent_coef,
        critic_coeff=args.vf_coef,
        normalize_advantage=True,
    )
    loss_module.set_keys(
        reward=(GROUP, "reward"),
        action=(GROUP, "action"),
        sample_log_prob=(GROUP, "sample_log_prob"),
        value=(GROUP, "state_value"),
        done=(GROUP, "done"),
        terminated=(GROUP, "terminated"),
        advantage=(GROUP, "advantage"),
        value_target=(GROUP, "value_target"),
    )
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda)
    gae = loss_module.value_estimator

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        device=device,
        storing_device=device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch_size,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.tb_logdir and SummaryWriter is not None:
        run_name = f"shape_comm20_battery_{int(time.time())}"
        writer = SummaryWriter(log_dir=str(Path(args.tb_logdir) / run_name))
        print(f"TensorBoard logging -> {writer.log_dir}")
    elif args.tb_logdir and SummaryWriter is None:
        print("tensorboard not installed; skipping TB logging (pip install tensorboard)")

    for it, data in enumerate(collector):
        with torch.no_grad():
            gae(data)

        data_flat = data.reshape(-1)
        replay_buffer.empty()
        replay_buffer.extend(data_flat)

        n_minibatches = max(1, args.frames_per_batch // args.minibatch_size)
        running_loss = 0.0
        n_updates = 0
        for _ in range(args.ppo_epochs):
            for _ in range(n_minibatches):
                batch = replay_buffer.sample()
                loss_vals = loss_module(batch)
                loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), args.max_grad_norm)
                optim.step()
                running_loss += float(loss.item())
                n_updates += 1

        collector.update_policy_weights_()

        ep_rew = data.get(("next", GROUP, "episode_reward"))
        done = data.get(("next", GROUP, "done"))
        terminated = data.get(("next", GROUP, "terminated"))
        finished_rewards = ep_rew[done]
        finished_terminated = terminated[done]
        n_finished_entries = finished_rewards.numel()
        mean_ep = finished_rewards.mean().item() if n_finished_entries > 0 else float("nan")
        success_rate = finished_terminated.float().mean().item() if n_finished_entries > 0 else float("nan")
        avg_loss = running_loss / max(1, n_updates)
        frames_seen = (it + 1) * args.frames_per_batch
        print(
            f"iter={it:4d}  frames={frames_seen:>8d}  "
            f"mean_ep_reward={mean_ep:+7.3f}  success={success_rate:5.1%}  "
            f"loss={avg_loss:7.4f}"
        )

        if writer is not None:
            if n_finished_entries > 0:
                writer.add_scalar("train/mean_episode_reward", mean_ep, frames_seen)
                writer.add_scalar("train/success_rate", success_rate, frames_seen)
            writer.add_scalar("train/loss_total", avg_loss, frames_seen)

        if (it + 1) % args.ckpt_every == 0:
            torch.save(
                {"actor": actor.state_dict(), "critic": critic.state_dict()},
                save_dir / f"ckpt_{it + 1}.pt",
            )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
