"""MAPPO training for the formation-path communication environment.

Default setup is 20 drones on a 25x25 grid.

Examples:
    python comm_train.py --shapes GROUND,X --total-frames 300000
    python comm_train.py --shapes GROUND,+,X,I --total-frames 1048576
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

from comm_env import ShapeFormationEnv


GROUP = "drone"


def make_env(
    seed: int,
    device: torch.device,
    grid_size: int,
    n_agents: int,
    max_steps: int,
    comm_fail_prob: float,
    shaping_coef: float,
    assigned_target_reward: float,
    coverage_delta_reward: float,
    hover_penalty: float,
    shapes: list[str] | None = None,
) -> TransformedEnv:
    base = ShapeFormationEnv(
        grid_size=grid_size,
        n_agents=n_agents,
        max_steps=max_steps,
        comm_fail_prob=comm_fail_prob,
        shaping_coef=shaping_coef,
        assigned_target_reward=assigned_target_reward,
        coverage_delta_reward=coverage_delta_reward,
        hover_penalty=hover_penalty,
        shapes=shapes,
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


def build_models(
    obs_dim: int,
    n_actions: int,
    n_agents: int,
    hidden: int,
    device: torch.device,
) -> tuple[ProbabilisticActor, TensorDictModule]:
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
    parser.add_argument("--n-agents", type=int, default=20)
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
    parser.add_argument("--save-dir", type=str, default="checkpoints_comm20")
    parser.add_argument("--ckpt-every", type=int, default=5)
    parser.add_argument(
        "--comm-fail-prob",
        type=float,
        default=0.0,
        help="Per-link Bernoulli probability that an incoming position packet is lost.",
    )
    parser.add_argument(
        "--shaping-coef",
        type=float,
        default=0.3,
        help="Potential-based shaping coefficient toward assigned target. 0 disables.",
    )
    parser.add_argument(
        "--assigned-target-reward",
        type=float,
        default=0.3,
        help="Extra per-step reward when a drone is exactly on its assigned target cell.",
    )
    parser.add_argument(
        "--coverage-delta-reward",
        type=float,
        default=0.05,
        help="Team reward multiplier when the stage reaches a new best target coverage count.",
    )
    parser.add_argument(
        "--hover-penalty",
        type=float,
        default=0.02,
        help="Penalty for choosing hover/stay outside the current target set.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        default="GROUND,X",
        help=(
            "Comma-separated formation path, e.g. 'GROUND,X' or "
            "'GROUND,+,X,I,-'. If the first name is not GROUND, "
            "GROUND is automatically prepended by the environment."
        ),
    )
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default="",
        help="Path to a previous checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default="runs_comm20",
        help="TensorBoard log directory (empty string disables logging).",
    )
    parser.add_argument(
        "--completion-reward",
        type=float,
        default=10.0,
        help="Team reward given to all drones when the current formation stage is completed.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    shapes = [shape.strip() for shape in args.shapes.split(",") if shape.strip()]

    env = make_env(
        seed=args.seed,
        device=device,
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        comm_fail_prob=args.comm_fail_prob,
        shaping_coef=args.shaping_coef,
        assigned_target_reward=args.assigned_target_reward,
        coverage_delta_reward=args.coverage_delta_reward,
        completion_reward=completion_reward,
        hover_penalty=args.hover_penalty,
        shapes=shapes,
    )

    probe = ShapeFormationEnv(
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
    )
    obs_dim = probe.obs_dim
    n_actions = 5
    n_agents = probe.n_agents

    print(
        f"Training config: grid_size={args.grid_size}, n_agents={n_agents}, "
        f"max_steps={args.max_steps}, obs_dim={obs_dim}, shapes='{probe.formation_path.label}', "
        f"assigned_target_reward={args.assigned_target_reward}, "
        f"coverage_delta_reward={args.coverage_delta_reward}, hover_penalty={args.hover_penalty}"
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
        run_name = f"shape_comm20_{int(time.time())}"
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
