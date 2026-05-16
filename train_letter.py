"""MAPPO training for the letter-formation environment (박근희 v5).

Variable letter sizes (each letter uses its natural cell count). Fleet size
= MAX_CELLS drones; remaining drones go to wait cells along the grid border.

Examples:
    # all letters, random placement
    python train_letter.py --total-frames 500000

    # subset of letters
    python train_letter.py --letters A,B,C --total-frames 300000

    # one letter, fixed origin
    python train_letter.py --letters T --letter-origin 5,8 --total-frames 100000
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

from letter_env import LetterFormationEnv


GROUP = "drone"


def make_env(seed, device, comm_fail_prob, shaping_coef, target_letters, letter_origin):
    base = LetterFormationEnv(
        comm_fail_prob=comm_fail_prob,
        shaping_coef=shaping_coef,
        target_letters=target_letters,
        letter_origin=letter_origin,
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
        n_agent_inputs=obs_dim, n_agent_outputs=n_actions,
        n_agents=n_agents, centralized=False, share_params=True,
        depth=2, num_cells=hidden, device=device,
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
        n_agent_inputs=obs_dim, n_agent_outputs=1,
        n_agents=n_agents, centralized=True, share_params=True,
        depth=2, num_cells=hidden, device=device,
    )
    critic = TensorDictModule(
        critic_net,
        in_keys=[(GROUP, "observation")],
        out_keys=[(GROUP, "state_value")],
    )
    return actor, critic


def parse_origin(s):
    s = s.strip()
    if not s:
        return None
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--letter-origin must be 'row,col' (e.g. '5,8') or empty"
        )
    return (int(parts[0]), int(parts[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--total-frames", type=int, default=500_000)
    p.add_argument("--frames-per-batch", type=int, default=4096)
    p.add_argument("--minibatch-size", type=int, default=512)
    p.add_argument("--ppo-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lmbda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-dir", type=str, default="checkpoints_letter")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--comm-fail-prob", type=float, default=0.0)
    p.add_argument("--shaping-coef", type=float, default=0.3)
    p.add_argument("--letters", type=str, default="")
    p.add_argument("--letter-origin", type=str, default="")
    p.add_argument("--load-ckpt", type=str, default="")
    p.add_argument("--tb-logdir", type=str, default="runs_letter")
    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    target_letters = (
        [s.strip() for s in args.letters.split(",") if s.strip()]
        if args.letters else None
    )
    letter_origin = parse_origin(args.letter_origin)

    env = make_env(
        args.seed, device, args.comm_fail_prob, args.shaping_coef,
        target_letters, letter_origin,
    )
    probe = LetterFormationEnv(
        comm_fail_prob=args.comm_fail_prob, shaping_coef=args.shaping_coef,
        target_letters=target_letters, letter_origin=letter_origin,
    )
    obs_dim = probe.obs_dim
    n_actions = 5
    n_agents = probe.n_agents
    print(f"obs_dim={obs_dim}  n_agents={n_agents}  grid={probe.grid_size}  "
          f"letters={target_letters or 'ALL 26'}  "
          f"origin={letter_origin or 'RANDOM'}")

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

    loss = ClipPPOLoss(
        actor_network=actor, critic_network=critic,
        clip_epsilon=args.clip_eps, entropy_bonus=True,
        entropy_coeff=args.ent_coef, critic_coeff=args.vf_coef,
        normalize_advantage=True,
    )
    loss.set_keys(
        reward=(GROUP, "reward"), action=(GROUP, "action"),
        sample_log_prob=(GROUP, "sample_log_prob"),
        value=(GROUP, "state_value"),
        done=(GROUP, "done"), terminated=(GROUP, "terminated"),
        advantage=(GROUP, "advantage"), value_target=(GROUP, "value_target"),
    )
    loss.make_value_estimator(
        ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda
    )
    gae = loss.value_estimator

    collector = SyncDataCollector(
        env, actor,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        device=device, storing_device=device,
    )
    replay = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch_size,
    )
    optim = torch.optim.Adam(loss.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.tb_logdir and SummaryWriter is not None:
        run_name = f"letter_{int(time.time())}"
        writer = SummaryWriter(log_dir=str(Path(args.tb_logdir) / run_name))
        print(f"TensorBoard logging -> {writer.log_dir}")

    for it, data in enumerate(collector):
        with torch.no_grad():
            gae(data)

        flat = data.reshape(-1)
        replay.empty()
        replay.extend(flat)

        n_minibatches = max(1, args.frames_per_batch // args.minibatch_size)
        running_loss = 0.0
        n_updates = 0
        for _ in range(args.ppo_epochs):
            for _ in range(n_minibatches):
                batch = replay.sample()
                vals = loss(batch)
                total = vals["loss_objective"] + vals["loss_critic"] + vals["loss_entropy"]
                optim.zero_grad()
                total.backward()
                torch.nn.utils.clip_grad_norm_(loss.parameters(), args.max_grad_norm)
                optim.step()
                running_loss += float(total.item())
                n_updates += 1

        collector.update_policy_weights_()

        ep_rew = data.get(("next", GROUP, "episode_reward"))
        done = data.get(("next", GROUP, "done"))
        terminated = data.get(("next", GROUP, "terminated"))
        finished_rewards = ep_rew[done]
        finished_terminated = terminated[done]
        n_finished = finished_rewards.numel()
        mean_ep = finished_rewards.mean().item() if n_finished else float("nan")
        success_rate = (
            finished_terminated.float().mean().item() if n_finished else float("nan")
        )
        avg_loss = running_loss / max(1, n_updates)
        frames_seen = (it + 1) * args.frames_per_batch
        print(
            f"iter={it:4d}  frames={frames_seen:>8d}  "
            f"mean_ep_reward={mean_ep:+7.3f}  success={success_rate:5.1%}  "
            f"loss={avg_loss:7.4f}"
        )

        if writer is not None:
            if n_finished > 0:
                writer.add_scalar("train/mean_episode_reward", mean_ep, frames_seen)
                writer.add_scalar("train/success_rate", success_rate, frames_seen)
            writer.add_scalar("train/loss_total", avg_loss, frames_seen)

        if (it + 1) % args.ckpt_every == 0:
            torch.save(
                {"actor": actor.state_dict(), "critic": critic.state_dict()},
                save_dir / f"ckpt_{it + 1}.pt",
            )

    torch.save(
        {"actor": actor.state_dict(), "critic": critic.state_dict()},
        save_dir / "ckpt_final.pt",
    )
    if writer is not None:
        writer.close()
    print("Training done.")


if __name__ == "__main__":
    main()
