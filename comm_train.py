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
    completion_reward: float,
    assigned_target_reward: float,
    coverage_delta_reward: float,
    coverage_step_reward: float,
    hover_penalty: float,
    shapes: list[str] | None = None,
    wind_prob: float = 0.0,
    wind_strength: int = 1,
    randomize_wind: bool = False,
    randomize_comm_fail: bool = False,
) -> tuple[TransformedEnv, ShapeFormationEnv]:
    base = ShapeFormationEnv(
        grid_size=grid_size,
        n_agents=n_agents,
        max_steps=max_steps,
        comm_fail_prob=comm_fail_prob,
        shaping_coef=shaping_coef,
        completion_reward=completion_reward,
        assigned_target_reward=assigned_target_reward,
        coverage_delta_reward=coverage_delta_reward,
        coverage_step_reward=coverage_step_reward,
        hover_penalty=hover_penalty,
        shapes=shapes,
        wind_prob=wind_prob,
        wind_strength=wind_strength,
        randomize_wind=randomize_wind,
        randomize_comm_fail=randomize_comm_fail,
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
    # Return the unwrapped base env too: SyncDataCollector steps this exact
    # object, so mutating base.collision_penalty mid-training (collision-penalty
    # curriculum) takes effect on the next collected batch.
    return env, base


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
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="checkpoints_comm20")
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument(
        "--early-stop-success",
        type=float,
        default=0.0,
        help=(
            "Stop training once the batch success rate reaches this fraction "
            "(e.g. 1.0 = 100%%) for --early-stop-patience consecutive iters; the "
            "current ckpt is saved and the run exits early. 0 disables."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="Consecutive iters at/above --early-stop-success required to stop.",
    )
    parser.add_argument(
        "--save-best-above",
        type=float,
        default=0.0,
        help=(
            "Whenever the batch success rate is >= this fraction AND beats the best "
            "seen so far, save that ckpt off the ckpt-every grid (replacing the "
            "previous off-grid best). 0 disables. Keeps the best high-success model."
        ),
    )
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
        "--completion-reward",
        type=float,
        default=30.0,
        help="Team reward given to every drone when the current formation stage is completed.",
    )
    parser.add_argument(
        "--assigned-target-reward",
        type=float,
        default=0.4,
        help="Extra per-step reward when a drone is exactly on its assigned target cell.",
    )
    parser.add_argument(
        "--coverage-delta-reward",
        type=float,
        default=0.2,
        help="Team reward multiplier when the stage reaches a new best target coverage count.",
    )
    parser.add_argument(
        "--coverage-step-reward",
        type=float,
        default=0.01,
        help=(
            "Dense per-step team reward = this x (#target cells currently occupied). "
            "Paid EVERY step (unlike --coverage-delta-reward), so an unfilled cell is "
            "a persistent loss -> pressures drones to fill/hold the last cells instead "
            "of freezing one step short. 0 disables."
        ),
    )
    parser.add_argument(
        "--hover-penalty",
        type=float,
        default=0.02,
        help="Penalty for choosing hover/stay outside the current target set.",
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=0.2,
        help=(
            "Per-drone penalty on a collision. Used as a constant unless the "
            "collision-penalty curriculum (--collision-penalty-start/-end) is set."
        ),
    )
    parser.add_argument(
        "--collision-penalty-start",
        type=float,
        default=None,
        help=(
            "Curriculum: initial (low) collision penalty so early exploration is "
            "not punished into a frozen policy. Defaults to --collision-penalty "
            "(constant, no curriculum)."
        ),
    )
    parser.add_argument(
        "--collision-penalty-end",
        type=float,
        default=None,
        help=(
            "Curriculum: final (high) collision penalty, reached as success rate "
            "climbs. Defaults to --collision-penalty (constant, no curriculum)."
        ),
    )
    parser.add_argument(
        "--collision-penalty-ema",
        type=float,
        default=0.1,
        help=(
            "EMA smoothing factor for the success-rate signal that drives the "
            "collision-penalty curriculum. Higher = faster, noisier ramp."
        ),
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
        "--wind-prob",
        type=float,
        default=0.0,
        help=(
            "Per-step per-drone probability a wind gust perturbs the move. "
            "0 disables wind. With --randomize-wind this acts as the upper bound."
        ),
    )
    parser.add_argument(
        "--wind-strength",
        type=int,
        default=1,
        help="Number of cells a wind gust pushes a drone.",
    )
    parser.add_argument(
        "--randomize-wind",
        action="store_true",
        help=(
            "Domain randomization: each episode samples wind severity from "
            "[0, wind_prob] and a random direction. Use to train a robust policy."
        ),
    )
    parser.add_argument(
        "--randomize-comm-fail",
        action="store_true",
        help=(
            "Domain randomization for communication failure: each episode "
            "samples its drop rate from [0, comm_fail_prob]. Use to train a "
            "policy robust across a range of comm loss conditions."
        ),
    )
    parser.add_argument(
        "--load-ckpt",
        type=str,
        default="",
        help="Path to a previous checkpoint to warm-start from.",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help=(
            "Iteration offset for ckpt numbering when resuming. Saved files become "
            "ckpt_{start-iter + it + 1}.pt so a continued run keeps incrementing in the "
            "same folder (e.g. --start-iter 110 -> next save ckpt_115.pt). Default 0."
        ),
    )
    parser.add_argument(
        "--tb-logdir",
        type=str,
        default="runs_comm20",
        help="TensorBoard log directory (empty string disables logging).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    shapes = [shape.strip() for shape in args.shapes.split(",") if shape.strip()]

    env, base_env = make_env(
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
        coverage_step_reward=args.coverage_step_reward,
        hover_penalty=args.hover_penalty,
        shapes=shapes,
        wind_prob=args.wind_prob,
        wind_strength=args.wind_strength,
        randomize_wind=args.randomize_wind,
        randomize_comm_fail=args.randomize_comm_fail,
    )

    # Collision-penalty curriculum. Both bounds default to --collision-penalty,
    # so without --collision-penalty-start/-end the penalty is a fixed constant
    # (backward compatible). When set, the penalty ramps from start -> end as a
    # ratcheted EMA of the success rate climbs: a frozen policy has success ~0,
    # so the penalty stays low and movement can be learned before collisions are
    # punished hard. The ratchet keeps the penalty non-decreasing for a more
    # stationary critic.
    cp_start = args.collision_penalty if args.collision_penalty_start is None else args.collision_penalty_start
    cp_end = args.collision_penalty if args.collision_penalty_end is None else args.collision_penalty_end
    cp_curriculum = cp_start != cp_end
    base_env.collision_penalty = cp_start

    probe = ShapeFormationEnv(
        grid_size=args.grid_size,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        comm_fail_prob=args.comm_fail_prob,
        shaping_coef=args.shaping_coef,
        completion_reward=args.completion_reward,
        assigned_target_reward=args.assigned_target_reward,
        coverage_delta_reward=args.coverage_delta_reward,
        coverage_step_reward=args.coverage_step_reward,
        hover_penalty=args.hover_penalty,
        shapes=shapes,
        wind_prob=args.wind_prob,
        wind_strength=args.wind_strength,
        randomize_wind=args.randomize_wind,
        randomize_comm_fail=args.randomize_comm_fail,
    )
    obs_dim = probe.obs_dim
    n_actions = 5
    n_agents = probe.n_agents

    print(
        f"Training config: grid_size={args.grid_size}, n_agents={n_agents}, "
        f"max_steps={args.max_steps}, obs_dim={obs_dim}, shapes='{probe.formation_path.label}', "
        f"completion_reward={args.completion_reward}, "
        f"assigned_target_reward={args.assigned_target_reward}, "
        f"coverage_delta_reward={args.coverage_delta_reward}, "
        f"coverage_step_reward={args.coverage_step_reward}, hover_penalty={args.hover_penalty}"
    )
    if args.wind_prob > 0.0:
        print(
            f"Wind: prob={args.wind_prob}, strength={args.wind_strength}, "
            f"randomize={args.randomize_wind}"
        )
    if cp_curriculum:
        print(
            f"Collision-penalty curriculum: {cp_start:.3f} -> {cp_end:.3f} "
            f"(success-driven, ema={args.collision_penalty_ema})"
        )
    else:
        print(f"Collision penalty: {cp_start:.3f} (constant)")

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

    # Collision-penalty curriculum state: ratcheted EMA of the success rate.
    cp_success_ema = 0.0
    cp_progress = 0.0
    es_hits = 0  # consecutive iters meeting the early-stop success threshold
    best_success = -1.0          # best batch success rate seen (for --save-best-above)
    best_offsched_path = None    # last off-grid "best" ckpt saved (to replace it)

    # === Speed profiling: timing measurement for each iteration ===
    t_total_start = time.time()
    last_iter_end = t_total_start

    for it, data in enumerate(collector):
        t_rollout = time.time() - last_iter_end

        t0 = time.time()
        with torch.no_grad():
            gae(data)
        t_gae = time.time() - t0

        t0 = time.time()
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
        t_update = time.time() - t0

        ep_rew = data.get(("next", GROUP, "episode_reward"))
        done = data.get(("next", GROUP, "done"))
        terminated = data.get(("next", GROUP, "terminated"))
        finished_rewards = ep_rew[done]
        finished_terminated = terminated[done]
        n_finished_entries = finished_rewards.numel()
        mean_ep = finished_rewards.mean().item() if n_finished_entries > 0 else float("nan")
        success_rate = finished_terminated.float().mean().item() if n_finished_entries > 0 else float("nan")
        avg_loss = running_loss / max(1, n_updates)
        # Continue iter / frames numbering across resumes (--start-iter offset)
        # so a resumed run's logs and ckpt numbers pick up where it left off.
        global_iter = args.start_iter + it + 1
        frames_seen = global_iter * args.frames_per_batch

        # Update the collision-penalty curriculum from this batch's success rate.
        # Only step the EMA when episodes actually finished (otherwise success is
        # nan). cp_progress ratchets up so the penalty never relaxes.
        if cp_curriculum and n_finished_entries > 0:
            cp_success_ema = (
                (1.0 - args.collision_penalty_ema) * cp_success_ema
                + args.collision_penalty_ema * success_rate
            )
            cp_progress = max(cp_progress, cp_success_ema)
            base_env.collision_penalty = cp_start + (cp_end - cp_start) * min(1.0, max(0.0, cp_progress))
        t_iter = t_rollout + t_gae + t_update
        cp_str = f"  coll_pen={base_env.collision_penalty:.3f}" if cp_curriculum else ""
        print(
            f"iter={global_iter:4d}  frames={frames_seen:>8d}  "
            f"mean_ep_reward={mean_ep:+7.3f}  success={success_rate:5.1%}  "
            f"loss={avg_loss:7.4f}{cp_str}  "
            f"| iter={t_iter:.1f}s (rollout={t_rollout:.1f}s gae={t_gae:.2f}s update={t_update:.1f}s)"
        )

        if writer is not None:
            if n_finished_entries > 0:
                writer.add_scalar("train/mean_episode_reward", mean_ep, frames_seen)
                writer.add_scalar("train/success_rate", success_rate, frames_seen)
            writer.add_scalar("train/loss_total", avg_loss, frames_seen)
            writer.add_scalar("train/collision_penalty", base_env.collision_penalty, frames_seen)

        ckpt_path = save_dir / f"ckpt_{global_iter}.pt"
        if (it + 1) % args.ckpt_every == 0:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt_path)

        # Off-schedule "best" save: success clears --save-best-above AND beats the
        # previous best -> keep that ckpt, replacing the previous off-grid best.
        if (args.save_best_above > 0 and n_finished_entries > 0
                and success_rate >= args.save_best_above and success_rate > best_success):
            best_success = success_rate
            if not ckpt_path.exists():
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt_path)
                if (best_offsched_path is not None and best_offsched_path != ckpt_path
                        and best_offsched_path.exists()):
                    best_offsched_path.unlink()
                best_offsched_path = ckpt_path
            print(f"[best] success {success_rate:.1%} (>= {args.save_best_above:.0%}) -> kept {ckpt_path.name}")

        # Early stop: success rate at/above threshold for `patience` iters in a row.
        if args.early_stop_success > 0 and n_finished_entries > 0 and success_rate >= args.early_stop_success:
            es_hits += 1
            if es_hits >= args.early_stop_patience:
                torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, ckpt_path)
                print(
                    f"[early-stop] success {success_rate:.1%} >= {args.early_stop_success:.0%} "
                    f"for {es_hits} iters -> saved {ckpt_path.name}, stopping."
                )
                break
        else:
            es_hits = 0

        last_iter_end = time.time()

    total_time = time.time() - t_total_start
    total_frames = (it + 1) * args.frames_per_batch
    print(
        f"\n=== Profiling summary ===\n"
        f"Total time: {total_time:.1f}s\n"
        f"Total frames: {total_frames}\n"
        f"FPS: {total_frames/total_time:.0f} frames/sec\n"
        f"1-hour throughput estimate: {total_frames/total_time*3600:.0f} frames/hour"
    )

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
