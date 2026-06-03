"""Quick smoke test for the wind disturbance added to ShapeFormationEnv.

Run:  python test_wind.py

Verifies three things about the new wind feature:
  [1] wind actually pushes drones,
  [2] domain randomization samples a fresh wind condition each episode,
  [3] disabling wind (the default) preserves the original behavior.
"""
from comm_env import ShapeFormationEnv


def test_deterministic_wind() -> bool:
    """Wind fixed to 'right' must drift drones that choose to stay."""
    env = ShapeFormationEnv(
        shapes=["GROUND", "X"], wind_prob=1.0, wind_dir=(0, 1), wind_strength=1
    )
    env.reset(seed=0)
    before = sorted(env.agent_pos.values())
    env.step({ag: 0 for ag in env.possible_agents})  # action 0 = stay
    after = sorted(env.agent_pos.values())
    drifted = all(a[0] == b[0] and a[1] == b[1] + 1 for a, b in zip(after, before))
    print(f"[1] deterministic wind-right drift : {'OK' if drifted else 'FAIL'}")
    print(f"    cols before {[c for _, c in before][:5]} -> after {[c for _, c in after][:5]}")
    return drifted


def test_domain_randomization() -> bool:
    """randomize_wind must sample a fresh severity/direction each episode."""
    env = ShapeFormationEnv(
        shapes=["GROUND", "X"], wind_prob=0.3, randomize_wind=True
    )
    samples = []
    for seed in range(6):
        env.reset(seed=seed)
        samples.append((round(env._cur_wind_prob, 3), env._cur_wind_dir))
    in_range = all(0.0 <= p <= 0.3 for p, _ in samples)
    varied = len({d for _, d in samples}) > 1
    print(f"[2] domain randomization per episode : {'OK' if in_range and varied else 'FAIL'}")
    for p, d in samples:
        print(f"    wind_prob={p}  dir={d}")
    return in_range and varied


def test_wind_off() -> bool:
    """Default (wind_prob=0) must keep obs_dim and leave wind inactive.

    obs_dim = 2 + 2*n_agents + 3*(n_agents-1) + 2 = 71 for the default 14 drones.
    """
    env = ShapeFormationEnv(shapes=["GROUND", "X"])
    env.reset(seed=0)
    ok = env.obs_dim == 71 and env._cur_wind_prob == 0.0
    print(f"[3] wind off keeps original behavior : {'OK' if ok else 'FAIL'}")
    print(f"    obs_dim={env.obs_dim}  cur_wind_prob={env._cur_wind_prob}")
    return ok


if __name__ == "__main__":
    results = [
        test_deterministic_wind(),
        test_domain_randomization(),
        test_wind_off(),
    ]
    print()
    print("ALL PASSED" if all(results) else "SOME TESTS FAILED")
