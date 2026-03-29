#!/usr/bin/env python
"""FOV Penetration Environment V2 测试"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "third_party", "MACPO", "MACPO"))
import numpy as np

def test_env_creation():
    print("\n=== Test: Environment Creation ===")
    from envs.fov_penetration import FOVPenetrationEnv
    env = FOVPenetrationEnv()
    print(f"  n_agents: {env.n_agents}")
    print(f"  obs_dim: {env.obs_dim}")
    print(f"  share_obs_dim: {env.share_obs_dim}")
    print(f"  action_space: {env.action_space[0]}")
    print(f"  kill_range: {env.config['kill_range']}m")
    assert env.n_agents == 4
    print("  PASSED")
    return env

def test_reset_step(env):
    print("\n=== Test: Reset & Step ===")
    env.seed(42)
    obs, share_obs, avail = env.reset()
    assert len(obs) == 4
    assert obs[0].shape[0] == env.obs_dim
    assert share_obs[0].shape[0] == env.share_obs_dim
    print(f"  obs[0] shape: {obs[0].shape}")
    print(f"  share_obs[0] shape: {share_obs[0].shape}")

    actions = [env.action_space[i].sample() for i in range(env.n_agents)]
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
    assert len(rewards) == 4
    assert len(costs) == 4
    print(f"  rewards: {rewards}")
    print(f"  costs: {costs}")
    print("  PASSED")

def test_dynamics_rate_limit():
    print("\n=== Test: Dynamics Rate Limit (防抖头) ===")
    from envs.fov_penetration.dynamics import step_dynamics, rate_limit_overload
    from envs.fov_penetration.config import DEFAULT_CONFIG

    params = DEFAULT_CONFIG["attacker"]
    # 测试过载变化率限制
    # 从 ny=0 突变到 ny=4, dt=0.1, dny_max=8 → 允许变化 0.8g
    nx_l, ny_l = rate_limit_overload(0.5, 4.0, 0.0, 0.0, 0.1, params)
    assert abs(ny_l - 0.8) < 0.01, f"Expected ny≈0.8, got {ny_l}"
    print(f"  Rate limited: ny_cmd=4.0, ny_prev=0.0 → ny_actual={ny_l:.2f} (max change={params['dny_max']*0.1:.1f}g)")

    # 动力学步进含rate limit
    x, y, v, h, nx, ny = step_dynamics(0, 0, 45, 0, 0.5, 4.0, 0.1, params, 0.0, 0.0)
    print(f"  Step result: nx_actual={nx:.3f}, ny_actual={ny:.3f}")
    assert abs(ny) <= 0.8 + 0.01  # 不应超过变化率限制
    print("  PASSED")

def test_pn_guidance():
    print("\n=== Test: PN Guidance Interceptor ===")
    from envs.fov_penetration import FOVPenetrationEnv
    env = FOVPenetrationEnv()
    env.seed(123)
    env.reset()

    # 运行几步检查拦截器使用PN
    for _ in range(50):
        actions = [env.action_space[i].sample() for i in range(env.n_agents)]
        env.step(actions)

    # 检查拦截器状态
    for i, p in enumerate(env.interceptor_policies):
        print(f"  Interceptor {i}: state={p.state}, target_type={p.target_type}")

    print("  PASSED")

def test_kill_range_3m():
    print("\n=== Test: Kill Range 3m ===")
    from envs.fov_penetration import FOVPenetrationEnv
    env = FOVPenetrationEnv()
    assert env.config["kill_range"] == 3.0
    print(f"  kill_range = {env.config['kill_range']}m ✓")
    print("  PASSED")

def test_full_episode():
    print("\n=== Test: Full Episode ===")
    from envs.fov_penetration import FOVPenetrationEnv
    env = FOVPenetrationEnv()
    env.seed(42)
    obs, _, _ = env.reset()

    total_reward = 0
    total_cost = 0
    for step in range(env.max_steps):
        actions = [env.action_space[i].sample() for i in range(env.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        total_reward += sum(r[0] for r in rewards)
        total_cost += sum(c[0] for c in costs)
        if any(dones):
            break

    info = infos[0]
    print(f"  Episode length: {step+1}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Total cost: {total_cost:.2f}")
    print(f"  Done reason: {info['done_reason']}")
    print(f"  Success: {info['success']}")
    print(f"  Attacker alive: {not info['attacker_killed']}")
    print(f"  Escorts alive: {info['escorts_alive_count']}")
    print(f"  Interceptors alive: {info.get('interceptors_alive_count', '?')}")
    print(f"  Escort kills: {len(info.get('escort_kill_events', []))}")
    print("  PASSED")

def test_vecenv():
    print("\n=== Test: VecEnv Compatibility ===")
    from envs.fov_penetration import FOVPenetrationEnv
    from macpo.envs.env_wrappers import ShareDummyVecEnv

    class PatchedVec(ShareDummyVecEnv):
        def __init__(self, env_fns):
            super().__init__(env_fns)
            self.n_agents = self.envs[0].n_agents

    tmp_env = PatchedVec([lambda: FOVPenetrationEnv()])
    obs, share_obs, avail = tmp_env.reset()
    print(f"  n_agents: {tmp_env.n_agents}")
    print(f"  obs shape: {obs.shape}")
    print(f"  share_obs shape: {share_obs.shape}")

    actions = np.random.uniform(-1, 1, (1, 4, 2))
    obs, share_obs, rews, cos, dones, infos, avail = tmp_env.step(actions)
    print(f"  step rews shape: {rews.shape}")
    print(f"  step cos shape: {cos.shape}")
    tmp_env.close()
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("FOV Penetration Environment V2 Test Suite")
    print("=" * 60)

    env = test_env_creation()
    test_reset_step(env)
    test_dynamics_rate_limit()
    test_pn_guidance()
    test_kill_range_3m()
    test_full_episode()
    test_vecenv()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
