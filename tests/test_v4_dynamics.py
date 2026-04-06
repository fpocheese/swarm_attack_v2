"""Smoke test for V4 dynamics model (ax, ay, mu)"""
import sys, traceback

try:
    from envs.fov_penetration.config import get_config, G
    cfg = get_config(scenario='scenario_1')
    print('[OK] config loaded')
    print(f'  offensive ax_min={cfg["offensive"]["ax_min"]}, ax_max={cfg["offensive"]["ax_max"]}, ay_max={cfg["offensive"]["ay_max"]:.2f} (n_max=2.5, g={G})')
    print(f'  defensive ax_min={cfg["defensive"]["ax_min"]}, ax_max={cfg["defensive"]["ax_max"]}, ay_max={cfg["defensive"]["ay_max"]:.2f} (n_max=5.0)')

    from envs.fov_penetration.dynamics import step_dynamics_3d, action_to_control_3d
    import numpy as np

    # Test action mapping
    params = cfg['offensive']
    ax, ay, mu = action_to_control_3d([0, 0, 0], params)
    print(f'[OK] action_to_control_3d([0,0,0]) = ax={ax:.2f}, ay={ay:.2f}, mu={mu:.4f}')
    ax, ay, mu = action_to_control_3d([1, 1, 1], params)
    print(f'[OK] action_to_control_3d([1,1,1]) = ax={ax:.2f}, ay={ay:.2f}, mu={mu:.4f}')
    ax, ay, mu = action_to_control_3d([-1, -1, -1], params)
    print(f'[OK] action_to_control_3d([-1,-1,-1]) = ax={ax:.2f}, ay={ay:.2f}, mu={mu:.4f}')

    # Test dynamics step
    x,y,z,v,h,g_a,ax_a,ay_a,mu_a = step_dynamics_3d(
        0, 0, 300, 60, 0, 0,
        0.0, 0.0, 0.0,
        0.01, params)
    print(f'[OK] step_dynamics_3d cruise: v={v:.2f}, heading={h:.4f}, gamma={g_a:.4f}')
    print(f'     pos=({x:.2f}, {y:.2f}, {z:.2f}), ax_actual={ax_a:.2f}, ay_actual={ay_a:.2f}')

    # Test with maneuver
    x2,y2,z2,v2,h2,g2,_,_,_ = step_dynamics_3d(
        0, 0, 300, 60, 0, 0,
        0.0, 15.0, 0.0,
        0.01, params)
    print(f'[OK] step_dynamics_3d maneuver(ay=15,mu=0): v={v2:.2f}, heading={h2:.4f}rad, psi_rate~{h2/0.01:.2f}rad/s')

    # Test entity
    from envs.fov_penetration.entities import Aircraft
    ac = Aircraft(0, 'offensive', params, x=0, y=0, z=300, v=60, heading=0, gamma=0)
    ac.step_with_action([0, 0, 0], 0.01)
    print(f'[OK] Aircraft.step_with_action: pos=({ac.x:.2f},{ac.y:.2f},{ac.z:.2f}), v={ac.v:.2f}, ax={ac.ax:.2f}, ay={ac.ay:.2f}, mu={ac.mu:.4f}')

    # Test full env
    from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
    env = FOVPenetrationEnv(scenario='scenario_1')
    obs, share_obs, avail = env.reset()
    print(f'[OK] env.reset() obs_shape={obs[0].shape}, n_agents={env.n_agents}')

    # Step with zero actions
    actions = [[0, 0, 0]] * env.n_agents
    obs2, share_obs2, rewards, dones, infos, costs, avail2 = env.step(actions)
    r_str = ", ".join(f"{float(r[0]) if hasattr(r, '__len__') else float(r):.2f}" for r in rewards)
    print(f'[OK] env.step() rewards=[{r_str}]')
    print(f'     all agents alive: {all(off.alive for off in env.offensives)}')
    print(f'     all defenders alive: {all(d.alive for d in env.defensives)}')

    # Run 100 steps
    for step_i in range(100):
        actions = [[0, 0, 0]] * env.n_agents
        obs2, share_obs2, rewards, dones, infos, costs, avail2 = env.step(actions)
    print(f'[OK] 100 steps done. Step={env.current_step}')
    off0 = env.offensives[0]
    print(f'     off[0] pos=({off0.x:.1f},{off0.y:.1f},{off0.z:.1f}), v={off0.v:.1f}, ax={off0.ax:.2f}, ay={off0.ay:.2f}')
    def0 = env.defensives[0]
    print(f'     def[0] pos=({def0.x:.1f},{def0.y:.1f},{def0.z:.1f}), v={def0.v:.1f}, ax={def0.ax:.2f}, ay={def0.ay:.2f}')

    print()
    print('=== ALL SMOKE TESTS PASSED ===')

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
