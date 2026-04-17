"""
PN-vs-PN 测试脚本 — 验证V5惯性系加速度动力学模型
====================================================
攻防双方均使用比例导引(PN)，验证:
1. 拦截器能否用PN打中进攻飞行器 (拦截测试)
2. 进攻飞行器能否用PN打中HVT目标 (打击测试)
3. 飞行器是否能平飞 (action=[0,0,0]不坠落)
4. 排他锁定是否生效 (每个进攻方最多被1个拦截器锁定)

用法: python scripts/test_pn_vs_pn.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.fov_penetration.config import get_config, G
from envs.fov_penetration.entities import Aircraft, HVT
from envs.fov_penetration.dynamics import step_dynamics_3d, action_to_control_3d
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv


def pn_guidance_to_target(aircraft, target_x, target_y, target_z,
                          prev_los_az, prev_los_el, dt,
                          N=4, target_vx=0.0, target_vy=0.0, target_vz=0.0):
    """
    3D比例导引 — 输出 (ax_cmd, an_pitch_cmd, an_yaw_cmd)
    直接输出惯性系加速度，无需合成步骤
    """
    dx = target_x - aircraft.x
    dy = target_y - aircraft.y
    dz = target_z - aircraft.z
    r = max(np.sqrt(dx**2 + dy**2 + dz**2), 1.0)

    los_az = np.arctan2(dy, dx)
    r_horiz = max(np.sqrt(dx**2 + dy**2), 1.0)
    los_el = np.arctan2(dz, r_horiz)

    # LOS角速率
    if prev_los_az is not None:
        d_az = np.arctan2(np.sin(los_az - prev_los_az), np.cos(los_az - prev_los_az))
        los_rate_az = d_az / dt
    else:
        los_rate_az = 0.0

    if prev_los_el is not None:
        d_el = np.arctan2(np.sin(los_el - prev_los_el), np.cos(los_el - prev_los_el))
        los_rate_el = d_el / dt
    else:
        los_rate_el = 0.0

    # 闭合速度
    cg = np.cos(aircraft.gamma)
    vx = aircraft.v * cg * np.cos(aircraft.heading)
    vy = aircraft.v * cg * np.sin(aircraft.heading)
    vz = aircraft.v * np.sin(aircraft.gamma)

    dvx = target_vx - vx
    dvy = target_vy - vy
    dvz = target_vz - vz
    v_closing = -(dx * dvx + dy * dvy + dz * dvz) / r

    # PN加速度指令 — V5直接输出俯仰/偏航
    an_yaw_cmd = N * v_closing * los_rate_az
    an_pitch_cmd = N * v_closing * los_rate_el + G * np.cos(aircraft.gamma)

    # 轴向: 重力补偿 + 微量加速
    ax_cmd = G * np.sin(aircraft.gamma) + 2.0

    # 饱和
    params = aircraft.params
    ax_cmd = np.clip(ax_cmd, params["ax_min"], params["ax_max"])
    an_pitch_cmd = np.clip(an_pitch_cmd, -params["an_pitch_max"], params["an_pitch_max"])
    an_yaw_cmd = np.clip(an_yaw_cmd, -params["an_yaw_max"], params["an_yaw_max"])

    return ax_cmd, an_pitch_cmd, an_yaw_cmd, los_az, los_el


def test_level_flight():
    """测试1: 平飞测试 — action=[0,0,0]是否保持水平飞行"""
    print("=" * 60)
    print("测试1: 平飞测试 (action=[0,0,0])")
    print("=" * 60)

    config = get_config()
    off_params = config["offensive"]

    ac = Aircraft(0, "offensive", off_params,
                  x=0, y=0, z=300, v=45.0, heading=0.0, gamma=0.0)

    dt = 0.01
    n_steps = 5000  # 50秒

    z_history = []
    v_history = []
    gamma_history = []
    heading_history = []

    for step in range(n_steps):
        # action=[0,0,0] → ax=0, an_pitch=G, an_yaw=0 (平飞trim)
        ac.step_with_action([0.0, 0.0, 0.0], dt)
        z_history.append(ac.z)
        v_history.append(ac.v)
        gamma_history.append(np.degrees(ac.gamma))
        heading_history.append(np.degrees(ac.heading))

    z_init = 300.0
    z_final = z_history[-1]
    v_final = v_history[-1]
    gamma_final = gamma_history[-1]
    heading_final = heading_history[-1]
    z_drift = abs(z_final - z_init)
    heading_drift = abs(heading_final - 0.0)

    print(f"  初始高度: {z_init:.1f}m")
    print(f"  最终高度: {z_final:.1f}m (漂移: {z_drift:.2f}m)")
    print(f"  最终速度: {v_final:.1f}m/s")
    print(f"  最终gamma: {gamma_final:.3f}°")
    print(f"  最终heading: {heading_final:.3f}° (漂移: {heading_drift:.3f}°)")
    print(f"  高度范围: [{min(z_history):.1f}, {max(z_history):.1f}]m")

    # 判定
    ok = z_drift < 5.0 and heading_drift < 1.0
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  结果: {status}")
    print()
    return ok


def test_attacker_pn_to_hvt():
    """测试2: 进攻飞行器用PN打击HVT"""
    print("=" * 60)
    print("测试2: 进攻飞行器PN打击HVT")
    print("=" * 60)

    config = get_config()
    off_params = config["offensive"]

    hvt = HVT(1200.0, 0.0, 0.0)

    # 从(-1200, 0, 300)出发，朝向HVT
    heading_to_hvt = np.arctan2(hvt.y - 0.0, hvt.x - (-1200.0))
    ac = Aircraft(0, "offensive", off_params,
                  x=-1200, y=0, z=300, v=45.0,
                  heading=heading_to_hvt, gamma=0.0)

    dt = 0.01
    max_steps = 8000
    prev_los_az = None
    prev_los_el = None
    hit = False
    min_dist = float('inf')

    for step in range(max_steps):
        dist = ac.distance_to(hvt.x, hvt.y, hvt.z)
        min_dist = min(min_dist, dist)

        if dist < 5.0:
            hit = True
            print(f"  命中! step={step}, dist={dist:.1f}m, t={step*dt:.1f}s")
            break

        ax, an_pitch, an_yaw, prev_los_az, prev_los_el = pn_guidance_to_target(
            ac, hvt.x, hvt.y, hvt.z,
            prev_los_az, prev_los_el, dt, N=3)
        ac.step(ax, an_pitch, an_yaw, dt)

    print(f"  最小脱靶量: {min_dist:.2f}m")
    print(f"  最终位置: ({ac.x:.0f}, {ac.y:.0f}, {ac.z:.0f})")
    print(f"  最终速度: {ac.v:.1f}m/s")

    ok = hit or min_dist < 5.0
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  结果: {status}")
    print()
    return ok


def test_interceptor_pn_to_attacker():
    """测试3: 拦截器用PN拦截进攻飞行器(直飞目标)"""
    print("=" * 60)
    print("测试3: 拦截器PN拦截直飞进攻方")
    print("=" * 60)

    config = get_config()
    off_params = config["offensive"]
    def_params = config["defensive"]

    hvt = HVT(1200.0, 0.0, 0.0)

    # 进攻方: 从(-1200, 0, 300)直飞HVT
    heading_to_hvt = np.arctan2(0.0, 1200.0 - (-1200.0))
    attacker = Aircraft(0, "offensive", off_params,
                        x=-1200, y=0, z=300, v=45.0,
                        heading=heading_to_hvt, gamma=0.0)

    # 拦截器: 从(600, 200, 350)出发, 面朝进攻方
    heading_to_atk = np.arctan2(0.0 - 200.0, -1200.0 - 600.0)
    interceptor = Aircraft(100, "defensive", def_params,
                           x=600, y=200, z=350, v=55.0,
                           heading=heading_to_atk, gamma=0.0)

    dt = 0.01
    max_steps = 8000
    prev_los_az_def = None
    prev_los_el_def = None
    hit = False
    min_dist = float('inf')

    for step in range(max_steps):
        # 进攻方: PN飞向HVT
        ax_atk, an_p_atk, an_y_atk = 0.0, G, 0.0  # 简单平飞直奔(无PN, 直线)
        # 给一点前向力维持速度
        ax_atk = G * np.sin(attacker.gamma) + 2.0
        ax_atk = np.clip(ax_atk, off_params["ax_min"], off_params["ax_max"])
        attacker.step(ax_atk, an_p_atk, an_y_atk, dt)

        # 拦截器: PN追击进攻方
        cos_g_a = np.cos(attacker.gamma)
        t_vx = attacker.v * cos_g_a * np.cos(attacker.heading)
        t_vy = attacker.v * cos_g_a * np.sin(attacker.heading)
        t_vz = attacker.v * np.sin(attacker.gamma)

        ax_def, an_p_def, an_y_def, prev_los_az_def, prev_los_el_def = pn_guidance_to_target(
            interceptor, attacker.x, attacker.y, attacker.z,
            prev_los_az_def, prev_los_el_def, dt, N=4,
            target_vx=t_vx, target_vy=t_vy, target_vz=t_vz)
        interceptor.step(ax_def, an_p_def, an_y_def, dt)

        dist = interceptor.distance_3d(attacker)
        min_dist = min(min_dist, dist)

        if dist < 5.0:
            hit = True
            print(f"  拦截成功! step={step}, dist={dist:.1f}m, t={step*dt:.1f}s")
            break

    print(f"  最小脱靶量: {min_dist:.2f}m")
    print(f"  拦截器最终位置: ({interceptor.x:.0f}, {interceptor.y:.0f}, {interceptor.z:.0f})")
    print(f"  进攻方最终位置: ({attacker.x:.0f}, {attacker.y:.0f}, {attacker.z:.0f})")

    ok = hit or min_dist < 10.0
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  结果: {status}")
    print()
    return ok


def test_env_integration():
    """测试4: 完整环境集成测试 (PN动作控制进攻方)"""
    print("=" * 60)
    print("测试4: 完整环境集成测试")
    print("=" * 60)

    env = FOVPenetrationEnv()
    obs, share_obs, avail = env.reset()

    n_steps = 0
    done = False

    while not done:
        n_steps += 1
        # 进攻方: 全部直飞 (action=[0,0,0] = 平飞)
        actions = [[0.0, 0.0, 0.0] for _ in range(env.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
        done = dones[0]

    info = infos[0]
    print(f"  结束原因: {info['done_reason']}")
    print(f"  总步数: {n_steps}")
    print(f"  命中HVT: {info['hit_count']}")
    print(f"  进攻方存活: {info['offensive_alive']}/{env.n_offensive}")
    print(f"  防御方存活: {info['defensive_alive']}/{env.n_defensive}")
    print(f"  排他锁定映射: {info.get('locked_target_by_defender', {})}")

    # 检查排他锁定
    lock_map = info.get('locked_target_by_defender', {})
    locked_targets = [v for v in lock_map.values() if v is not None]
    unique_targets = set(locked_targets)
    is_exclusive = len(locked_targets) == len(unique_targets)
    print(f"  锁定排他性: {'✓ 无重复' if is_exclusive else '✗ 有重复锁定!'}")

    # 检查飞行器高度
    for off in env.offensives:
        if off.alive:
            print(f"  进攻方{off.uid}: z={off.z:.1f}m, v={off.v:.1f}m/s")
    for d in env.defensives:
        if d.alive:
            print(f"  拦截器{d.uid}: z={d.z:.1f}m, v={d.v:.1f}m/s")

    # 基本检查: 环境能跑完不崩溃
    ok = True
    status = "✓ PASS"
    print(f"  结果: {status}")
    print()
    return ok


def test_exclusive_lock():
    """测试5: 排他锁定验证 — 4v4场景下每个进攻方最多被1个拦截器锁定"""
    print("=" * 60)
    print("测试5: 排他锁定验证")
    print("=" * 60)

    env = FOVPenetrationEnv()
    obs, share_obs, avail = env.reset()

    max_violation = 0
    violation_steps = 0
    total_steps = 0

    for step in range(3000):  # 30秒
        actions = [[0.0, 0.0, 0.0] for _ in range(env.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
        total_steps += 1

        if dones[0]:
            break

        # 检查锁定映射
        lock_map = infos[0].get('locked_target_by_defender', {})
        locked_targets = [v for v in lock_map.values() if v is not None]

        # 统计每个进攻方被几个拦截器锁定
        from collections import Counter
        counts = Counter(locked_targets)
        for off_idx, count in counts.items():
            if count > 1:
                violation_steps += 1
                max_violation = max(max_violation, count)
                if violation_steps <= 3:
                    print(f"  ⚠ step={step}: 进攻方{off_idx}被{count}个拦截器锁定!")

    print(f"  总步数: {total_steps}")
    print(f"  违反排他锁定的步数: {violation_steps}")
    print(f"  最大重复锁定: {max_violation}")

    ok = violation_steps == 0
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"  结果: {status}")
    print()
    return ok


def test_attacker_pn_env():
    """测试6: 在环境中用PN控制进攻方打击HVT"""
    print("=" * 60)
    print("测试6: 环境中进攻方PN打击HVT")
    print("=" * 60)

    env = FOVPenetrationEnv()
    obs, share_obs, avail = env.reset()

    hvt = env.hvt
    N_pn = 3  # PN导引比

    # 每个进攻方的LOS状态
    prev_los_az = [None] * env.n_agents
    prev_los_el = [None] * env.n_agents

    n_steps = 0
    done = False

    while not done and n_steps < 8000:
        n_steps += 1
        actions = []
        for i, off in enumerate(env.offensives):
            if not off.alive or off.hit_hvt:
                actions.append([0.0, 0.0, 0.0])
                continue

            # 用PN制导到HVT
            ax, an_p, an_y, prev_los_az[i], prev_los_el[i] = pn_guidance_to_target(
                off, hvt.x, hvt.y, hvt.z,
                prev_los_az[i], prev_los_el[i], env.dt, N=N_pn)

            # 反映射为归一化动作 [-1, 1]
            params = off.params
            # ax映射逆
            ax_center = 0.0
            if ax <= ax_center:
                a0 = (ax - ax_center) / max(ax_center - params["ax_min"], 0.01)
            else:
                a0 = (ax - ax_center) / max(params["ax_max"] - ax_center, 0.01)
            # an_pitch映射逆
            an_pitch_trim = G
            if an_p >= an_pitch_trim:
                a1 = (an_p - an_pitch_trim) / max(params["an_pitch_max"] - an_pitch_trim, 0.01)
            else:
                a1 = (an_p - an_pitch_trim) / max(params["an_pitch_max"] + an_pitch_trim, 0.01)
            # an_yaw映射逆
            a2 = an_y / max(params["an_yaw_max"], 0.01)

            actions.append([
                np.clip(a0, -1.0, 1.0),
                np.clip(a1, -1.0, 1.0),
                np.clip(a2, -1.0, 1.0)
            ])

        obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
        done = dones[0]

    info = infos[0]
    print(f"  结束原因: {info['done_reason']}")
    print(f"  步数: {n_steps}, 时间: {n_steps * env.dt:.1f}s")
    print(f"  命中HVT: {info['hit_count']}")
    print(f"  进攻方存活: {info['offensive_alive']}/{env.n_offensive}")
    print(f"  防御方存活: {info['defensive_alive']}/{env.n_defensive}")
    miss_dists = info.get('terminal_miss_distance_per_agent', [])
    if miss_dists:
        print(f"  各进攻方最小脱靶量: {[f'{d:.1f}m' for d in miss_dists]}")
        print(f"  全局最小脱靶量: {min(miss_dists):.1f}m")

    ok = info['hit_count'] > 0
    status = "✓ PASS" if ok else "✗ FAIL (但可能被拦截)"
    print(f"  结果: {status}")
    print()
    return ok


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PN-vs-PN 动力学验证测试 (V5 惯性系加速度)")
    print("=" * 60 + "\n")

    results = {}
    results["平飞测试"] = test_level_flight()
    results["攻击方PN→HVT"] = test_attacker_pn_to_hvt()
    results["拦截器PN→攻击方"] = test_interceptor_pn_to_attacker()
    results["环境集成"] = test_env_integration()
    results["排他锁定"] = test_exclusive_lock()
    results["环境中PN打击"] = test_attacker_pn_env()

    print("\n" + "=" * 60)
    print("  测试汇总")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ★ 所有测试通过! V5惯性系加速度模型工作正常")
    else:
        print("  ⚠ 部分测试失败, 请检查")
    print()
