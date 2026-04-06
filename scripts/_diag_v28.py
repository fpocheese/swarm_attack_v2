"""V28 诊断: 检查进攻方/拦截器死亡原因"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv

env = FOVPenetrationEnv(config={"analytic_priors": {
    "enable_cone_cost": True,
    "enable_escape_reward": True,
    "enable_decoy_game": True,
    "enable_effective_penetration": True,
    "enable_hvt_guidance": True,
}})
obs, share_obs, avail = env.reset()
cfg = env.config

print(f"=== V28 诊断 ===")
print(f"dt={cfg['dt']}, max_steps={cfg['max_steps']}")
print(f"z_min={cfg['z_min']}, z_max={cfg['z_max']}")
print(f"map_size={cfg['map_size']}")
print(f"ground_kill_z={cfg.get('ground_kill_z', 'NOT SET')}")
print(f"boundary_kill={cfg.get('boundary_kill', 'NOT SET')}")
print(f"collision_kill_range={cfg.get('collision_kill_range', 'NOT SET')}")
print(f"speed_range: off={cfg.get('offensive_speed_range', 'N/A')}, def={cfg.get('defensive_speed_range', 'N/A')}")
print()

# 初始状态
print("=== 初始位置 ===")
for i, off in enumerate(env.offensives):
    print(f"  OFF[{i}]: pos=({off.x:.0f},{off.y:.0f},{off.z:.0f}) v={off.v:.1f} heading={np.degrees(off.heading):.1f}° gamma={np.degrees(off.gamma):.1f}°")
for i, d in enumerate(env.defensives):
    print(f"  DEF[{i}]: pos=({d.x:.0f},{d.y:.0f},{d.z:.0f}) v={d.v:.1f} heading={np.degrees(d.heading):.1f}° gamma={np.degrees(d.gamma):.1f}°")
print(f"  HVT: pos=({env.hvt.x:.0f},{env.hvt.y:.0f},{env.hvt.z:.0f})")
print()

# 跑到结束，追踪死亡事件
kill_log = []
z_history = {i: [] for i in range(env.n_offensive)}
def_z_history = {i: [] for i in range(len(env.defensives))}

prev_off_alive = [True] * env.n_offensive
prev_def_alive = [True] * len(env.defensives)

for step in range(cfg['max_steps'] + 5):
    # zero action = 直飞
    actions = [np.zeros(3, dtype=np.float32) for _ in range(env.n_agents)]
    obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
    
    # 记录高度
    for i, off in enumerate(env.offensives):
        z_history[i].append(off.z)
        if prev_off_alive[i] and not off.alive:
            kill_log.append(f"Step {step+1}: OFF[{i}] DIED at pos=({off.x:.0f},{off.y:.0f},{off.z:.1f}) v={off.v:.1f}")
            prev_off_alive[i] = False
    
    for i, d in enumerate(env.defensives):
        def_z_history[i].append(d.z)
        if prev_def_alive[i] and not d.alive:
            kill_log.append(f"Step {step+1}: DEF[{i}] DIED at pos=({d.x:.0f},{d.y:.0f},{d.z:.1f}) v={d.v:.1f}")
            prev_def_alive[i] = False
    
    # 每500步打印状态
    if (step + 1) % 500 == 0:
        off_alive = sum(1 for o in env.offensives if o.alive)
        def_alive = sum(1 for d in env.defensives if d.alive)
        off_z = [f"{o.z:.1f}" for o in env.offensives]
        print(f"Step {step+1}: off_alive={off_alive}, def_alive={def_alive}, off_z={off_z}")
    
    if dones[0]:
        info = infos[0]
        print(f"\n=== DONE at step {step+1} ===")
        print(f"  done_reason: {info['done_reason']}")
        print(f"  hit_count: {info['hit_count']}")
        print(f"  offensive_alive: {info['offensive_alive']}")
        print(f"  defensive_alive: {info['defensive_alive']}")
        print(f"  rewards: {[r[0] for r in rewards]}")
        break

print(f"\n=== 死亡日志 ({len(kill_log)} events) ===")
for log in kill_log:
    print(f"  {log}")

# 高度统计
print(f"\n=== 进攻方高度统计 ===")
for i in range(env.n_offensive):
    if z_history[i]:
        arr = np.array(z_history[i])
        print(f"  OFF[{i}]: min_z={arr.min():.1f}, max_z={arr.max():.1f}, final_z={arr[-1]:.1f}, "
              f"z<10_steps={np.sum(arr < 10)}, z<0_steps={np.sum(arr < 0)}")

print(f"\n=== 拦截器高度统计 ===")
for i in range(len(env.defensives)):
    if def_z_history[i]:
        arr = np.array(def_z_history[i])
        print(f"  DEF[{i}]: min_z={arr.min():.1f}, max_z={arr.max():.1f}, final_z={arr[-1]:.1f}, "
              f"z<10_steps={np.sum(arr < 10)}, z<0_steps={np.sum(arr < 0)}")

# 检查dynamics
print(f"\n=== 检查 dynamics ===\n")
from envs.fov_penetration.dynamics import action_to_control_3d
off0 = env.offensives[0]
print(f"  OFF[0] after episode: v={off0.v:.1f} gamma={np.degrees(off0.gamma):.1f}° az={off0.az:.1f}")
print(f"  params: {off0.params}")

# 检查边界/地面击杀条件
print(f"\n=== 击杀条件 ===")
for key in ['ground_kill_z', 'boundary_kill_margin', 'z_min', 'z_max', 'map_size']:
    print(f"  {key}: {cfg.get(key, 'NOT IN CONFIG')}")
