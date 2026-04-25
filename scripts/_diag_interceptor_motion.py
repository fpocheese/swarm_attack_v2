"""诊断拦截器是否运动 / PN 是否生效。

进攻方采用 trim 动作 [0,0,0] 直飞 HVT。
打印每隔若干步: 进攻方/拦截器位置、速度、距离、PN 指令、锁定状态。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv
from envs.fov_penetration.config import get_config

cfg = get_config(scenario="scenario_1")
env = FOVPenetrationEnv(cfg)
env.seed(0)
obs, share, avail = env.reset()

n_off = env.n_offensive
n_def = env.n_defensive
print(f"n_off={n_off}, n_def={n_def}, dt={env.dt}, max_steps={env.config['max_steps']}")
print(f"HVT={env.config['hvt_position']}")
for i, o in enumerate(env.offensives):
    print(f"  OFF[{i}] init pos=({o.x:.1f},{o.y:.1f},{o.z:.1f}) v={o.v:.1f} heading={np.rad2deg(o.heading):.1f}deg")
for i, d in enumerate(env.defensives):
    print(f"  DEF[{i}] init pos=({d.x:.1f},{d.y:.1f},{d.z:.1f}) v={d.v:.1f} heading={np.rad2deg(d.heading):.1f}deg")

trim = np.zeros((n_off, 3), dtype=np.float32)

prev_def_pos = [(d.x, d.y, d.z) for d in env.defensives]
log_interval = 200
max_steps = 4000

for step in range(1, max_steps + 1):
    obs, share, rew, costs, done, info, avail = env.step(trim)

    if step % log_interval == 0 or step == 1:
        print(f"\n=== step {step} (t={step*env.dt:.2f}s) ===")
        for i, o in enumerate(env.offensives):
            d_hvt = np.linalg.norm([o.x - env.config['hvt_position'][0],
                                    o.y - env.config['hvt_position'][1],
                                    o.z - env.config['hvt_position'][2]])
            print(f"  OFF[{i}] alive={o.alive} hit={o.hit_hvt} pos=({o.x:.1f},{o.y:.1f},{o.z:.1f}) v={o.v:.1f} d2hvt={d_hvt:.1f}")
        for i, d in enumerate(env.defensives):
            dx = d.x - prev_def_pos[i][0]
            dy = d.y - prev_def_pos[i][1]
            dz = d.z - prev_def_pos[i][2]
            disp = np.sqrt(dx*dx + dy*dy + dz*dz)
            policy = env.defensive_policies[i]
            min_off_dist = min(np.linalg.norm([d.x-o.x, d.y-o.y, d.z-o.z]) for o in env.offensives if o.alive)
            print(f"  DEF[{i}] alive={d.alive} pos=({d.x:.1f},{d.y:.1f},{d.z:.1f}) v={d.v:.1f} "
                  f"disp_since_last={disp:.1f}m lock_mode={policy.lock_mode} "
                  f"locked={policy.current_locked_target_idx} min_d2off={min_off_dist:.1f}")
        prev_def_pos = [(d.x, d.y, d.z) for d in env.defensives]

    if all(not o.alive or o.hit_hvt for o in env.offensives):
        print(f"\nAll offensives done at step {step}")
        break

print("\n=== final ===")
for i, o in enumerate(env.offensives):
    print(f"  OFF[{i}] alive={o.alive} hit_hvt={o.hit_hvt}")
for i, d in enumerate(env.defensives):
    print(f"  DEF[{i}] alive={d.alive}")
