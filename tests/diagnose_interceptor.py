"""
诊断拦截器在交汇后的行为：
1. 交汇前后的制导指令变化
2. 拦截器是否能有效回头追击
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.config import G
from tests.test_v4_pn_vs_pn import OffensivePNPolicy


def angle_between_heading_and_target(ac, tx, ty):
    """计算飞行器航向与目标方向之间的夹角 (deg)"""
    dx = tx - ac.x
    dy = ty - ac.y
    target_bearing = np.arctan2(dy, dx)
    diff = np.arctan2(np.sin(target_bearing - ac.heading),
                      np.cos(target_bearing - ac.heading))
    return np.rad2deg(diff)


def main():
    env = FOVPenetrationEnv(scenario='scenario_1')
    cfg = env.config
    dt = cfg['dt']
    
    off_policies = [
        OffensivePNPolicy(env.hvt, cfg['offensive'], nav_gain=3.0)
        for _ in range(cfg['n_offensive'])
    ]
    
    obs, share_obs, _ = env.reset()
    for p in off_policies:
        p.reset()

    # 记录每架拦截器的关键信息
    print("=" * 90)
    print("拦截器行为诊断")
    print("=" * 90)
    
    # 打印初始状态
    for di, d in enumerate(env.defensives):
        pol = env.defensive_policies[di]
        tgt_idx = pol.initial_assigned_target_idx
        print(f"D{di}: pos=({d.x:.0f},{d.y:.0f},{d.z:.0f}) heading={np.rad2deg(d.heading):.1f}° "
              f"v={d.v:.1f} assigned_target=off{tgt_idx} lock_mode={pol.lock_mode}")
    for oi, off in enumerate(env.offensives):
        print(f"O{oi}: pos=({off.x:.0f},{off.y:.0f},{off.z:.0f}) heading={np.rad2deg(off.heading):.1f}° v={off.v:.1f}")
    print(f"HVT: ({env.hvt.x:.0f},{env.hvt.y:.0f},{env.hvt.z:.0f})")
    print()
    
    # 记录关键时刻
    # 追踪每架拦截器是否已交汇
    passed = [False] * cfg['n_defensive']
    min_dist_to_target = [float('inf')] * cfg['n_defensive']
    pass_step = [None] * cfg['n_defensive']
    
    max_steps = 8000
    
    for step in range(max_steps):
        # 进攻方PN
        actions = []
        for i, off in enumerate(env.offensives):
            if off.alive and not off.hit_hvt:
                action = off_policies[i].get_action(off, dt)
            else:
                action = [0.0, 0.0, 0.0]
            actions.append(action)
        
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        
        # 拦截器诊断
        for di, d in enumerate(env.defensives):
            pol = env.defensive_policies[di]
            if not d.alive:
                continue
            
            # 找到当前追击目标
            target = pol.target
            if target is None:
                continue
            
            dist = d.distance_3d(target)
            
            # 更新最小距离
            if dist < min_dist_to_target[di]:
                min_dist_to_target[di] = dist
            
            # 检测是否已交汇 (距离先减后增)
            if not passed[di] and min_dist_to_target[di] < 50.0 and dist > min_dist_to_target[di] + 30:
                passed[di] = True
                pass_step[di] = step
                
                # 交汇时详细信息
                off_angle = angle_between_heading_and_target(d, target.x, target.y)
                print(f"[t={step*dt:.1f}s] ★ D{di} 交汇! 最小距离={min_dist_to_target[di]:.1f}m")
                print(f"  D{di} pos=({d.x:.0f},{d.y:.0f},{d.z:.0f}) heading={np.rad2deg(d.heading):.1f}° v={d.v:.1f}")
                print(f"  目标 off: pos=({target.x:.0f},{target.y:.0f},{target.z:.0f}) heading={np.rad2deg(target.heading):.1f}°")
                print(f"  目标偏角={off_angle:.1f}° (正前方=0°)")
                print(f"  lock_mode={pol.lock_mode} demanded_ay={pol.demanded_ay:.1f}m/s² "
                      f"(={pol.demanded_ay/G:.2f}g) closing_v={pol.closing_speed:.1f}m/s")
                print()
            
            # 每5秒报告一次交汇后的拦截器状态
            if passed[di] and step % 500 == 0:
                off_angle = angle_between_heading_and_target(d, target.x, target.y)
                print(f"[t={step*dt:.1f}s] D{di} 交汇后 {(step-pass_step[di])*dt:.1f}s:")
                print(f"  pos=({d.x:.0f},{d.y:.0f},{d.z:.0f}) heading={np.rad2deg(d.heading):.1f}° v={d.v:.1f}")
                print(f"  目标 pos=({target.x:.0f},{target.y:.0f},{target.z:.0f})")
                print(f"  dist={dist:.0f}m 目标偏角={off_angle:.1f}° lock_mode={pol.lock_mode}")
                print(f"  demanded_ay={pol.demanded_ay:.1f}m/s² ({pol.demanded_ay/G:.2f}g) "
                      f"ay_max={d.params['ay_max']:.1f}m/s² ({d.params['ay_max']/G:.1f}g)")
                print(f"  actual: ax={d.ax:.1f} ay={d.ay:.1f} mu={np.rad2deg(d.mu):.1f}°")
                print(f"  closing_v={pol.closing_speed:.1f} los_rate_az={pol.los_rate_az:.4f} los_rate_el={pol.los_rate_el:.4f}")
                
                # 计算转弯半径
                if d.ay > 0.1:
                    turn_radius = d.v**2 / d.ay
                    print(f"  转弯半径={turn_radius:.0f}m (v²/ay)")
                print()
        
        # 检查终止
        if isinstance(dones, (list, np.ndarray)):
            if all(dones):
                break
        elif dones:
            break
    
    print("=" * 90)
    print("诊断总结")
    print("=" * 90)
    for di, d in enumerate(env.defensives):
        pol = env.defensive_policies[di]
        status = "交汇" if passed[di] else "未交汇"
        kill_str = "存活" if d.alive else "已死亡"
        target_idx = pol.assigned_target_idx
        print(f"D{di}: {status}, 最近距离={min_dist_to_target[di]:.1f}m, 状态={kill_str}, "
              f"lock_mode={pol.lock_mode}, target=off{target_idx}")
    print()
    for oi, off in enumerate(env.offensives):
        status = "命中HVT" if off.hit_hvt else ("死亡" if not off.alive else "存活")
        d_hvt = off.distance_to(env.hvt.x, env.hvt.y, env.hvt.z) if off.alive else -1
        print(f"O{oi}: {status}, d_hvt={d_hvt:.0f}m")
    print(f"总步数: {env.current_step}, 总耗时: {env.current_step*dt:.1f}s")


if __name__ == "__main__":
    main()
