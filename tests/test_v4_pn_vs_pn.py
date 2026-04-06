"""
V4 动力学验证测试: 进攻方也用 PN 制导飞向 HVT, 拦截器用 PN 制导拦截
=======================================================================
运行3次完整仿真, 每次生成3D GIF动画, 验证新动力学模型(ax, ay, mu)下
攻防双方的飞行轨迹是否合理。

用法:
  conda run --no-capture-output -n rlgpu python -m tests.test_v4_pn_vs_pn
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio

from envs.fov_penetration import FOVPenetrationEnv
from envs.fov_penetration.render import render_frame
from envs.fov_penetration.config import G


class OffensivePNPolicy:
    """进攻方简易PN制导: 直接飞向HVT (用新的 ax/ay/mu 控制输出)"""

    def __init__(self, hvt, params, nav_gain=3.0):
        self.hvt = hvt
        self.params = params
        self.N = nav_gain
        self.prev_los_az = None
        self.prev_los_el = None

    def reset(self):
        self.prev_los_az = None
        self.prev_los_el = None

    def get_action(self, aircraft, dt):
        """返回归一化动作 [-1,1]^3, 由 env 内部映射到 (ax, ay, mu)"""
        # 计算到HVT的LOS
        dx = self.hvt.x - aircraft.x
        dy = self.hvt.y - aircraft.y
        dz = self.hvt.z - aircraft.z
        r = max(np.sqrt(dx**2 + dy**2 + dz**2), 1.0)

        los_az = np.arctan2(dy, dx)
        r_horiz = max(np.sqrt(dx**2 + dy**2), 1.0)
        los_el = np.arctan2(dz, r_horiz)

        # LOS角速率
        if self.prev_los_az is not None:
            d_az = np.arctan2(np.sin(los_az - self.prev_los_az),
                              np.cos(los_az - self.prev_los_az))
            los_rate_az = d_az / dt
        else:
            los_rate_az = 0.0

        if self.prev_los_el is not None:
            d_el = np.arctan2(np.sin(los_el - self.prev_los_el),
                              np.cos(los_el - self.prev_los_el))
            los_rate_el = d_el / dt
        else:
            los_rate_el = 0.0

        self.prev_los_az = los_az
        self.prev_los_el = los_el

        # 闭合速度 (HVT静止, V_closing = -dR/dt ≈ V*cos(off-axis))
        cos_g = np.cos(aircraft.gamma)
        vx = aircraft.v * cos_g * np.cos(aircraft.heading)
        vy = aircraft.v * cos_g * np.sin(aircraft.heading)
        vz = aircraft.v * np.sin(aircraft.gamma)
        v_closing = max((dx * vx + dy * vy + dz * vz) / r, 10.0)

        # PN加速度指令 (m/s²)
        a_h = self.N * v_closing * los_rate_az
        a_v = self.N * v_closing * los_rate_el + G * np.cos(aircraft.gamma)

        ay_cmd = np.sqrt(a_h**2 + a_v**2)
        mu_cmd = np.arctan2(a_v, a_h)

        # 轴向: 重力补偿 (维持速度)
        ax_cmd = G * np.sin(aircraft.gamma)

        # 映射到归一化动作 [-1,1]^3
        params = self.params
        ax_min = params["ax_min"]
        ax_max = params["ax_max"]
        ay_max = params["ay_max"]

        # ax -> action[0]: 逆映射 action_to_control_3d
        ax_center = 0.0
        if ax_cmd <= ax_center:
            a0 = (ax_cmd - ax_center) / max(ax_center - ax_min, 1e-6)
        else:
            a0 = (ax_cmd - ax_center) / max(ax_max - ax_center, 1e-6)

        # ay -> action[1]: ay = (a1+1)*0.5*ay_max => a1 = 2*ay/ay_max - 1
        a1 = 2.0 * ay_cmd / max(ay_max, 1e-6) - 1.0

        # mu -> action[2]: mu = a2*pi => a2 = mu/pi
        a2 = mu_cmd / np.pi

        action = np.clip([a0, a1, a2], -1.0, 1.0)
        return action


def run_one_episode(env, off_policies, max_steps=3000, frame_stride=10):
    """运行一个episode, 返回 (frames, info_dict)"""
    obs, share_obs, _ = env.reset()

    # 重置进攻方PN策略
    for p in off_policies:
        p.reset()

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    frames = []
    dt = env.dt

    done = False
    for step in range(max_steps):
        # 进攻方用PN制导
        actions = []
        for i, off in enumerate(env.offensives):
            if off.alive and not off.hit_hvt:
                action = off_policies[i].get_action(off, dt)
            else:
                action = [0.0, 0.0, 0.0]
            actions.append(action)

        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)

        # 每 frame_stride 步渲染一帧
        if step % frame_stride == 0:
            render_frame(ax, env, step_num=step + 1, show_fov=True)
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image.copy())

        # 周期性位置诊断
        if step % 1000 == 0 and step > 0:
            for oi, off in enumerate(env.offensives):
                d2hvt = np.sqrt((off.x - env.hvt.x)**2 + (off.y - env.hvt.y)**2 + (off.z - env.hvt.z)**2)
                print(f"    [t={step*dt:.0f}s] off{oi}: pos=({off.x:.0f},{off.y:.0f},{off.z:.0f}) v={off.v:.1f} d_hvt={d2hvt:.0f}m alive={off.alive}")

        # 检查终止
        if isinstance(dones, (list, np.ndarray)):
            if all(dones):
                done = True
                break
        elif dones:
            done = True
            break

    # 渲染最后一帧
    render_frame(ax, env, step_num=env.current_step, show_fov=True)
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image.copy())
    plt.close(fig)

    # 收集结果
    n_hit = sum(1 for o in env.offensives if o.hit_hvt)
    n_off_alive = sum(1 for o in env.offensives if o.alive)
    n_def_alive = sum(1 for d in env.defensives if d.alive)
    n_off_killed = sum(1 for o in env.offensives if not o.alive)
    n_def_killed = sum(1 for d in env.defensives if not d.alive)

    info = {
        "steps": env.current_step,
        "done": done,
        "n_hit_hvt": n_hit,
        "n_off_alive": n_off_alive,
        "n_off_killed": n_off_killed,
        "n_def_alive": n_def_alive,
        "n_def_killed": n_def_killed,
    }
    return frames, info


def main():
    output_dir = "outputs/gifs/v4_pn_vs_pn_test"
    os.makedirs(output_dir, exist_ok=True)

    n_episodes = 3
    max_steps = 8000  # dt=0.01 => 80s (需要~40s飞完2400m)
    frame_stride = 10  # 每10步一帧 => 0.1s/帧

    env = FOVPenetrationEnv(scenario='scenario_1')
    cfg = env.config

    print("=" * 60)
    print("V4 动力学验证: 进攻方PN制导 vs 拦截器PN制导")
    print("=" * 60)
    print(f"  n_offensive={cfg['n_offensive']}, n_defensive={cfg['n_defensive']}")
    print(f"  进攻方 ay_max={cfg['offensive']['ay_max']:.1f} m/s² (n={cfg['offensive']['ay_max']/G:.1f}g)")
    print(f"  拦截器 ay_max={cfg['defensive']['ay_max']:.1f} m/s² (n={cfg['defensive']['ay_max']/G:.1f}g)")
    print(f"  dt={cfg['dt']}, max_steps={max_steps}, frame_stride={frame_stride}")
    print(f"  输出目录: {output_dir}")
    print()

    for ep in range(n_episodes):
        print(f"--- Episode {ep+1}/{n_episodes} ---")

        # 创建进攻方PN策略
        off_policies = [
            OffensivePNPolicy(env.hvt, cfg['offensive'], nav_gain=3.0)
            for _ in range(cfg['n_offensive'])
        ]

        frames, info = run_one_episode(env, off_policies,
                                       max_steps=max_steps,
                                       frame_stride=frame_stride)

        gif_path = os.path.join(output_dir, f"ep{ep+1:02d}_pn_vs_pn.gif")
        imageio.mimsave(gif_path, frames, fps=12)

        print(f"  steps={info['steps']}, done={info['done']}")
        print(f"  进攻方: {info['n_hit_hvt']} hit HVT, {info['n_off_killed']} killed, {info['n_off_alive']} alive")
        print(f"  拦截器: {info['n_def_killed']} killed, {info['n_def_alive']} alive")
        print(f"  GIF saved: {gif_path} ({len(frames)} frames)")
        print()

    print("=" * 60)
    print(f"全部完成! GIF保存在: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
