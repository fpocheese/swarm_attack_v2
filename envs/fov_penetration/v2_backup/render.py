"""
FOV Penetration Environment - 可视化渲染
==========================================
使用 matplotlib 动画渲染固定翼无人机俯视图
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无头模式，支持服务器环境
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Wedge
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import os


def render_frame(ax, env, step_num=0, show_trajectory=True, show_fov=True):
    """
    渲染单帧
    
    Args:
        ax: matplotlib axes
        env: FOVPenetrationEnv 实例
        step_num: 当前步数
        show_trajectory: 是否显示轨迹
        show_fov: 是否显示 FOV
    """
    ax.clear()
    cfg = env.config
    map_size = cfg["map_size"]
    
    # 设置地图
    margin = 500
    ax.set_xlim(-map_size - margin, map_size + margin)
    ax.set_ylim(-map_size - margin, map_size + margin)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'FOV Penetration - Step {step_num}', fontsize=12)
    
    # 绘制地图边界
    boundary = plt.Rectangle((-map_size, -map_size), 2 * map_size, 2 * map_size,
                             fill=False, edgecolor='gray', linewidth=2, linestyle='--')
    ax.add_patch(boundary)
    
    # ---- 绘制 HVT ----
    hvt = env.hvt
    ax.plot(hvt.x, hvt.y, marker='*', markersize=20, color='gold', 
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    ax.annotate('HVT', (hvt.x, hvt.y), textcoords="offset points", 
                xytext=(10, 10), fontsize=10, fontweight='bold', color='black')
    
    # 绘制命中区 (V2: kill_range 统一3m)
    kr = cfg.get("kill_range", cfg.get("hit_range", 3.0))
    hit_circle = plt.Circle((hvt.x, hvt.y), kr, 
                            fill=True, facecolor='gold', alpha=0.2, 
                            edgecolor='orange', linewidth=1)
    ax.add_patch(hit_circle)
    
    # ---- 绘制拦截器和 FOV ----
    for i, intc in enumerate(env.interceptors):
        color = 'red' if intc.alive else 'gray'
        alpha = 1.0 if intc.alive else 0.3
        
        if show_fov and intc.alive:
            # 绘制 FOV 扇形
            fov_half = np.degrees(cfg["fov_half_angle"])
            heading_deg = np.degrees(intc.heading)
            wedge = Wedge((intc.x, intc.y), cfg["detection_range"],
                         heading_deg - fov_half, heading_deg + fov_half,
                         facecolor='red', alpha=0.08, edgecolor='red',
                         linewidth=0.5, linestyle='--')
            ax.add_patch(wedge)
        
        # 危险区 (V2: 使用 kill_range 代替 danger_range)
        if intc.alive:
            dr = cfg.get("danger_range", cfg.get("kill_range", 3.0) * 50)  # 放大显示
            danger_circle = plt.Circle((intc.x, intc.y), dr,
                                      fill=True, facecolor='red', alpha=0.05,
                                      edgecolor='red', linewidth=0.5, linestyle=':')
            ax.add_patch(danger_circle)
        
        # 绘制飞行器
        _draw_aircraft(ax, intc, color, alpha, label=f'I{i}')
        
        # 轨迹
        if show_trajectory and intc.alive and len(intc.trajectory) > 1:
            traj = np.array(intc.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color='red', alpha=0.2, linewidth=0.8)
    
    # ---- 绘制 attacker ----
    atk = env.attacker
    color = 'blue' if atk.alive else 'gray'
    alpha = 1.0 if atk.alive else 0.3
    _draw_aircraft(ax, atk, color, alpha, label='ATK', marker_size=12)
    
    if show_trajectory and len(atk.trajectory) > 1:
        traj = np.array(atk.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.5, linewidth=1.5)
    
    # ---- 绘制 escorts ----
    for i, esc in enumerate(env.escorts):
        color = 'green' if esc.alive else 'gray'
        alpha = 1.0 if esc.alive else 0.3
        _draw_aircraft(ax, esc, color, alpha, label=f'E{i}')
        
        if show_trajectory and esc.alive and len(esc.trajectory) > 1:
            traj = np.array(esc.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color='green', alpha=0.3, linewidth=1.0)
    
    # ---- 图例 ----
    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Attacker'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='Escort'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Interceptor'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=15, label='HVT'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)
    
    # 状态信息
    info_text = f'Attacker: {"ALIVE" if atk.alive else "KILLED"}\n'
    info_text += f'Escorts alive: {sum(1 for e in env.escorts if e.alive)}/{len(env.escorts)}\n'
    info_text += f'FOV violations: {env.fov_violation_count}'
    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _draw_aircraft(ax, aircraft, color, alpha=1.0, label='', marker_size=10):
    """绘制单个飞行器（三角形表示固定翼无人机+朝向箭头）"""
    if not aircraft.alive:
        ax.plot(aircraft.x, aircraft.y, 'x', color='gray', markersize=8, alpha=0.5)
        return
    
    # 三角形表示固定翼无人机
    heading = aircraft.heading
    size = marker_size * 15  # 放大以在大地图上可见
    
    # 三角形顶点（机头朝前）
    triangle = np.array([
        [size, 0],
        [-size * 0.6, size * 0.4],
        [-size * 0.6, -size * 0.4],
    ])
    
    # 旋转
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated = triangle @ rotation.T
    
    # 平移
    rotated[:, 0] += aircraft.x
    rotated[:, 1] += aircraft.y
    
    triangle_patch = plt.Polygon(rotated, closed=True, facecolor=color, 
                                  edgecolor='black', linewidth=0.8, alpha=alpha, zorder=5)
    ax.add_patch(triangle_patch)
    
    # 朝向箭头
    arrow_len = size * 1.5
    ax.annotate('', xy=(aircraft.x + arrow_len * cos_h, aircraft.y + arrow_len * sin_h),
                xytext=(aircraft.x, aircraft.y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=alpha),
                zorder=6)
    
    # 标签
    if label:
        ax.annotate(label, (aircraft.x, aircraft.y), textcoords="offset points",
                    xytext=(8, 8), fontsize=7, color=color, alpha=alpha, fontweight='bold')


def render_episode(env, actions_list=None, policy_fn=None, save_path=None, 
                   fps=10, max_steps=None, figsize=(12, 10)):
    """
    渲染完整 episode 动画
    
    Args:
        env: FOVPenetrationEnv 实例
        actions_list: 预先计算的动作列表，shape (T, n_agents, 2)
        policy_fn: 策略函数 fn(obs) -> actions，如果 actions_list 为 None 则使用
        save_path: 保存路径（.gif 或 .mp4），None 则不保存
        fps: 帧率
        max_steps: 最大步数
        figsize: 图像尺寸
    
    Returns:
        frames: list of frame data
    """
    if max_steps is None:
        max_steps = env.config["max_steps"]
    
    obs, share_obs, _ = env.reset()
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    frames_data = []
    
    for step in range(max_steps):
        # 获取动作
        if actions_list is not None and step < len(actions_list):
            actions = actions_list[step]
        elif policy_fn is not None:
            actions = policy_fn(obs)
        else:
            # 随机动作
            actions = [env.action_space[i].sample() for i in range(env.n_agents)]
        
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        
        # 保存帧数据
        frames_data.append({
            'step': step,
            'attacker': (env.attacker.x, env.attacker.y, env.attacker.heading, env.attacker.alive),
            'escorts': [(e.x, e.y, e.heading, e.alive) for e in env.escorts],
            'interceptors': [(i.x, i.y, i.heading, i.alive) for i in env.interceptors],
        })
        
        if any(dones):
            break
    
    # 创建动画
    def update(frame_idx):
        render_frame(ax, env, step_num=frame_idx)
        return []
    
    # 重新运行以生成动画
    obs, share_obs, _ = env.reset()
    
    all_frames = []
    
    render_frame(ax, env, step_num=0)
    fig.tight_layout()
    
    if save_path:
        # 逐帧保存
        frame_images = []
        
        for step in range(max_steps):
            if actions_list is not None and step < len(actions_list):
                actions = actions_list[step]
            elif policy_fn is not None:
                actions = policy_fn(obs)
            else:
                actions = [env.action_space[i].sample() for i in range(env.n_agents)]
            
            obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
            
            render_frame(ax, env, step_num=step + 1)
            fig.canvas.draw()
            
            # 转为图像
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frame_images.append(image)
            
            if any(dones):
                break
        
        # 保存
        if save_path.endswith('.gif'):
            import imageio
            imageio.mimsave(save_path, frame_images, fps=fps)
            print(f"Animation saved to {save_path}")
        elif save_path.endswith('.mp4'):
            import imageio
            writer = imageio.get_writer(save_path, fps=fps)
            for img in frame_images:
                writer.append_data(img)
            writer.close()
            print(f"Video saved to {save_path}")
    
    plt.close(fig)
    return frames_data


def render_single_frame_to_file(env, save_path, step_num=0, figsize=(12, 10)):
    """保存单帧图像"""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    render_frame(ax, env, step_num=step_num)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Frame saved to {save_path}")
