"""
FOV Penetration Environment - Render V3
=========================================
3D视图渲染
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render_frame(ax, env, step_num=0, show_fov=True):
    ax.clear()
    cfg = env.config
    map_size = cfg["map_size"]
    margin = 500
    
    ax.set_xlim(-map_size - margin, map_size + margin)
    ax.set_ylim(-map_size - margin, map_size + margin)
    # 假设高度默认0到3000
    z_max = 3000
    ax.set_zlim(0, z_max)
    
    # 试图设置比例
    try:
        ax.set_box_aspect((1, 1, z_max/(2*map_size + 2*margin)))
    except AttributeError:
        pass # 对于不支持set_box_aspect的旧版本

    ax.set_facecolor('#f0f0f0')
    ax.set_title(f'FOV Penetration V3 3D - Step {step_num}', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 底面边界线
    bx = [-map_size, map_size, map_size, -map_size, -map_size]
    by = [-map_size, -map_size, map_size, map_size, -map_size]
    bz = [0, 0, 0, 0, 0]
    ax.plot3D(bx, by, bz, color='gray', linestyle='--', linewidth=1)

    # HVT
    hvt = env.hvt
    ax.scatter3D(hvt.x, hvt.y, hvt.z, marker='*', s=200, color='gold',
                 edgecolor='black', linewidth=1.5, zorder=10)
    ax.text3D(hvt.x, hvt.y, hvt.z + 50, 'HVT', fontsize=10, fontweight='bold')

    kr = cfg.get("kill_range", 3.0)
    # 画一个水平的圆圈表示击杀范围在所在高度
    theta = np.linspace(0, 2*np.pi, 50)
    cx = hvt.x + kr * 50 * np.cos(theta)
    cy = hvt.y + kr * 50 * np.sin(theta)
    cz = np.ones_like(theta) * hvt.z
    ax.plot3D(cx, cy, cz, color='orange', linewidth=1, linestyle='-', alpha=0.5)

    no_target_alive_defs = []

    # 防御方
    for i, d in enumerate(env.defensives):
        # 用户需求: 判定死亡后飞机从动图中消失
        if not d.alive:
            continue

        color = 'red'
        alpha_val = 1.0
        
        # 3D视野：真实三维圆锥示意
        if show_fov and d.alive:
            fov_half = cfg["fov_half_angle"]
            det_range = cfg["detection_range"]
            heading = d.heading
            gamma = d.gamma
            
            # 圆锥的顶点和中轴向量
            vx = np.cos(gamma) * np.cos(heading)
            vy = np.cos(gamma) * np.sin(heading)
            vz = np.sin(gamma)
            
            # 中轴线终点
            D = det_range * np.cos(fov_half)
            R = det_range * np.sin(fov_half)
            cx_h = d.x + vx * D
            cy_h = d.y + vy * D
            cz_h = d.z + vz * D
            
            # 绘制中轴线
            ax.plot3D([d.x, d.x + vx*det_range], 
                      [d.y, d.y + vy*det_range], 
                      [d.z, d.z + vz*det_range], 
                      color='red', alpha=0.2, linestyle=':')
            
            # 计算垂直于 V 的两个正交基向量 (u, w) 来生成圆
            # 找一个与 v 不平行的向量，做叉乘
            v_vec = np.array([vx, vy, vz])
            if abs(v_vec[2]) < 0.99:
                temp = np.array([0, 0, 1])
            else:
                temp = np.array([1, 0, 0])
            u_vec = np.cross(v_vec, temp)
            u_vec = u_vec / np.linalg.norm(u_vec)
            w_vec = np.cross(v_vec, u_vec)
            
            # 生成圆上的点
            theta_cone = np.linspace(0, 2*np.pi, 20)
            circle_x = cx_h + R * (np.cos(theta_cone)*u_vec[0] + np.sin(theta_cone)*w_vec[0])
            circle_y = cy_h + R * (np.cos(theta_cone)*u_vec[1] + np.sin(theta_cone)*w_vec[1])
            circle_z = cz_h + R * (np.cos(theta_cone)*u_vec[2] + np.sin(theta_cone)*w_vec[2])
            
            # 绘制圆锥截面圆
            ax.plot3D(circle_x, circle_y, circle_z, color='red', alpha=0.3, linestyle='-')
            
            # 绘制几条母线连起顶点和圆
            for j in range(0, 20, 4):
                ax.plot3D([d.x, circle_x[j]], [d.y, circle_y[j]], [d.z, circle_z[j]], 
                          color='red', alpha=0.2, linestyle='--')
            
        _draw_aircraft_3d(ax, d, color, alpha_val, label=f'D{i}')
        if len(d.trajectory) > 1:
            traj = np.array(d.trajectory)
            ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color='red', alpha=0.3, linewidth=1.0)

        # 目标可视化: 画出拦截器当前打击目标连线(若有)
        if hasattr(env, 'defensive_policies') and i < len(env.defensive_policies):
            policy = env.defensive_policies[i]
            tgt = getattr(policy, 'target', None)
            if tgt is not None and getattr(tgt, 'alive', False) and not getattr(tgt, 'hit_hvt', False):
                ax.plot3D([d.x, tgt.x], [d.y, tgt.y], [d.z, tgt.z],
                          color='yellow', alpha=0.35, linewidth=1.2, linestyle='-')
            else:
                no_target_alive_defs.append(i)

    # 进攻方
    for i, off in enumerate(env.offensives):
        # 用户需求: 判定死亡后飞机从动图中消失
        if (not off.alive) and (not getattr(off, 'hit_hvt', False)):
            continue

        if hasattr(off, 'hit_hvt') and off.hit_hvt:
            color = 'lime'
        elif off.alive:
            color = 'blue'
        else:
            color = 'gray'
        alpha_val = 1.0 if off.alive else 0.6
        
        _draw_aircraft_3d(ax, off, color, alpha_val, label=f'O{i}',
                       marker_size=12 if off.alive else 8)
        if len(off.trajectory) > 1:
            traj = np.array(off.trajectory)
            ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2], color='blue', alpha=0.4, linewidth=1.5)

    legend_elements = [
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=10, label='Offensive'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10, label='Defensive'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', markersize=12, label='HVT'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    off_alive = sum(1 for o in env.offensives if o.alive)
    n_det = sum(1 for o in env.offensives if o.alive and o.detected)
    info_text = f'Off alive: {off_alive}/{env.n_offensive}\n'
    info_text += f'Detected: {n_det}\nHits: {env.hit_count}'
    if no_target_alive_defs:
        info_text += f"\nNoTarget Def: {','.join([f'D{i}' for i in no_target_alive_defs])}"
    ax.text2D(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _draw_aircraft_3d(ax, aircraft, color, alpha=1.0, label='', marker_size=10):
    if not aircraft.alive and not getattr(aircraft, 'hit_hvt', False):
        return
    
    # 用简单的散点或线表示3D朝向的无人机
    # 为了简化，画一个带延长线指向机头方向的点
    ax.scatter3D(aircraft.x, aircraft.y, aircraft.z, marker='o', color=color, s=marker_size*2, alpha=alpha)
    
    heading = aircraft.heading
    gamma = aircraft.gamma
    l = marker_size * 20
    
    dx = l * np.cos(gamma) * np.cos(heading)
    dy = l * np.cos(gamma) * np.sin(heading)
    dz = l * np.sin(gamma)
    
    ax.plot3D([aircraft.x, aircraft.x + dx], 
              [aircraft.y, aircraft.y + dy], 
              [aircraft.z, aircraft.z + dz], 
              color=color, linewidth=2, alpha=alpha)

    if label:
        ax.text3D(aircraft.x, aircraft.y, aircraft.z + 30, label, fontsize=7, color=color, fontweight='bold')

def render_episode(env, save_path=None, fps=10, max_steps=None, figsize=(10, 8)):
    if max_steps is None:
        max_steps = env.config["max_steps"]
    obs, share_obs, _ = env.reset()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    frame_images = []

    for step_i in range(max_steps):
        actions = [env.action_space[i].sample() for i in range(env.n_agents)]
        obs, share_obs, rewards, costs, dones, infos, _ = env.step(actions)
        render_frame(ax, env, step_num=step_i + 1)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame_images.append(image)
        if any(dones):
            break

    if save_path and frame_images:
        if save_path.endswith('.gif'):
            import imageio
            imageio.mimsave(save_path, frame_images, fps=fps)
            print(f"Saved to {save_path}")

    plt.close(fig)
    return frame_images
