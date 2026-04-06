"""
FOV Penetration Environment - 3D Dynamics V4
==============================================
三维固定翼无人机质点运动学模型 (轴向加速度 + 法向加速度大小 + 法向方向角)

状态量: [x, y, z, v, heading(psi), gamma]
控制量: [ax, ay, mu]
  - ax:  轴向加速度 (沿速度方向, m/s²)
  - ay:  法向加速度大小 (≥0, m/s²)
  - mu:  法向加速度方向角 (rad, 确定法向加速度在水平/垂直方向的分配)

运动方程:
  x_dot     = v * cos(gamma) * cos(psi)
  y_dot     = v * cos(gamma) * sin(psi)
  z_dot     = v * sin(gamma)
  v_dot     = ax - g * sin(gamma)
  psi_dot   = ay * cos(mu) / (v * cos(gamma))
  gamma_dot = (ay * sin(mu) - g * cos(gamma)) / v

积分方法: 四阶 Runge-Kutta (RK4)
防抖: 控制量变化率限制
"""

import numpy as np
from .config import G

# --- 数值保护常量 ---
_V_MIN_EPS = 1.0       # 速度下限保护 (m/s)
_COS_GAMMA_EPS = 0.01  # cos(gamma) 下限保护


def rate_limit_control(ax_cmd, ay_cmd, mu_cmd,
                       ax_prev, ay_prev, mu_prev,
                       dt, params):
    """控制量变化率限制 (ax, ay, mu)"""
    dax_max = params.get("dax_max", 999.0)
    day_max = params.get("day_max", 999.0)
    dmu_max = params.get("dmu_max", 999.0)

    max_dax = dax_max * dt
    max_day = day_max * dt
    max_dmu = dmu_max * dt

    ax_lim = ax_prev + np.clip(ax_cmd - ax_prev, -max_dax, max_dax)
    ay_lim = ay_prev + np.clip(ay_cmd - ay_prev, -max_day, max_day)
    # mu 角度差需要处理周期性
    dmu = np.arctan2(np.sin(mu_cmd - mu_prev), np.cos(mu_cmd - mu_prev))
    mu_lim = mu_prev + np.clip(dmu, -max_dmu, max_dmu)

    return ax_lim, ay_lim, mu_lim


def _derivatives_3d(state, ax, ay, mu, v_min):
    """
    三维运动学微分方程 (新控制输入形式)
    state = [v, heading(psi), gamma]
    control = [ax, ay, mu]
    返回 [v_dot, psi_dot, gamma_dot]
    """
    v, psi, gamma = state
    v_safe = max(v, max(v_min, _V_MIN_EPS))

    cos_gamma = np.cos(gamma)
    # 防止 cos(gamma) 过小导致除零
    if abs(cos_gamma) < _COS_GAMMA_EPS:
        cos_gamma = _COS_GAMMA_EPS * (1.0 if cos_gamma >= 0 else -1.0)

    v_dot = ax - G * np.sin(gamma)
    psi_dot = ay * np.cos(mu) / (v_safe * cos_gamma)
    gamma_dot = (ay * np.sin(mu) - G * np.cos(gamma)) / v_safe

    return np.array([v_dot, psi_dot, gamma_dot])


def step_dynamics_3d(x, y, z, v, heading, gamma,
                     ax_cmd, ay_cmd, mu_cmd,
                     dt, params,
                     ax_prev=None, ay_prev=None, mu_prev=None):
    """
    一步三维动力学更新 (RK4 积分 + 控制量变化率限制)

    控制输入:
        ax_cmd: 轴向加速度指令 (m/s²)
        ay_cmd: 法向加速度大小指令 (m/s², ≥0)
        mu_cmd: 法向加速度方向角指令 (rad)

    Returns:
        x, y, z, v, heading, gamma, ax_actual, ay_actual, mu_actual
    """
    v_min = params["v_min"]
    v_max = params["v_max"]
    ax_min = params["ax_min"]
    ax_max_val = params["ax_max"]
    ay_max_val = params["ay_max"]
    gamma_min = params.get("gamma_min", np.deg2rad(-45.0))
    gamma_max = params.get("gamma_max", np.deg2rad(45.0))

    # 裁剪控制指令
    ax_cmd = np.clip(ax_cmd, ax_min, ax_max_val)
    ay_cmd = np.clip(ay_cmd, 0.0, ay_max_val)
    # mu 归一化到 [-pi, pi]
    mu_cmd = np.arctan2(np.sin(mu_cmd), np.cos(mu_cmd))

    # 控制量变化率限制
    if ax_prev is not None and ay_prev is not None and mu_prev is not None:
        ax_actual, ay_actual, mu_actual = rate_limit_control(
            ax_cmd, ay_cmd, mu_cmd, ax_prev, ay_prev, mu_prev, dt, params)
    else:
        ax_actual, ay_actual, mu_actual = ax_cmd, ay_cmd, mu_cmd

    # 再次裁剪 (变化率限制后可能仍需保证合法)
    ax_actual = np.clip(ax_actual, ax_min, ax_max_val)
    ay_actual = np.clip(ay_actual, 0.0, ay_max_val)
    mu_actual = np.arctan2(np.sin(mu_actual), np.cos(mu_actual))

    # RK4 积分
    state = np.array([v, heading, gamma])

    k1 = _derivatives_3d(state, ax_actual, ay_actual, mu_actual, v_min)
    k2 = _derivatives_3d(state + 0.5 * dt * k1, ax_actual, ay_actual, mu_actual, v_min)
    k3 = _derivatives_3d(state + 0.5 * dt * k2, ax_actual, ay_actual, mu_actual, v_min)
    k4 = _derivatives_3d(state + dt * k3, ax_actual, ay_actual, mu_actual, v_min)

    state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    v_new = np.clip(state_new[0], v_min, v_max)
    heading_new = state_new[1]
    gamma_new = np.clip(state_new[2], gamma_min, gamma_max)

    # 角度归一化
    heading_new = np.arctan2(np.sin(heading_new), np.cos(heading_new))

    # 位置更新
    cos_g = np.cos(gamma_new)
    x_new = x + v_new * cos_g * np.cos(heading_new) * dt
    y_new = y + v_new * cos_g * np.sin(heading_new) * dt
    z_new = z + v_new * np.sin(gamma_new) * dt

    return x_new, y_new, z_new, v_new, heading_new, gamma_new, ax_actual, ay_actual, mu_actual


def action_to_control_3d(action, params):
    """
    将归一化动作 [-1, 1]^3 映射到实际控制指令 (ax, ay, mu)

    V28: 添加重力补偿偏置 → action=[0,0,0] 对应平飞(trim)
    
    action[0] -> ax:  偏置映射, 0 -> 0 (水平巡航时 v_dot ≈ 0)
    action[1] -> ay:  偏置映射, 0 -> G (重力补偿), -1 -> 0, +1 -> ay_max
    action[2] -> mu:  偏置映射, 0 -> pi/2 (法向力朝上), 范围 [-pi/2, 3pi/2]

    这样 action=[0,0,0] 时:
      ax=0, ay=G, mu=pi/2  
      → gamma_dot = (G*sin(pi/2) - G*cos(gamma))/v ≈ 0  (平飞)
      → psi_dot = G*cos(pi/2)/(v*cos(gamma)) = 0         (直飞)
    """
    a = np.array(action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)

    # 动作缩放: 让策略从小扰动开始学
    action_scale = params.get("action_scale", 1.0)
    a = a * action_scale

    ax_min = params["ax_min"]
    ax_max_val = params["ax_max"]
    ay_max_val = params["ay_max"]

    # ax: 偏置映射, action=0 → ax=0 (水平巡航)
    ax_center = 0.0
    if a[0] <= 0:
        ax = ax_center + a[0] * (ax_center - ax_min)
    else:
        ax = ax_center + a[0] * (ax_max_val - ax_center)

    # ay: 偏置映射, action=0 → ay=G (重力补偿), -1 → 0, +1 → ay_max
    ay_level = G  # 9.81 m/s², 刚好抵消重力
    if a[1] >= 0:
        ay = ay_level + a[1] * (ay_max_val - ay_level)
    else:
        ay = ay_level * (1.0 + a[1])  # a[1]=-1 → ay=0

    # mu: 偏置映射, action=0 → mu=pi/2 (法向力朝上=平飞)
    # 范围: a[2]=-1 → mu=-pi/2, a[2]=0 → mu=pi/2, a[2]=1 → mu=3pi/2
    mu = np.pi / 2.0 + a[2] * np.pi

    return ax, ay, mu


# === 向后兼容别名 ===
action_to_overload_3d = action_to_control_3d
