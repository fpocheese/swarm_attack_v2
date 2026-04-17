"""
FOV Penetration Environment - 3D Dynamics V5 (惯性系加速度)
=============================================================
三维固定翼无人机质点运动学模型 (轴向加速度 + 俯仰法向加速度 + 偏航法向加速度)

状态量: [x, y, z, v, heading(psi), gamma]
控制量: [ax, an_pitch, an_yaw]
  - ax:       轴向加速度 (沿速度方向, m/s²)
  - an_pitch: 俯仰平面法向加速度 (m/s², 正值=抬头, 含重力补偿)
  - an_yaw:   偏航平面法向加速度 (m/s², 正值=左转)

运动方程:
  x_dot     = v * cos(gamma) * cos(psi)
  y_dot     = v * cos(gamma) * sin(psi)
  z_dot     = v * sin(gamma)
  v_dot     = ax - g * sin(gamma)
  psi_dot   = an_yaw / (v * cos(gamma))
  gamma_dot = (an_pitch - g * cos(gamma)) / v

默认平飞(trim)条件: ax=0, an_pitch=g, an_yaw=0
  → v_dot = 0 (匀速)
  → psi_dot = 0 (直线)
  → gamma_dot = (g - g*cos(0))/v = 0 (平飞)

积分方法: 四阶 Runge-Kutta (RK4)
防抖: 控制量变化率限制
"""

import numpy as np
from .config import G

# --- 数值保护常量 ---
_V_MIN_EPS = 1.0       # 速度下限保护 (m/s)
_COS_GAMMA_EPS = 0.01  # cos(gamma) 下限保护


def rate_limit_control(ax_cmd, an_pitch_cmd, an_yaw_cmd,
                       ax_prev, an_pitch_prev, an_yaw_prev,
                       dt, params):
    """控制量变化率限制 (ax, an_pitch, an_yaw)"""
    dax_max = params.get("dax_max", 999.0)
    dan_pitch_max = params.get("dan_pitch_max", 999.0)
    dan_yaw_max = params.get("dan_yaw_max", 999.0)

    max_dax = dax_max * dt
    max_dan_pitch = dan_pitch_max * dt
    max_dan_yaw = dan_yaw_max * dt

    ax_lim = ax_prev + np.clip(ax_cmd - ax_prev, -max_dax, max_dax)
    an_pitch_lim = an_pitch_prev + np.clip(an_pitch_cmd - an_pitch_prev,
                                           -max_dan_pitch, max_dan_pitch)
    an_yaw_lim = an_yaw_prev + np.clip(an_yaw_cmd - an_yaw_prev,
                                       -max_dan_yaw, max_dan_yaw)

    return ax_lim, an_pitch_lim, an_yaw_lim


def _derivatives_3d(state, ax, an_pitch, an_yaw, v_min):
    """
    三维运动学微分方程 (惯性系加速度控制)
    state = [v, heading(psi), gamma]
    control = [ax, an_pitch, an_yaw]
    返回 [v_dot, psi_dot, gamma_dot]
    """
    v, psi, gamma = state
    v_safe = max(v, max(v_min, _V_MIN_EPS))

    cos_gamma = np.cos(gamma)
    # 防止 cos(gamma) 过小导致除零
    if abs(cos_gamma) < _COS_GAMMA_EPS:
        cos_gamma = _COS_GAMMA_EPS * (1.0 if cos_gamma >= 0 else -1.0)

    v_dot = ax - G * np.sin(gamma)
    psi_dot = an_yaw / (v_safe * cos_gamma)
    gamma_dot = (an_pitch - G * np.cos(gamma)) / v_safe

    return np.array([v_dot, psi_dot, gamma_dot])


def step_dynamics_3d(x, y, z, v, heading, gamma,
                     ax_cmd, an_pitch_cmd, an_yaw_cmd,
                     dt, params,
                     ax_prev=None, an_pitch_prev=None, an_yaw_prev=None):
    """
    一步三维动力学更新 (RK4 积分 + 控制量变化率限制)

    控制输入:
        ax_cmd:       轴向加速度指令 (m/s²)
        an_pitch_cmd: 俯仰平面法向加速度指令 (m/s²)
        an_yaw_cmd:   偏航平面法向加速度指令 (m/s²)

    Returns:
        x, y, z, v, heading, gamma, ax_actual, an_pitch_actual, an_yaw_actual
    """
    v_min = params["v_min"]
    v_max = params["v_max"]
    ax_min = params["ax_min"]
    ax_max_val = params["ax_max"]
    an_pitch_max = params["an_pitch_max"]
    an_yaw_max = params["an_yaw_max"]
    gamma_min = params.get("gamma_min", np.deg2rad(-45.0))
    gamma_max = params.get("gamma_max", np.deg2rad(45.0))

    # 裁剪控制指令
    ax_cmd = np.clip(ax_cmd, ax_min, ax_max_val)
    an_pitch_cmd = np.clip(an_pitch_cmd, -an_pitch_max, an_pitch_max)
    an_yaw_cmd = np.clip(an_yaw_cmd, -an_yaw_max, an_yaw_max)

    # 控制量变化率限制
    if ax_prev is not None and an_pitch_prev is not None and an_yaw_prev is not None:
        ax_actual, an_pitch_actual, an_yaw_actual = rate_limit_control(
            ax_cmd, an_pitch_cmd, an_yaw_cmd,
            ax_prev, an_pitch_prev, an_yaw_prev, dt, params)
    else:
        ax_actual, an_pitch_actual, an_yaw_actual = ax_cmd, an_pitch_cmd, an_yaw_cmd

    # 再次裁剪 (变化率限制后可能仍需保证合法)
    ax_actual = np.clip(ax_actual, ax_min, ax_max_val)
    an_pitch_actual = np.clip(an_pitch_actual, -an_pitch_max, an_pitch_max)
    an_yaw_actual = np.clip(an_yaw_actual, -an_yaw_max, an_yaw_max)

    # RK4 积分
    state = np.array([v, heading, gamma])

    k1 = _derivatives_3d(state, ax_actual, an_pitch_actual, an_yaw_actual, v_min)
    k2 = _derivatives_3d(state + 0.5 * dt * k1, ax_actual, an_pitch_actual, an_yaw_actual, v_min)
    k3 = _derivatives_3d(state + 0.5 * dt * k2, ax_actual, an_pitch_actual, an_yaw_actual, v_min)
    k4 = _derivatives_3d(state + dt * k3, ax_actual, an_pitch_actual, an_yaw_actual, v_min)

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

    return x_new, y_new, z_new, v_new, heading_new, gamma_new, ax_actual, an_pitch_actual, an_yaw_actual


def action_to_control_3d(action, params):
    """
    将归一化动作 [-1, 1]^3 映射到实际控制指令 (ax, an_pitch, an_yaw)

    V5 惯性系加速度 → action=[0,0,0] 对应平飞(trim)
    
    action[0] -> ax:       偏置映射, 0 -> 0 (水平巡航时 v_dot ≈ 0)
    action[1] -> an_pitch: 偏置映射, 0 -> G (重力补偿)
                           +1 -> an_pitch_max, -1 -> -an_pitch_max
    action[2] -> an_yaw:   线性映射, 0 -> 0 (直飞)
                           +1 -> an_yaw_max, -1 -> -an_yaw_max

    这样 action=[0,0,0] 时:
      ax=0, an_pitch=G, an_yaw=0  
      → gamma_dot = (G - G*cos(gamma))/v ≈ 0  (平飞)
      → psi_dot = 0/(v*cos(gamma)) = 0         (直飞)
      → v_dot = 0 - g*sin(0) = 0               (匀速)
    """
    a = np.array(action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)

    # 动作缩放: 让策略从小扰动开始学
    action_scale = params.get("action_scale", 1.0)
    a = a * action_scale

    ax_min = params["ax_min"]
    ax_max_val = params["ax_max"]
    an_pitch_max = params["an_pitch_max"]
    an_yaw_max = params["an_yaw_max"]

    # ax: 偏置映射, action=0 → ax=0 (水平巡航)
    ax_center = 0.0
    if a[0] <= 0:
        ax = ax_center + a[0] * (ax_center - ax_min)
    else:
        ax = ax_center + a[0] * (ax_max_val - ax_center)

    # an_pitch: 偏置映射, action=0 → an_pitch=G (重力补偿=平飞)
    # action=+1 → an_pitch_max (最大抬头)
    # action=-1 → -an_pitch_max (最大俯冲)
    an_pitch_trim = G  # 9.81 m/s², 刚好抵消重力
    if a[1] >= 0:
        an_pitch = an_pitch_trim + a[1] * (an_pitch_max - an_pitch_trim)
    else:
        an_pitch = an_pitch_trim + a[1] * (an_pitch_max + an_pitch_trim)
        # a[1]=-1 → G + (-1)*(an_pitch_max + G) = -an_pitch_max

    # an_yaw: 对称线性映射, action=0 → 0 (直飞)
    an_yaw = a[2] * an_yaw_max

    return ax, an_pitch, an_yaw


# === 向后兼容别名 ===
action_to_overload_3d = action_to_control_3d
