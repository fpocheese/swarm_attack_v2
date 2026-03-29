"""
FOV Penetration Environment - 3D Dynamics V3
==============================================
三维固定翼无人机质点运动学模型 (含过载变化率限制)

状态量: [x, y, z, v, heading, gamma]
控制量: [nx_cmd, ny_cmd, nz_cmd]

运动方程:
  v_dot     = g * nx
  psi_dot   = g * ny / (v * cos(gamma))
  gamma_dot = g * (nz - cos(gamma)) / v
  x_dot     = v * cos(gamma) * cos(psi)
  y_dot     = v * cos(gamma) * sin(psi)
  z_dot     = v * sin(gamma)

积分方法: 四阶 Runge-Kutta (RK4)
防抖头: 过载变化率限制 |dn/dt| <= dn_max
"""

import numpy as np
from .config import G


def rate_limit_overload(nx_cmd, ny_cmd, nz_cmd,
                        nx_prev, ny_prev, nz_prev,
                        dt, params):
    """三维过载变化率限制"""
    dnx_max = params.get("dnx_max", 999.0)
    dny_max = params.get("dny_max", 999.0)
    dnz_max = params.get("dnz_max", 999.0)

    max_dnx = dnx_max * dt
    max_dny = dny_max * dt
    max_dnz = dnz_max * dt

    nx_lim = nx_prev + np.clip(nx_cmd - nx_prev, -max_dnx, max_dnx)
    ny_lim = ny_prev + np.clip(ny_cmd - ny_prev, -max_dny, max_dny)
    nz_lim = nz_prev + np.clip(nz_cmd - nz_prev, -max_dnz, max_dnz)

    return nx_lim, ny_lim, nz_lim


def _derivatives_3d(state, nx, ny, nz, v_min):
    """
    三维运动学微分方程
    state = [v, heading, gamma]
    返回 [v_dot, heading_dot, gamma_dot]
    """
    v, psi, gamma = state
    v = max(v, v_min)
    cos_gamma = np.cos(gamma)
    if abs(cos_gamma) < 0.01:
        cos_gamma = 0.01 * np.sign(cos_gamma) if cos_gamma != 0 else 0.01

    v_dot = G * nx
    psi_dot = G * ny / (v * cos_gamma)
    gamma_dot = G * (nz - np.cos(gamma)) / v

    return np.array([v_dot, psi_dot, gamma_dot])


def step_dynamics_3d(x, y, z, v, heading, gamma,
                     nx_cmd, ny_cmd, nz_cmd,
                     dt, params,
                     nx_prev=None, ny_prev=None, nz_prev=None):
    """
    一步三维动力学更新 (RK4 积分 + 过载变化率限制)

    Returns:
        x, y, z, v, heading, gamma, nx_actual, ny_actual, nz_actual
    """
    v_min = params["v_min"]
    v_max = params["v_max"]
    nx_min = params["nx_min"]
    nx_max = params["nx_max"]
    ny_min = params["ny_min"]
    ny_max = params["ny_max"]
    nz_min = params.get("nz_min", -3.0)
    nz_max_val = params.get("nz_max", 3.0)
    gamma_min = params.get("gamma_min", np.deg2rad(-45.0))
    gamma_max = params.get("gamma_max", np.deg2rad(45.0))

    # 裁剪过载指令
    nx_cmd = np.clip(nx_cmd, nx_min, nx_max)
    ny_cmd = np.clip(ny_cmd, ny_min, ny_max)
    nz_cmd = np.clip(nz_cmd, nz_min, nz_max_val)

    # 过载变化率限制
    if nx_prev is not None and ny_prev is not None and nz_prev is not None:
        nx_actual, ny_actual, nz_actual = rate_limit_overload(
            nx_cmd, ny_cmd, nz_cmd, nx_prev, ny_prev, nz_prev, dt, params)
    else:
        nx_actual, ny_actual, nz_actual = nx_cmd, ny_cmd, nz_cmd

    nx_actual = np.clip(nx_actual, nx_min, nx_max)
    ny_actual = np.clip(ny_actual, ny_min, ny_max)
    nz_actual = np.clip(nz_actual, nz_min, nz_max_val)

    # RK4
    state = np.array([v, heading, gamma])

    k1 = _derivatives_3d(state, nx_actual, ny_actual, nz_actual, v_min)
    k2 = _derivatives_3d(state + 0.5 * dt * k1, nx_actual, ny_actual, nz_actual, v_min)
    k3 = _derivatives_3d(state + 0.5 * dt * k2, nx_actual, ny_actual, nz_actual, v_min)
    k4 = _derivatives_3d(state + dt * k3, nx_actual, ny_actual, nz_actual, v_min)

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

    return x_new, y_new, z_new, v_new, heading_new, gamma_new, nx_actual, ny_actual, nz_actual


def action_to_overload_3d(action, params):
    """
    将归一化动作 [-1, 1]^3 映射到实际过载指令

    action[0] -> nx: [-1,1] -> [nx_min, nx_max]  (线性映射, 0 -> 中值)
    action[1] -> ny: [-1,1] -> [ny_min, ny_max]  (线性映射, 0 -> 0)
    action[2] -> nz: [-1,1] -> [nz_min, nz_max]  (偏置映射, 0 -> 1.0 即平飞)

    V11c关键改进: 加入action_scale参数, 控制动作幅度
    scale=0.3意味着最大只用到30%的过载范围
    这让策略从"接近直飞"开始微调, 而不是剧烈机动
    """
    a = np.array(action, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)

    # 动作缩放: 让策略从小扰动开始学
    action_scale = params.get("action_scale", 0.3)
    a = a * action_scale

    nx_min = params["nx_min"]
    nx_max = params["nx_max"]
    ny_min = params["ny_min"]
    ny_max = params["ny_max"]
    nz_min = params.get("nz_min", -0.5)
    nz_max_val = params.get("nz_max", 1.5)

    # nx: 偏置映射, action=0 -> nx=0.5 (轻微加速/巡航)
    nx_center = 0.5
    if a[0] <= 0:
        nx = nx_center + a[0] * (nx_center - nx_min)
    else:
        nx = nx_center + a[0] * (nx_max - nx_center)

    # ny: 标准线性映射 (0 -> 0 = 直飞)
    ny = ny_min + (a[1] + 1.0) * 0.5 * (ny_max - ny_min)

    # nz: 偏置映射, action=0 -> nz=1.0(平飞)
    nz_level = 1.0
    if a[2] <= 0:
        nz = nz_level + a[2] * (nz_level - nz_min)
    else:
        nz = nz_level + a[2] * (nz_max_val - nz_level)

    return nx, ny, nz
