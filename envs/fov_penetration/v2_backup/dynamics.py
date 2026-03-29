"""
FOV Penetration Environment - 动力学模型 V2
=============================================
二维平面固定翼无人机质点运动学模型 (含过载变化率限制)

状态量: [x, y, v, heading]
控制量: [nx_cmd, ny_cmd] (轴向过载指令, 侧向过载指令)

运动方程 (参考 航空学报 2022, JGCD 2023):
  v_dot       = g * nx
  heading_dot = g * ny / max(v, v_min)
  x_dot       = v * cos(heading)
  y_dot       = v * sin(heading)

防抖头机制 — 过载变化率限制:
  |nx(t) - nx(t-1)| / dt <= dnx_max
  |ny(t) - ny(t-1)| / dt <= dny_max

  这确保过载指令不会突变, 产生物理上合理的平滑机动轨迹.
  参考: 吕强等(2022)取 |dn/dt|<=10g/s; 本文取 dnx_max=5g/s, dny_max=8g/s

积分方法: 四阶 Runge-Kutta (RK4) 替代欧拉积分, 提高数值精度.
"""

import numpy as np
from .config import G


def rate_limit_overload(nx_cmd, ny_cmd, nx_prev, ny_prev, dt, params):
    """
    过载变化率限制 — 防止指令跳变导致的抖头现象

    将过载指令的变化率限制在 [-dn_max, dn_max] 范围内,
    使得实际过载平滑过渡而非瞬间跳变.

    Args:
        nx_cmd, ny_cmd: 期望过载指令 (g)
        nx_prev, ny_prev: 上一步实际过载 (g)
        dt: 时间步长 (s)
        params: 平台参数, 需含 dnx_max, dny_max

    Returns:
        nx_limited, ny_limited: 变化率受限后的过载指令
    """
    dnx_max = params.get("dnx_max", 999.0)
    dny_max = params.get("dny_max", 999.0)

    # 计算最大允许的过载变化量
    max_dnx = dnx_max * dt
    max_dny = dny_max * dt

    # 限制变化量
    dnx = np.clip(nx_cmd - nx_prev, -max_dnx, max_dnx)
    dny = np.clip(ny_cmd - ny_prev, -max_dny, max_dny)

    nx_limited = nx_prev + dnx
    ny_limited = ny_prev + dny

    return nx_limited, ny_limited


def step_dynamics(x, y, v, heading, nx_cmd, ny_cmd, dt, params,
                  nx_prev=None, ny_prev=None):
    """
    一步动力学更新 (RK4 积分 + 过载变化率限制)

    Args:
        x, y: 当前位置 (m)
        v: 当前速度 (m/s)
        heading: 当前航向角 (rad)
        nx_cmd: 轴向过载指令 (g)
        ny_cmd: 侧向过载指令 (g)
        dt: 时间步长 (s)
        params: 平台参数
        nx_prev, ny_prev: 上一步实际过载 (用于变化率限制)

    Returns:
        x_new, y_new, v_new, heading_new, nx_actual, ny_actual
    """
    v_min = params["v_min"]
    v_max = params["v_max"]
    nx_min = params["nx_min"]
    nx_max = params["nx_max"]
    ny_min = params["ny_min"]
    ny_max = params["ny_max"]

    # 1. 裁剪过载指令到允许范围
    nx_cmd = np.clip(nx_cmd, nx_min, nx_max)
    ny_cmd = np.clip(ny_cmd, ny_min, ny_max)

    # 2. 过载变化率限制 (防抖头)
    if nx_prev is not None and ny_prev is not None:
        nx_actual, ny_actual = rate_limit_overload(
            nx_cmd, ny_cmd, nx_prev, ny_prev, dt, params)
    else:
        nx_actual, ny_actual = nx_cmd, ny_cmd

    # 再次裁剪到硬限制
    nx_actual = np.clip(nx_actual, nx_min, nx_max)
    ny_actual = np.clip(ny_actual, ny_min, ny_max)

    # 3. RK4 积分
    def derivatives(state, nx, ny):
        _v, _h = state
        _v = max(_v, v_min)
        v_dot = G * nx
        h_dot = G * ny / max(_v, v_min)
        return np.array([v_dot, h_dot])

    state = np.array([v, heading])

    k1 = derivatives(state, nx_actual, ny_actual)
    k2 = derivatives(state + 0.5 * dt * k1, nx_actual, ny_actual)
    k3 = derivatives(state + 0.5 * dt * k2, nx_actual, ny_actual)
    k4 = derivatives(state + dt * k3, nx_actual, ny_actual)

    state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    v_new = np.clip(state_new[0], v_min, v_max)
    heading_new = np.arctan2(np.sin(state_new[1]), np.cos(state_new[1]))

    # 位置用梯形积分 (更精确)
    v_avg = 0.5 * (v + v_new)
    h_avg = 0.5 * (heading + heading_new)
    x_new = x + v_avg * np.cos(h_avg) * dt
    y_new = y + v_avg * np.sin(h_avg) * dt

    return x_new, y_new, v_new, heading_new, nx_actual, ny_actual


def action_to_overload(action, params):
    """
    将归一化动作 [-1, 1] 映射到真实过载值

    Args:
        action: ndarray shape (2,), 归一化动作 [nx_norm, ny_norm]
        params: 平台参数

    Returns:
        nx_cmd, ny_cmd: 真实过载指令
    """
    nx_norm, ny_norm = action[0], action[1]

    nx_cmd = 0.5 * ((params["nx_max"] + params["nx_min"]) +
                     nx_norm * (params["nx_max"] - params["nx_min"]))
    ny_cmd = 0.5 * ((params["ny_max"] + params["ny_min"]) +
                     ny_norm * (params["ny_max"] - params["ny_min"]))

    return nx_cmd, ny_cmd
