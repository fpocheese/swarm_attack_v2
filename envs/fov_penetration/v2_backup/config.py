"""
FOV Penetration Environment - Configuration V2
==============================================
基于敌方视场角限制的固定翼无人机集群协同突防

动力学模型参考:
  [1] 吕强等, "基于深度强化学习的无人机空战机动决策", 航空学报, 2022
  [2] Pope et al., "Hierarchical reinforcement learning for air combat", JGCD, 2023
  [3] 杨秀霞等, "基于MADDPG的多无人机协同对抗", 系统工程与电子技术, 2023

过载变化率限制防止指令跳变(抖头):
  参考文献[1]中取过载变化率限制 |dn/dt| <= 10g/s (典型固定翼UAV)
  本文取轴向 |dnx/dt| <= 5g/s, 侧向 |dny/dt| <= 8g/s

比例导引法拦截器:
  N=3, 双频率信息更新 (直接探测20Hz / 目指信息2Hz)
"""

import numpy as np

G = 9.81

DEFAULT_CONFIG = {
    # ======== 场景参数 ========
    "map_size": 5000.0,
    "dt": 0.1,                   # 仿真步长 (s), 10Hz
    "max_steps": 500,            # 最大步数 (50s)

    # ======== 对象数量 ========
    "n_attackers": 1,
    "n_escorts": 3,
    "n_interceptors": 4,

    # ======== attacker 参数 ========
    "attacker": {
        "v_min": 35.0,
        "v_nominal": 45.0,
        "v_max": 60.0,
        "nx_min": -0.5,
        "nx_max": 1.5,
        "ny_min": -4.0,
        "ny_max": 4.0,
        "dnx_max": 5.0,          # 轴向过载变化率上限 (g/s) — 防抖头
        "dny_max": 8.0,          # 侧向过载变化率上限 (g/s) — 防抖头
    },

    # ======== escort 参数 ========
    "escort": {
        "v_min": 40.0,
        "v_nominal": 50.0,
        "v_max": 65.0,
        "nx_min": -0.5,
        "nx_max": 1.5,
        "ny_min": -4.0,
        "ny_max": 4.0,
        "dnx_max": 5.0,
        "dny_max": 8.0,
    },

    # ======== interceptor 参数 ========
    "interceptor": {
        "v_min": 70.0,
        "v_nominal": 90.0,
        "v_max": 120.0,
        "nx_min": -1.0,
        "nx_max": 3.0,
        "ny_min": -8.0,
        "ny_max": 8.0,
        "dnx_max": 8.0,
        "dny_max": 15.0,
    },

    # ======== HVT ========
    "hvt_position": [3000.0, 0.0],

    # ======== 初始位置 ========
    "attacker_init": {
        "x_range": [-3500.0, -2500.0],
        "y_range": [-500.0, 500.0],
        "heading_range": [-0.3, 0.3],
    },
    "escort_init": {
        "x_range": [-3500.0, -2500.0],
        "y_range": [-1000.0, 1000.0],
        "heading_range": [-0.5, 0.5],
    },
    "interceptor_init": {
        "x_range": [1500.0, 2500.0],
        "y_range": [-1500.0, 1500.0],
        "heading_range": [2.5, 3.8],
    },

    # ======== FOV 和探测参数 ========
    "fov_half_angle": np.deg2rad(30.0),
    "detection_range": 2000.0,

    # ======== 击杀/命中参数 (统一3m) ========
    "kill_range": 3.0,           # 通用击杀距离: 拦截打进攻、护卫打拦截、进攻打HVT

    # ======== 比例导引法参数 ========
    "pn_nav_gain": 3,            # 比例导引系数 N
    "pn_direct_freq": 20.0,      # 直接探测更新频率 (Hz)
    "pn_guide_freq": 2.0,        # 目指信息更新频率 (Hz)

    # ======== 碰撞 ========
    "collision_range": 3.0,

    # ======== 奖励权重 ========
    "reward": {
        # 进攻飞行器
        "approach_hvt_coef": 2.0,
        "hit_hvt_bonus": 200.0,
        "attacker_killed_penalty": -100.0,

        # FOV 规避
        "fov_evasion_coef": 0.5,
        "in_fov_penalty": -1.0,
        "multi_fov_penalty_coef": -2.0,

        # 护卫飞行器
        "escort_kill_interceptor_bonus": 50.0,
        "escort_divert_coef": 0.3,
        "escort_approach_intc_coef": 0.3,
        "escort_killed_penalty": -10.0,

        # 全局协同
        "team_spread_coef": 0.05,
        "survival_coef": 0.1,
        "step_penalty": -0.05,
        "timeout_penalty": -30.0,

        # 过载平滑
        "smooth_action_coef": -0.05,
    },

    # ======== 成本(约束)权重 ========
    "cost": {
        "fov_exposure": 1.0,
        "danger_zone": 3.0,
        "escort_danger": 0.5,
        "collision": 5.0,
        "boundary": 2.0,
        "speed_violation": 0.5,
    },

    # ======== 归一化参数 ========
    "obs_range": 5000.0,
    "vel_range": 120.0,
}


def get_config(custom_config=None):
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    if custom_config is not None:
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    return config
