"""
FOV Penetration Environment - Configuration V3
================================================
三维同构集群协同突防环境配置
"""

import numpy as np
import copy

G = 9.81

DEFAULT_CONFIG = {
    "map_size": 5000.0,
    "z_min": 100.0,
    "z_max": 2000.0,
    "dt": 0.1,
    "max_steps": 500,

    "n_offensive": 4,
    "n_defensive": 4,

    "offensive": {
        "v_min": 40.0,
        "v_nominal": 50.0,
        "v_max": 65.0,
        "nx_min": -0.5,
        "nx_max": 1.5,
        "ny_min": -4.0,
        "ny_max": 4.0,
        "nz_min": -2.0,
        "nz_max": 2.0,
        "dnx_max": 5.0,
        "dny_max": 8.0,
        "dnz_max": 5.0,
        "gamma_min": np.deg2rad(-30.0),
        "gamma_max": np.deg2rad(30.0),
    },

    "defensive": {
        "v_min": 70.0,
        "v_nominal": 90.0,
        "v_max": 120.0,
        "nx_min": -1.0,
        "nx_max": 3.0,
        "ny_min": -8.0,
        "ny_max": 8.0,
        "nz_min": -3.0,
        "nz_max": 3.0,
        "dnx_max": 8.0,
        "dny_max": 15.0,
        "dnz_max": 8.0,
        "gamma_min": np.deg2rad(-45.0),
        "gamma_max": np.deg2rad(45.0),
    },

    "hvt_position": [3000.0, 0.0, 0.0],

    "offensive_init": {
        "center_x": -3000.0,
        "center_y": 0.0,
        "center_z": 500.0,
        "spread_xy": 300.0,
        "spread_z": 50.0,
        "heading_to_hvt": True,
        "heading_noise": 0.3,
        "gamma_noise": 0.1,
        "pos_noise_xy": 200.0,
        "pos_noise_z": 30.0,
    },
    "defensive_init": {
        "center_x": 2000.0,
        "center_y": 0.0,
        "center_z": 600.0,
        "spread_xy": 400.0,
        "spread_z": 80.0,
        "heading_to_offense": True,
        "heading_noise": 0.3,
        "gamma_noise": 0.1,
        "pos_noise_xy": 200.0,
        "pos_noise_z": 50.0,
    },

    "fov_half_angle": np.deg2rad(30.0),
    "detection_range": 2000.0,

    "kill_range": 3.0,
    "collision_range": 5.0,

    "pn_nav_gain": 3,
    "pn_direct_freq": 20.0,
    "pn_guide_freq": 2.0,

    "assignment": {
        "method": "hungarian",
        "threat_weight": 0.6,
        "intercept_cost_weight": 0.4,
        "reassign": False,
        "reassign_interval": 50,
    },

    "reward": {
        # ===== V10: 前方牺牲, 后方突防 =====
        "hit_hvt_bonus": 1000.0,           # 命中HVT: 超级奖励 (全队共享)
        "approach_hvt_coef": 500.0,         # 个人接近HVT: 绝对主导
        "closest_bonus_coef": 200.0,        # 最近突防者额外奖金 (只给1架)
        "progress_coef": 0.3,              # 进度奖励: 靠近就给持续正奖
        "retreat_penalty": -0.15,          # 后退惩罚: 远离HVT扣分
        "mutual_kill_team_bonus": 80.0,    # 同归于尽全队共享 (存活队友分)
        "detected_penalty": -0.01,         # 被探测: 极轻
        "killed_penalty": -10.0,           # 被击杀: 轻 (一次性)
        "step_penalty": -0.01,             # 步惩罚: 极轻
        "timeout_penalty": -50.0,          # 超时惩罚
        "smooth_action_coef": -0.003,      # 动作平滑
        "altitude_penalty_coef": 8.0,      # 低高度保护
        "high_alt_penalty_coef": 3.0,      # 高高度惩罚
        "spread_bonus_coef": 0.01,         # 分散阵型 (轻微)
    },

    "cost": {
        "fov_exposure": 1.0,
        "danger_zone": 3.0,
        "collision": 5.0,
        "boundary": 2.0,
        "ground_crash": 10.0,
        "speed_violation": 0.5,
    },

    "two_stage_eval": {
        "enabled": True,
        "stage1_weight": 1.0,
        "stage2_weight": 0.3,
    },

    "obs_range": 5000.0,
    "vel_range": 120.0,
    "z_range": 2000.0,
}

SCENARIO_CONFIGS = {
    "scenario_1": {"n_offensive": 4, "n_defensive": 4},
    "scenario_2": {"n_offensive": 4, "n_defensive": 6},
    "scenario_3": {"n_offensive": 6, "n_defensive": 4},
}


def get_config(custom_config=None, scenario=None):
    config = copy.deepcopy(DEFAULT_CONFIG)
    if scenario is not None:
        if scenario not in SCENARIO_CONFIGS:
            raise ValueError(f"Unknown scenario: {scenario}")
        for key, value in SCENARIO_CONFIGS[scenario].items():
            config[key] = value
    if custom_config is not None:
        for key, value in custom_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value
    return config
