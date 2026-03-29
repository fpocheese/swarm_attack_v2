"""
FOV Penetration Environment - 场景预设
========================================
可以定义不同的预设场景配置
"""

from .config import DEFAULT_CONFIG


def get_scenario(name="default"):
    """
    获取预设场景配置
    
    Args:
        name: 场景名称
    
    Returns:
        config: 配置字典
    """
    if name == "default":
        return DEFAULT_CONFIG.copy()
    
    elif name == "easy":
        # 简单场景：拦截器少，FOV 窄
        config = DEFAULT_CONFIG.copy()
        config["n_interceptors"] = 2
        config["fov_half_angle"] = 0.35  # ~20 degrees
        config["detection_range"] = 1500.0
        config["max_steps"] = 300
        return config
    
    elif name == "hard":
        # 困难场景：拦截器多，FOV 宽
        config = DEFAULT_CONFIG.copy()
        config["n_interceptors"] = 6
        config["fov_half_angle"] = 0.7  # ~40 degrees
        config["detection_range"] = 2500.0
        config["max_steps"] = 500
        return config
    
    elif name == "close_range":
        # 近距交战
        config = DEFAULT_CONFIG.copy()
        config["map_size"] = 3000.0
        config["hvt_position"] = [2000.0, 0.0]
        config["attacker_init"]["x_range"] = [-2000.0, -1500.0]
        config["interceptor_init"]["x_range"] = [800.0, 1500.0]
        config["max_steps"] = 200
        return config
    
    else:
        raise ValueError(f"Unknown scenario: {name}")
