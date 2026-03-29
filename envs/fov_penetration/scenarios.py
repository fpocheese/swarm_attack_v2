"""
FOV Penetration Environment - Scenarios V3
============================================
三工况配置
"""

from .config import get_config, SCENARIO_CONFIGS


def get_scenario(name="scenario_1"):
    if name in SCENARIO_CONFIGS:
        return get_config(scenario=name)
    alias = {
        "default": "scenario_1",
        "balanced": "scenario_1",
        "defense_advantage": "scenario_2",
        "offense_advantage": "scenario_3",
    }
    if name in alias:
        return get_config(scenario=alias[name])
    raise ValueError(f"Unknown scenario: {name}")


def list_scenarios():
    return {
        "scenario_1": "4v4 balanced",
        "scenario_2": "4v6 defense advantage",
        "scenario_3": "6v4 offense advantage",
    }
