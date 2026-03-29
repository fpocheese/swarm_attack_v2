"""
FOV Penetration Environment - 实体定义 V2
==========================================
支持过载变化率限制的飞行器实体
"""

import numpy as np
from .dynamics import step_dynamics, action_to_overload


class Aircraft:
    """固定翼无人机 (含过载变化率限制状态)"""

    def __init__(self, uid, role, params, x=0.0, y=0.0, v=None, heading=0.0):
        self.uid = uid
        self.role = role
        self.params = params

        self.x = x
        self.y = y
        self.v = v if v is not None else params["v_nominal"]
        self.heading = heading
        self.nx = 0.0
        self.ny = 0.0
        self.alive = True

        # 轨迹记录
        self.trajectory = [(x, y)]

    def step(self, nx_cmd, ny_cmd, dt):
        """执行一步动力学更新 (含过载变化率限制)"""
        if not self.alive:
            return

        self.x, self.y, self.v, self.heading, self.nx, self.ny = \
            step_dynamics(self.x, self.y, self.v, self.heading,
                         nx_cmd, ny_cmd, dt, self.params,
                         nx_prev=self.nx, ny_prev=self.ny)
        self.trajectory.append((self.x, self.y))

    def step_with_action(self, action, dt):
        """使用归一化动作执行一步"""
        nx_cmd, ny_cmd = action_to_overload(action, self.params)
        self.step(nx_cmd, ny_cmd, dt)

    def kill(self):
        self.alive = False

    def get_state(self):
        return np.array([self.x, self.y, self.v, self.heading,
                        float(self.alive)], dtype=np.float32)

    def distance_to(self, other_x, other_y):
        return np.sqrt((self.x - other_x)**2 + (self.y - other_y)**2)

    def relative_bearing(self, target_x, target_y):
        """目标相对于自身航向的方位角, [-pi, pi]"""
        dx = target_x - self.x
        dy = target_y - self.y
        angle_to_target = np.arctan2(dy, dx)
        relative = angle_to_target - self.heading
        return np.arctan2(np.sin(relative), np.cos(relative))

    def is_in_fov(self, target_x, target_y, fov_half_angle, detection_range):
        """判断目标是否在自身 FOV 内"""
        dist = self.distance_to(target_x, target_y)
        if dist > detection_range:
            return False
        rel_bearing = abs(self.relative_bearing(target_x, target_y))
        return rel_bearing <= fov_half_angle

    def reset(self, x, y, v, heading):
        self.x = x
        self.y = y
        self.v = v
        self.heading = heading
        self.nx = 0.0
        self.ny = 0.0
        self.alive = True
        self.trajectory = [(x, y)]


class HVT:
    """高价值目标 (静止)"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
