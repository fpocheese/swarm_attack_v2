"""  
FOV Penetration Environment - Entities V5
===========================================
三维同构飞行器 + 暴露追踪 + 命中HVT标记 + 脱靶量记录 + 锁定状态
V5: 控制量从 (ax, ay, mu) 改为 (ax, an_pitch, an_yaw) 惯性系加速度
"""

import numpy as np
from .dynamics import step_dynamics_3d, action_to_control_3d


class Aircraft:
    """三维固定翼无人机 (同构, 不区分 attacker/escort)"""

    def __init__(self, uid, role, params,
                 x=0.0, y=0.0, z=500.0, v=None, heading=0.0, gamma=0.0):
        self.uid = uid
        self.role = role
        self.params = params
        self.x = x
        self.y = y
        self.z = z
        self.v = v if v is not None else params["v_nominal"]
        self.heading = heading
        self.gamma = gamma
        self.ax = 0.0           # 轴向加速度 (m/s²)
        self.an_pitch = 9.81    # 俯仰平面法向加速度 (m/s², 默认=g平飞)
        self.an_yaw = 0.0       # 偏航平面法向加速度 (m/s²)
        self.alive = True
        self.hit_hvt = False
        # 暴露追踪
        self.detected = False
        self.detected_by_count = 0
        self.continuous_exposure = 0
        self.total_exposure_steps = 0
        self.first_detected_step = -1
        self.trajectory = [(x, y, z)]
        # V22: 脱靶量记录 (用于点目标命中)
        self.miss_distance_history = []     # 每步到HVT的距离
        self.min_miss_distance = float('inf')  # 历史最小脱靶量
        self.hit_time = -1                  # 命中时刻 (-1=未命中)
        # V22: 进攻方逃逸标记
        self._escaped_interceptor = False
        self._n_escapes = 0
        # V22: 锁定状态 (进攻方视角)
        self.locked_by_defenders = []       # 哪些拦截器锁定了自己
        self.locked_by_count = 0            # 被锁定数量

    def step(self, ax_cmd, an_pitch_cmd, an_yaw_cmd, dt):
        """执行一步动力学更新, 控制输入 (ax, an_pitch, an_yaw)"""
        if not self.alive:
            return
        result = step_dynamics_3d(
            self.x, self.y, self.z, self.v, self.heading, self.gamma,
            ax_cmd, an_pitch_cmd, an_yaw_cmd, dt, self.params,
            ax_prev=self.ax, an_pitch_prev=self.an_pitch, an_yaw_prev=self.an_yaw)
        self.x, self.y, self.z = result[0], result[1], result[2]
        self.v, self.heading, self.gamma = result[3], result[4], result[5]
        self.ax, self.an_pitch, self.an_yaw = result[6], result[7], result[8]
        self.trajectory.append((self.x, self.y, self.z))

    def step_with_action(self, action, dt):
        """RL 动作接口: 归一化动作 [-1,1]^3 → (ax, an_pitch, an_yaw) → 动力学更新"""
        ax_cmd, an_pitch_cmd, an_yaw_cmd = action_to_control_3d(action, self.params)
        self.step(ax_cmd, an_pitch_cmd, an_yaw_cmd, dt)

    def kill(self):
        self.alive = False

    def mark_hit_hvt(self):
        self.hit_hvt = True

    def update_detection(self, is_detected, detected_by_count, current_step):
        self.detected = is_detected
        self.detected_by_count = detected_by_count
        if is_detected:
            self.continuous_exposure += 1
            self.total_exposure_steps += 1
            if self.first_detected_step < 0:
                self.first_detected_step = current_step
        else:
            self.continuous_exposure = 0

    def distance_to(self, ox, oy, oz=None):
        dx = self.x - ox
        dy = self.y - oy
        if oz is not None:
            dz = self.z - oz
            return np.sqrt(dx**2 + dy**2 + dz**2)
        return np.sqrt(dx**2 + dy**2)

    def distance_3d(self, other):
        return self.distance_to(other.x, other.y, other.z)

    def relative_bearing(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        angle_to_target = np.arctan2(dy, dx)
        relative = angle_to_target - self.heading
        return np.arctan2(np.sin(relative), np.cos(relative))

    def relative_elevation(self, target_x, target_y, target_z):
        dx = target_x - self.x
        dy = target_y - self.y
        dz = target_z - self.z
        horizontal_dist = np.sqrt(dx**2 + dy**2)
        return np.arctan2(dz, max(horizontal_dist, 1.0))

    def is_in_fov(self, target_x, target_y, target_z,
                  fov_half_angle, detection_range):
        dist = self.distance_to(target_x, target_y, target_z)
        if dist > detection_range or dist < 0.1:
            return False
        dx = target_x - self.x
        dy = target_y - self.y
        dz = target_z - self.z
        cos_g = np.cos(self.gamma)
        vx = cos_g * np.cos(self.heading)
        vy = cos_g * np.sin(self.heading)
        vz = np.sin(self.gamma)
        tx, ty, tz = dx / dist, dy / dist, dz / dist
        dot = np.clip(vx * tx + vy * ty + vz * tz, -1.0, 1.0)
        off_axis_angle = np.arccos(dot)
        return off_axis_angle <= fov_half_angle

    def get_state(self):
        return np.array([self.x, self.y, self.z, self.v,
                         self.heading, self.gamma,
                         float(self.alive)], dtype=np.float32)

    def reset(self, x, y, z, v, heading, gamma=0.0):
        self.x, self.y, self.z = x, y, z
        self.v = v
        self.heading = heading
        self.gamma = gamma
        self.ax = 0.0
        self.an_pitch = 9.81  # 默认=g (平飞trim)
        self.an_yaw = 0.0
        self.alive = True
        self.hit_hvt = False
        self.detected = False
        self.detected_by_count = 0
        self.continuous_exposure = 0
        self.total_exposure_steps = 0
        self.first_detected_step = -1
        self.trajectory = [(x, y, z)]
        # V22
        self.miss_distance_history = []
        self.min_miss_distance = float('inf')
        self.hit_time = -1
        self._escaped_interceptor = False
        self._n_escapes = 0
        self.locked_by_defenders = []
        self.locked_by_count = 0


class HVT:
    """高价值目标 (静止)"""
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z
