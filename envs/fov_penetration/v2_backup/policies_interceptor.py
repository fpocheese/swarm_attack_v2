"""
FOV Penetration Environment - 比例导引法拦截器 V2
==================================================
真实比例导引法 (Proportional Navigation, PN) 制导律:
  a_cmd = N * V_c * d(lambda)/dt

其中:
  N = 3 (导引系数)
  V_c = 接近速度
  lambda = 视线角 (Line-of-Sight angle)
  d(lambda)/dt = 视线角速率

信息更新双模式:
  1. 直接探测模式 (20Hz): 拦截器FOV+距离约束内, 实时获取目标位置
  2. 目指信息模式 (2Hz): 超视距或FOV外, 通过指控雷达获取低频目标位置
     目指信息受通信延迟限制, 更新频率低但不受角度/距离约束

拦截器优先级: 进攻飞行器 > 护卫飞行器
"""

import numpy as np
from .config import G


class InterceptorPolicy:
    """
    基于比例导引法的拦截器控制器 (单个拦截器)
    """

    def __init__(self, interceptor, hvt, config, patrol_idx=0):
        self.interceptor = interceptor
        self.hvt = hvt
        self.config = config
        self.patrol_idx = patrol_idx

        # 导引参数
        self.N = config["pn_nav_gain"]       # 比例导引系数
        self.direct_freq = config["pn_direct_freq"]  # 直接探测频率 20Hz
        self.guide_freq = config["pn_guide_freq"]    # 目指信息频率 2Hz

        # 状态
        self.state = "patrol"
        self.target = None                    # 当前追踪目标
        self.target_type = None               # "attacker" or "escort"

        # 目标位置记录 (含延迟)
        self.target_pos_known = None          # 已知的目标位置 [x, y, vx, vy]
        self.prev_los_angle = None            # 上一次视线角
        self.info_timer = 0.0                 # 信息更新计时器 (s)
        self.guide_timer = 0.0                # 目指信息计时器 (s)

        # 巡逻位置
        n_intc = config["n_interceptors"]
        angle = 2 * np.pi * patrol_idx / n_intc + np.pi
        self.patrol_offset_x = 800.0 * np.cos(angle)
        self.patrol_offset_y = 800.0 * np.sin(angle)

    def reset(self):
        self.state = "patrol"
        self.target = None
        self.target_type = None
        self.target_pos_known = None
        self.prev_los_angle = None
        self.info_timer = 0.0
        self.guide_timer = 0.0

    def get_action(self, attacker, escorts, dt):
        """
        根据当前状态和目标信息选择控制指令

        Returns:
            nx_cmd, ny_cmd: 过载指令 (g)
        """
        intc = self.interceptor
        if not intc.alive:
            return 0.0, 0.0

        fov_half = self.config["fov_half_angle"]
        det_range = self.config["detection_range"]

        # ============ 目标选择和信息更新 ============

        # 更新计时器
        self.info_timer += dt
        self.guide_timer += dt

        # 检查 attacker 是否在直接探测范围
        attacker_in_fov = (attacker.alive and
                           intc.is_in_fov(attacker.x, attacker.y, fov_half, det_range))

        # 检查最近可见 escort
        nearest_escort = None
        nearest_escort_dist = float('inf')
        nearest_escort_in_fov = False
        for esc in escorts:
            if not esc.alive:
                continue
            d = intc.distance_to(esc.x, esc.y)
            if d < nearest_escort_dist:
                nearest_escort_dist = d
                nearest_escort = esc
                nearest_escort_in_fov = intc.is_in_fov(esc.x, esc.y, fov_half, det_range)

        # ============ 目标优先级: 进攻飞行器 > 护卫飞行器 ============
        # 只要attacker存活就追attacker, 除非有escort直接威胁

        old_target = self.target
        direct_detect = False  # 是否直接探测到目标

        if attacker.alive:
            # 优先追 attacker
            if attacker_in_fov:
                # 直接探测到 attacker
                self.target = attacker
                self.target_type = "attacker"
                self.state = "track_attacker"
                direct_detect = True
            elif self.guide_timer >= 1.0 / self.guide_freq:
                # 目指信息模式: 即使看不到, 也能通过指控雷达获取位置
                self.target = attacker
                self.target_type = "attacker"
                self.state = "track_attacker"
                direct_detect = False
            elif self.target_type == "attacker":
                # 保持追踪 (用上次已知位置外推)
                direct_detect = False
            else:
                # 找 escort
                if nearest_escort_in_fov and nearest_escort is not None:
                    self.target = nearest_escort
                    self.target_type = "escort"
                    self.state = "engage_escort"
                    direct_detect = True
                elif nearest_escort is not None and nearest_escort_dist < det_range * 1.5:
                    self.target = nearest_escort
                    self.target_type = "escort"
                    self.state = "engage_escort"
                    direct_detect = False
                else:
                    self.state = "patrol"
                    self.target = None
        else:
            # attacker 已被击杀, 追 escort
            if nearest_escort_in_fov and nearest_escort is not None:
                self.target = nearest_escort
                self.target_type = "escort"
                self.state = "engage_escort"
                direct_detect = True
            elif nearest_escort is not None:
                self.target = nearest_escort
                self.target_type = "escort"
                self.state = "engage_escort"
                direct_detect = False
            else:
                self.state = "patrol"
                self.target = None

        # ============ 信息更新逻辑 ============
        if self.target is not None and self.target.alive:
            should_update = False

            if direct_detect:
                # 直接探测模式: 20Hz 更新
                if self.info_timer >= 1.0 / self.direct_freq:
                    should_update = True
                    self.info_timer = 0.0
            else:
                # 目指信息模式: 2Hz 更新
                if self.guide_timer >= 1.0 / self.guide_freq:
                    should_update = True
                    self.guide_timer = 0.0

            if should_update or self.target_pos_known is None or old_target != self.target:
                # 更新已知位置
                tgt = self.target
                self.target_pos_known = np.array([
                    tgt.x, tgt.y,
                    tgt.v * np.cos(tgt.heading),
                    tgt.v * np.sin(tgt.heading)
                ])
            else:
                # 外推: 用上次已知速度推算当前位置
                if self.target_pos_known is not None:
                    self.target_pos_known[0] += self.target_pos_known[2] * dt
                    self.target_pos_known[1] += self.target_pos_known[3] * dt
        else:
            self.target_pos_known = None

        # ============ 控制律 ============
        if self.state in ("track_attacker", "engage_escort") and self.target_pos_known is not None:
            return self._pn_guidance(dt)
        elif self.state == "patrol":
            return self._patrol_control()
        else:
            return self._goto_hvt()

    def _pn_guidance(self, dt):
        """
        真实比例导引法 (True Proportional Navigation)

        制导指令:
          a_cmd = N * V_c * dλ/dt

        其中:
          λ = atan2(dy, dx) : 目标相对视线角
          V_c = -dR/dt      : 接近速度 (closing velocity)
        """
        intc = self.interceptor
        tgt = self.target_pos_known

        # 相对位置
        dx = tgt[0] - intc.x
        dy = tgt[1] - intc.y
        R = np.sqrt(dx**2 + dy**2)

        if R < 1.0:
            return 0.0, 0.0

        # 当前视线角 (Line-of-Sight angle)
        los_angle = np.arctan2(dy, dx)

        # 视线角速率
        if self.prev_los_angle is not None:
            d_los = los_angle - self.prev_los_angle
            d_los = np.arctan2(np.sin(d_los), np.cos(d_los))  # 归一化
            los_rate = d_los / dt
        else:
            los_rate = 0.0

        self.prev_los_angle = los_angle

        # 相对速度
        vx_intc = intc.v * np.cos(intc.heading)
        vy_intc = intc.v * np.sin(intc.heading)
        vx_tgt = tgt[2]
        vy_tgt = tgt[3]
        dvx = vx_tgt - vx_intc
        dvy = vy_tgt - vy_intc

        # 接近速度 V_c = -dR/dt (正值表示接近)
        V_c = -(dx * dvx + dy * dvy) / max(R, 1.0)
        V_c = max(V_c, intc.v * 0.3)  # 下限

        # PN 制导指令: 法向加速度
        a_cmd = self.N * V_c * los_rate  # m/s^2
        ny_cmd = a_cmd / G                # 转换为过载 (g)

        # 轴向: 根据接近速度和距离调整
        if R < 300.0:
            # 近距减速, 瞄准
            nx_cmd = -0.3
        elif R > 2000.0:
            # 远距加速追击
            nx_cmd = 2.5
        elif V_c < intc.v * 0.5:
            # 接近速度不够, 加速
            nx_cmd = 1.5
        else:
            nx_cmd = 0.5

        # 裁剪
        params = intc.params
        nx_cmd = np.clip(nx_cmd, params["nx_min"], params["nx_max"])
        ny_cmd = np.clip(ny_cmd, params["ny_min"], params["ny_max"])

        return nx_cmd, ny_cmd

    def _goto_hvt(self):
        """飞向 HVT 防御位置"""
        intc = self.interceptor
        patrol_x = self.hvt.x + self.patrol_offset_x
        patrol_y = self.hvt.y + self.patrol_offset_y

        dx = patrol_x - intc.x
        dy = patrol_y - intc.y
        dist = np.sqrt(dx**2 + dy**2)

        if dist < 100.0:
            self.state = "patrol"
            return self._patrol_control()

        target_angle = np.arctan2(dy, dx)
        angle_error = target_angle - intc.heading
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        ny_cmd = 2.5 * angle_error
        nx_cmd = 0.5 if dist > 500.0 else 0.0

        params = intc.params
        nx_cmd = np.clip(nx_cmd, params["nx_min"], params["nx_max"])
        ny_cmd = np.clip(ny_cmd, params["ny_min"], params["ny_max"])

        return nx_cmd, ny_cmd

    def _patrol_control(self):
        """巡逻: 在防御位置附近缓慢盘旋, 朝向进攻方"""
        intc = self.interceptor
        patrol_x = self.hvt.x + self.patrol_offset_x
        patrol_y = self.hvt.y + self.patrol_offset_y

        dist = intc.distance_to(patrol_x, patrol_y)

        if dist > 300.0:
            return self._goto_hvt()

        # 缓慢盘旋
        return 0.0, 1.0
