"""
FOV Penetration Environment - Interceptor Policy V22
====================================================
三维比例导引法 (3D PN) + 先入视场即锁定

V22 核心改动 (2026-03-29):
    新增事件触发式锁定状态机:
      INIT_GUIDE → SEARCH → LOCKED → MISSED → ABANDONED

    规则:
      1. 初始阶段: 拦截器按目指信息飞向初始分配目标 (INIT_GUIDE)
      2. 一旦某架进攻飞行器率先进入该拦截器视场 → 立即锁定 (LOCKED)
      3. 锁定后使用 3D PN 追击锁定目标
      4. 若锁定目标脱离视场超过宽限步数 → 标记 MISSED
      5. MISSED 后不重新锁定, 进入 SEARCH/ABANDONED
"""

import numpy as np
from .config import G


class InterceptorPolicy:
    """基于 3D PN 的拦截器控制器 — V22 先入视场即锁定"""

    # 锁定状态
    STATE_INIT_GUIDE = 0   # 按初始目指飞行
    STATE_SEARCH = 1       # 搜索中(无锁定目标)
    STATE_LOCKED = 2       # 已锁定目标, 主动追击
    STATE_MISSED = 3       # 锁定目标丢失/逃脱
    STATE_ABANDONED = 4    # 放弃

    # 向后兼容: 保持旧状态名映射
    STATE_PATROL = STATE_INIT_GUIDE
    STATE_APPROACH = STATE_INIT_GUIDE
    STATE_ENGAGED = STATE_LOCKED

    def __init__(self, interceptor, hvt, config, patrol_idx=0):
        self.interceptor = interceptor
        self.hvt = hvt
        self.config = config
        self.patrol_idx = patrol_idx
        self.N = config["pn_nav_gain"]
        self.direct_freq = config["pn_direct_freq"]
        self.guide_freq = config["pn_guide_freq"]

        # 目标分配 / 锁定
        self.initial_assigned_target_idx = None   # 初始目指目标
        self.current_locked_target_idx = None     # 当前锁定目标
        self.target = None                        # 目标 Aircraft 对象引用
        self.target_pos_known = None
        self.prev_los_az = None
        self.prev_los_el = None
        self.info_timer = 0.0
        self.guide_timer = 0.0

        # V22: 锁定状态机
        self.lock_mode = self.STATE_INIT_GUIDE
        self.first_lock_time = -1         # 首次锁定步数 (-1=从未锁定)
        self.has_ever_locked = False
        self.fov_loss_counter = 0         # 锁定后 FOV 丢失计数

        # 交战跟踪 (保留兼容)
        self.engagement_state = self.STATE_INIT_GUIDE
        self.tracking_steps = 0
        self.demanded_ny = 0.0
        self.demanded_nz = 0.0
        self.closing_speed = 0.0
        self.los_rate_az = 0.0
        self.los_rate_el = 0.0
        self.engagement_min_dist = float('inf')

        # 巡逻偏移
        n_def = config["n_defensive"]
        angle = 2 * np.pi * patrol_idx / max(n_def, 1) + np.pi
        self.patrol_offset_x = 800.0 * np.cos(angle)
        self.patrol_offset_y = 800.0 * np.sin(angle)

        # 配置
        self.pursuit_cfg = config.get("pursuit", {})
        self.fov_escape_cfg = config.get("fov_escape", {})
        self.lock_rules = config.get("enemy_lock_rules", {})

    # ------ 向后兼容属性 ------
    @property
    def assigned_target_idx(self):
        """兼容旧代码: 返回当前锁定目标, 否则返回初始分配目标"""
        if self.current_locked_target_idx is not None:
            return self.current_locked_target_idx
        return self.initial_assigned_target_idx

    def reset(self):
        self.initial_assigned_target_idx = None
        self.current_locked_target_idx = None
        self.target = None
        self.target_pos_known = None
        self.prev_los_az = None
        self.prev_los_el = None
        self.info_timer = 0.0
        self.guide_timer = 0.0
        self.lock_mode = self.STATE_INIT_GUIDE
        self.first_lock_time = -1
        self.has_ever_locked = False
        self.fov_loss_counter = 0
        self.engagement_state = self.STATE_INIT_GUIDE
        self.tracking_steps = 0
        self.demanded_ny = 0.0
        self.demanded_nz = 0.0
        self.closing_speed = 0.0
        self.los_rate_az = 0.0
        self.los_rate_el = 0.0
        self.engagement_min_dist = float('inf')

    def set_initial_target(self, target_idx, target_aircraft):
        """设置初始目指分配目标 (仅用于 INIT_GUIDE 阶段飞行参考)"""
        self.initial_assigned_target_idx = target_idx
        if self.lock_mode == self.STATE_INIT_GUIDE:
            self.target = target_aircraft
            if target_aircraft is not None and target_aircraft.alive:
                self._update_known_position(target_aircraft)

    def set_target(self, target_idx, target_aircraft):
        """兼容旧接口: 等价于 set_initial_target"""
        self.set_initial_target(target_idx, target_aircraft)

    def try_fov_lock(self, off_idx, offensive, current_step):
        """
        尝试视场触发锁定。
        当某架进攻飞行器首先进入该拦截器视场 → 立即锁定。

        Returns:
            locked: bool, 是否发生了新锁定
        """
        if self.lock_mode in (self.STATE_LOCKED, self.STATE_MISSED, self.STATE_ABANDONED):
            return False  # 已经锁定过或已放弃

        intc = self.interceptor
        if not intc.alive or not offensive.alive:
            return False

        fov_half = self.lock_rules.get("lock_fov_threshold",
                                       self.config["fov_half_angle"])
        lock_range = self.lock_rules.get("lock_range_threshold",
                                         self.config["detection_range"])

        in_fov = intc.is_in_fov(offensive.x, offensive.y, offensive.z,
                                fov_half, lock_range)
        if in_fov:
            # 锁定!
            self.current_locked_target_idx = off_idx
            self.target = offensive
            self.lock_mode = self.STATE_LOCKED
            self.engagement_state = self.STATE_LOCKED
            self.first_lock_time = current_step
            self.has_ever_locked = True
            self.fov_loss_counter = 0
            self.tracking_steps = 0
            self.prev_los_az = None
            self.prev_los_el = None
            self.engagement_min_dist = float('inf')
            self._update_known_position(offensive)
            return True

        return False

    def update_lock_state(self, offensives, current_step):
        """
        每步更新锁定状态 (在 env.step() 中调用)。

        Returns:
            lock_event: dict or None (锁定/丢失事件)
        """
        intc = self.interceptor
        if not intc.alive:
            return None

        # INIT_GUIDE 阶段: 扫描所有进攻方, 先入视场即锁定
        if self.lock_mode == self.STATE_INIT_GUIDE:
            if self.lock_rules.get("enable_fov_trigger_lock", True):
                for oi, off in enumerate(offensives):
                    if not off.alive or off.hit_hvt:
                        continue
                    locked = self.try_fov_lock(oi, off, current_step)
                    if locked:
                        return {
                            "type": "fov_trigger_lock",
                            "def_idx": self.patrol_idx,
                            "off_idx": oi,
                            "step": current_step,
                        }
            return None

        # LOCKED 阶段: 维持锁定 / 检测丢失
        if self.lock_mode == self.STATE_LOCKED:
            target = self.target
            if target is None or not target.alive or target.hit_hvt:
                self.lock_mode = self.STATE_ABANDONED
                self.engagement_state = self.STATE_ABANDONED
                return {"type": "lock_target_lost", "def_idx": self.patrol_idx,
                        "off_idx": self.current_locked_target_idx,
                        "step": current_step, "reason": "target_dead"}

            fov_half = self.config["fov_half_angle"]
            det_range = self.config["detection_range"]
            in_fov = intc.is_in_fov(target.x, target.y, target.z,
                                    fov_half, det_range)
            if in_fov:
                self.fov_loss_counter = 0
                self.tracking_steps += 1
            else:
                self.fov_loss_counter += 1
                self.tracking_steps = 0
                persist_limit = self.lock_rules.get("lock_persist_after_fov_loss", 20)
                if self.fov_loss_counter > persist_limit:
                    self.lock_mode = self.STATE_MISSED
                    self.engagement_state = self.STATE_MISSED
                    return {"type": "lock_fov_lost", "def_idx": self.patrol_idx,
                            "off_idx": self.current_locked_target_idx,
                            "step": current_step,
                            "fov_loss_steps": self.fov_loss_counter}

        return None

    def mark_target_missed(self, target_idx):
        """记录一次脱靶/逃逸事件"""
        if self.current_locked_target_idx == target_idx:
            self.lock_mode = self.STATE_MISSED
            self.engagement_state = self.STATE_MISSED
            self.tracking_steps = 0
            self.fov_loss_counter = 0

    def get_action(self, offensives, dt):
        intc = self.interceptor
        if not intc.alive:
            return 0.0, 0.0, 0.0

        fov_half = self.config["fov_half_angle"]
        det_range = self.config["detection_range"]
        self.info_timer += dt
        self.guide_timer += dt

        # 根据锁定状态选择行为
        if self.lock_mode == self.STATE_INIT_GUIDE:
            target = self._get_init_guide_target(offensives)
            if target is None:
                self.demanded_ny = 0.0
                self.demanded_nz = 0.0
                self.closing_speed = 0.0
                return self._patrol_action()
        elif self.lock_mode == self.STATE_LOCKED:
            target = self.target
            if target is None or not target.alive:
                self.demanded_ny = 0.0
                self.demanded_nz = 0.0
                return self._goto_hvt()
        elif self.lock_mode in (self.STATE_MISSED, self.STATE_ABANDONED):
            self.demanded_ny = 0.0
            self.demanded_nz = 0.0
            return self._goto_hvt()
        else:
            self.demanded_ny = 0.0
            self.demanded_nz = 0.0
            return self._patrol_action()

        # 确保有已知目标位置
        if self.target_pos_known is None:
            self._update_known_position(target)

        # 更新交战距离
        dist = intc.distance_3d(target)
        self.engagement_min_dist = min(self.engagement_min_dist, dist)

        target_in_fov = (target.alive and
                         intc.is_in_fov(target.x, target.y, target.z,
                                        fov_half, det_range))

        # 信息更新 (FOV内高频, FOV外低频)
        if target_in_fov and self.info_timer >= 1.0 / self.direct_freq:
            self._update_known_position(target)
            self.info_timer = 0.0
            self.guide_timer = 0.0
        elif self.guide_timer >= 1.0 / self.guide_freq:
            self._update_known_position(target)
            self.info_timer = 0.0
            self.guide_timer = 0.0

        # 外推目标位置
        tx = self.target_pos_known[0] + self.target_pos_known[3] * self.info_timer
        ty = self.target_pos_known[1] + self.target_pos_known[4] * self.info_timer
        tz = self.target_pos_known[2] + self.target_pos_known[5] * self.info_timer

        return self._pn_guidance_3d(tx, ty, tz, dt)

    def _get_init_guide_target(self, offensives):
        """获取初始目指阶段的飞行目标"""
        idx = self.initial_assigned_target_idx
        if idx is not None and idx < len(offensives):
            t = offensives[idx]
            if t.alive:
                self.target = t
                return t
        return None

    def _update_known_position(self, target):
        """更新已知目标位置"""
        cos_g = np.cos(target.gamma)
        self.target_pos_known = [
            target.x, target.y, target.z,
            target.v * cos_g * np.cos(target.heading),
            target.v * cos_g * np.sin(target.heading),
            target.v * np.sin(target.gamma),
        ]

    def _pn_guidance_3d(self, tx, ty, tz, dt):
        """3D比例导引 — 记录demanded overload供外部检查"""
        intc = self.interceptor
        dx = tx - intc.x
        dy = ty - intc.y
        dz = tz - intc.z
        r = max(np.sqrt(dx**2 + dy**2 + dz**2), 1.0)

        los_az = np.arctan2(dy, dx)
        r_horiz = max(np.sqrt(dx**2 + dy**2), 1.0)
        los_el = np.arctan2(dz, r_horiz)

        if self.prev_los_az is not None:
            d_az = np.arctan2(np.sin(los_az - self.prev_los_az),
                              np.cos(los_az - self.prev_los_az))
            los_rate_az = d_az / dt
        else:
            los_rate_az = 0.0

        if self.prev_los_el is not None:
            d_el = np.arctan2(np.sin(los_el - self.prev_los_el),
                              np.cos(los_el - self.prev_los_el))
            los_rate_el = d_el / dt
        else:
            los_rate_el = 0.0

        self.prev_los_az = los_az
        self.prev_los_el = los_el
        self.los_rate_az = los_rate_az
        self.los_rate_el = los_rate_el

        cos_g = np.cos(intc.gamma)
        vx_i = intc.v * cos_g * np.cos(intc.heading)
        vy_i = intc.v * cos_g * np.sin(intc.heading)
        vz_i = intc.v * np.sin(intc.gamma)
        v_closing = max(-(dx * vx_i + dy * vy_i + dz * vz_i) / r, 10.0)
        self.closing_speed = v_closing

        ny_demanded = self.N * v_closing * los_rate_az / G
        nz_demanded = self.N * v_closing * los_rate_el / G + np.cos(intc.gamma)
        nx_cmd = 0.5

        self.demanded_ny = ny_demanded
        self.demanded_nz = nz_demanded

        params = intc.params
        nx_cmd = np.clip(nx_cmd, params["nx_min"], params["nx_max"])
        ny_cmd = np.clip(ny_demanded, params["ny_min"], params["ny_max"])
        nz_cmd = np.clip(nz_demanded, params.get("nz_min", -3.0), params.get("nz_max", 3.0))
        return nx_cmd, ny_cmd, nz_cmd

    def _patrol_action(self):
        intc = self.interceptor
        px = self.hvt.x + self.patrol_offset_x
        py = self.hvt.y + self.patrol_offset_y
        pz = self.hvt.z + 500.0
        dx = px - intc.x
        dy = py - intc.y
        heading_err = np.arctan2(dy, dx) - intc.heading
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        ny_cmd = np.clip(2.0 * heading_err, -2.0, 2.0)
        dz = pz - intc.z
        nz_cmd = np.clip(0.5 * dz / 100.0, -1.0, 1.0) + np.cos(intc.gamma)
        return 0.0, ny_cmd, nz_cmd

    def _goto_hvt(self):
        intc = self.interceptor
        dx = self.hvt.x - intc.x
        dy = self.hvt.y - intc.y
        heading_err = np.arctan2(dy, dx) - intc.heading
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        ny_cmd = np.clip(3.0 * heading_err, -4.0, 4.0)
        dz = (self.hvt.z + 500.0) - intc.z
        nz_cmd = np.clip(0.5 * dz / 100.0, -1.0, 1.0) + np.cos(intc.gamma)
        return 0.5, ny_cmd, nz_cmd

    def is_overload_saturated(self):
        """
        检查当前PN要求的过载是否超出拦截器极限
        返回: (saturated_ny, saturated_nz, saturation_ratio)
        """
        params = self.interceptor.params
        ny_max = params["ny_max"]
        nz_max = params.get("nz_max", 3.0)
        ny_ratio = abs(self.demanded_ny) / max(ny_max, 0.1)
        nz_ratio = abs(self.demanded_nz) / max(nz_max, 0.1)
        saturated_ny = abs(self.demanded_ny) > ny_max
        saturated_nz = abs(self.demanded_nz) > nz_max
        return saturated_ny, saturated_nz, max(ny_ratio, nz_ratio)
