"""FOV Penetration Environment - Interceptor Policy V22/V5
====================================================
三维比例导引法 (3D PN) + 先入视场即锁定 + 排他锁定

V5 动力学改动 (2026-04-08):
    控制量从 (ax, ay, mu) 改为 (ax, an_pitch, an_yaw)
    PN制导直接输出俯仰/偏航加速度, 无需合成步骤

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
    """基于 3D PN 的拦截器控制器 — V5 惯性系加速度 + V22 排他锁定"""

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
        self.extrapolate_horizon = config.get("pn_extrapolate_horizon", 0.2)  # V26: 外推参数

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
        self.demanded_an_pitch = 0.0      # 俯仰加速度 (m/s²)
        self.demanded_an_yaw = 0.0        # 偏航加速度 (m/s²)
        self.closing_speed = 0.0
        self.los_rate_az = 0.0
        self.los_rate_el = 0.0
        self.engagement_min_dist = float('inf')

        # V31: 飞越后转弯退化状态
        self._target_passed = False       # 是否已飞越目标
        self._pass_step = -1              # 飞越时的步数
        self._pass_distance = float('inf')  # 飞越时的距离 (CPA)
        self._was_approaching = True      # 上一步是否在接近

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
        self.demanded_an_pitch = 0.0
        self.demanded_an_yaw = 0.0
        self.closing_speed = 0.0
        self.los_rate_az = 0.0
        self.los_rate_el = 0.0
        self.engagement_min_dist = float('inf')
        # V31
        self._target_passed = False
        self._pass_step = -1
        self._pass_distance = float('inf')
        self._was_approaching = True

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

    def try_fov_lock(self, off_idx, offensive, current_step, already_locked_offensives=None):
        """
        尝试视场触发锁定。
        当某架进攻飞行器首先进入该拦截器视场 → 立即锁定。
        已被其他拦截器锁定的进攻方会被跳过，保证一对一分配。

        Returns:
            locked: bool, 是否发生了新锁定
        """
        if self.lock_mode in (self.STATE_LOCKED, self.STATE_MISSED, self.STATE_ABANDONED):
            return False  # 已经锁定过或已放弃

        # 排他锁定: 已被其他拦截器锁定的进攻方不再锁定
        if already_locked_offensives is not None and off_idx in already_locked_offensives:
            return False

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

    def update_lock_state(self, offensives, current_step, already_locked_offensives=None):
        """
        每步更新锁定状态 (在 env.step() 中调用)。
        already_locked_offensives: 已被其他拦截器锁定的进攻方索引集合，
                                   保证不会重复锁定同一目标。

        Returns:
            lock_event: dict or None (锁定/丢失事件)
        """
        intc = self.interceptor
        if not intc.alive:
            return None

        if already_locked_offensives is None:
            already_locked_offensives = set()

        # INIT_GUIDE 阶段: 扫描所有进攻方, 先入视场即锁定 (排除已被锁定目标)
        if self.lock_mode == self.STATE_INIT_GUIDE:
            if self.lock_rules.get("enable_fov_trigger_lock", True):
                for oi, off in enumerate(offensives):
                    if not off.alive or off.hit_hvt:
                        continue
                    locked = self.try_fov_lock(oi, off, current_step,
                                              already_locked_offensives)
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
                # 用户规则: 当前目标死亡后，立即重分配到最近存活进攻方
                new_target = self._get_best_alive_target(offensives)
                if new_target is not None:
                    self.target = new_target
                    self._update_known_position(new_target)
                    self.prev_los_az = None
                    self.prev_los_el = None
                    self.info_timer = 0.0
                    self.guide_timer = 0.0
                    self.lock_mode = self.STATE_MISSED
                    self.engagement_state = self.STATE_MISSED
                    self.current_locked_target_idx = None
                    for oi, off in enumerate(offensives):
                        if off is new_target:
                            self.current_locked_target_idx = oi
                            break
                    return {"type": "lock_target_reassigned", "def_idx": self.patrol_idx,
                            "off_idx": self.current_locked_target_idx,
                            "step": current_step, "reason": "target_dead_retarget"}
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
        """V4动力学: 返回 (ax_cmd, ay_cmd, mu_cmd)"""
        intc = self.interceptor
        if not intc.alive:
            return 0.0, 0.0, 0.0

        fov_half = self.config["fov_half_angle"]
        det_range = self.config["detection_range"]
        self.info_timer += dt
        self.guide_timer += dt

        # V23: 全程PN制导 — 任何状态下都找目标并用PN追击，只有信息更新频率不同
        if self.lock_mode == self.STATE_INIT_GUIDE:
            target = self._get_init_guide_target(offensives)
            if target is None:
                target = self._get_best_alive_target(offensives)
            if target is None:
                self.demanded_ay = 0.0
                self.demanded_mu = 0.0
                self.closing_speed = 0.0
                return self._patrol_action()
        elif self.lock_mode == self.STATE_LOCKED:
            target = self.target
            if not self._is_target_valid(target):
                # 锁定目标已死，切换到最近存活进攻方，继续PN
                target = self._get_best_alive_target(offensives)
            if target is None:
                self.demanded_ay = 0.0
                self.demanded_mu = 0.0
                return self._goto_hvt()
        elif self.lock_mode in (self.STATE_MISSED, self.STATE_ABANDONED):
            # 规则: 一旦选定目标，保持追击直到该目标死亡/命中HVT，再重分配
            target = self.target
            if not self._is_target_valid(target):
                target = self._get_best_alive_target(offensives)
            if target is None:
                self.demanded_ay = 0.0
                self.demanded_mu = 0.0
                return self._goto_hvt()
        else:
            target = self._get_best_alive_target(offensives)
            if target is None:
                self.demanded_ay = 0.0
                self.demanded_mu = 0.0
                return self._patrol_action()

        # V23: 检测目标切换 — 重置导引状态(prev_los/timer)
        if target is not self.target:
            self.target = target
            self._update_known_position(target)
            self.prev_los_az = None
            self.prev_los_el = None
            self.info_timer = 0.0
            self.guide_timer = 0.0

        # 确保有已知目标位置
        if self.target_pos_known is None:
            self._update_known_position(target)

        # 更新交战距离
        dist = intc.distance_3d(target)
        self.engagement_min_dist = min(self.engagement_min_dist, dist)

        target_in_fov = (target.alive and
                         intc.is_in_fov(target.x, target.y, target.z,
                                        fov_half, det_range))

        # ===== V31: 检测飞越(pass-through)事件 =====
        # 策略: 当CPA达到后距离持续增大,说明目标已突破通过
        # 条件: (1) 曾经非常接近 (engagement_min_dist < 阈值)
        #        (2) 当前距离 > engagement_min_dist + 余量 (CPA已过)
        #        (3) 当前距离在继续增大 (非瞬间抖动)
        CPA_TRIGGER_DIST = 150.0   # 当 CPA < 150m 才触发飞越判定
        CPA_MARGIN = 30.0          # 距离比CPA大30m就算飞越

        if not self._target_passed:
            if (self.engagement_min_dist < CPA_TRIGGER_DIST and
                    dist > self.engagement_min_dist + CPA_MARGIN):
                self._target_passed = True
                self._pass_step = self.tracking_steps
                self._pass_distance = dist
        else:
            # 飞越后，检查是否应放弃追击
            abandon_dist = self.pursuit_cfg.get("passed_distance_abandon", 800.0)
            if dist > abandon_dist:
                # 彻底放弃, 飞回HVT保护区
                self.demanded_an_pitch = G
                self.demanded_an_yaw = 0.0
                return self._goto_hvt()

        # 信息更新 (FOV内高频, FOV外低频)
        # V31: 目标在后半球时使用更低频率 (模拟信息延迟)
        if self._target_passed:
            guide_freq_rear = self.config.get("pn_guide_freq_rear", 1.0)
            effective_guide_freq = guide_freq_rear
        else:
            effective_guide_freq = self.guide_freq

        if target_in_fov and self.info_timer >= 1.0 / self.direct_freq:
            self._update_known_position(target)
            self.info_timer = 0.0
            self.guide_timer = 0.0
        elif self.guide_timer >= 1.0 / effective_guide_freq:
            self._update_known_position(target)
            self.info_timer = 0.0
            self.guide_timer = 0.0

        # V26: 启用外推 - 使用速度信息预测目标未来位置
        # 外推时间由 pn_extrapolate_horizon 参数控制（默认0.2秒）
        tx = self.target_pos_known[0] + self.target_pos_known[3] * self.extrapolate_horizon
        ty = self.target_pos_known[1] + self.target_pos_known[4] * self.extrapolate_horizon
        tz = self.target_pos_known[2] + self.target_pos_known[5] * self.extrapolate_horizon

        dx = tx - intc.x
        dy = ty - intc.y
        dz = tz - intc.z
        bearing_err = np.arctan2(dy, dx) - intc.heading
        bearing_err = np.arctan2(np.sin(bearing_err), np.cos(bearing_err))

        range_norm = max(np.sqrt(dx**2 + dy**2 + dz**2), 1.0)
        cos_g = np.cos(intc.gamma)
        vx_i = intc.v * cos_g * np.cos(intc.heading)
        vy_i = intc.v * cos_g * np.sin(intc.heading)
        vz_i = intc.v * np.sin(intc.gamma)
        if self.target_pos_known is not None:
            vx_t = self.target_pos_known[3]
            vy_t = self.target_pos_known[4]
            vz_t = self.target_pos_known[5]
        else:
            vx_t, vy_t, vz_t = 0.0, 0.0, 0.0
        dvx = vx_t - vx_i
        dvy = vy_t - vy_i
        dvz = vz_t - vz_i
        current_closing_speed = -(dx * dvx + dy * dvy + dz * dvz) / range_norm
        self.closing_speed = current_closing_speed

        # PN 在目标进入后半球时会退化，切换到纯追踪制导 (V31: 带回头退化)
        if bearing_err > np.pi / 2 or bearing_err < -np.pi / 2 or current_closing_speed <= 0.0:
            return self._pursuit_guidance_3d(tx, ty, tz)

        return self._pn_guidance_3d(tx, ty, tz, dt)

    def _get_init_guide_target(self, offensives):
        """获取初始目指阶段的飞行目标 (优先初始分配目标)"""
        idx = self.initial_assigned_target_idx
        if idx is not None and idx < len(offensives):
            t = offensives[idx]
            if t.alive and not t.hit_hvt:
                self.target = t
                return t
        return None

    def _is_target_valid(self, target):
        """目标有效性: 存在且存活且未命中HVT。"""
        return (target is not None) and target.alive and (not target.hit_hvt)

    def _get_best_alive_target(self, offensives):
        """V23: 返回距本拦截器最近的存活进攻方 (用于全程PN制导)"""
        intc = self.interceptor
        best = None
        best_dist = float('inf')
        for off in offensives:
            if not off.alive or off.hit_hvt:
                continue
            d = intc.distance_3d(off)
            if d < best_dist:
                best_dist = d
                best = off
        return best

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
        """3D比例导引 — V5: 输出 (ax_cmd, an_pitch_cmd, an_yaw_cmd)"""
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

        # 相对速度计算闭合速度
        if self.target_pos_known is not None:
            vx_t = self.target_pos_known[3]
            vy_t = self.target_pos_known[4]
            vz_t = self.target_pos_known[5]
        else:
            vx_t, vy_t, vz_t = 0.0, 0.0, 0.0
        dvx = vx_t - vx_i
        dvy = vy_t - vy_i
        dvz = vz_t - vz_i
        v_closing = -(dx * dvx + dy * dvy + dz * dvz) / r
        self.closing_speed = v_closing

        # V5 PN 加速度指令 — 使用向量投影法计算加速度向量后投影到机体轴
        # 这样可以避免标量 LOS 角速率在倾斜/俯仰角存在时的误差
        range_vec = np.array([dx, dy, dz], dtype=np.float64)
        rel_vel = np.array([dvx, dvy, dvz], dtype=np.float64)  # v_target - v_interceptor
        # los_omega_vec = cross(range, rel_vel) / r^2  (向量 LOS 角速率)
        los_omega_vec = np.cross(range_vec, rel_vel) / max(r * r, 1e-6)

        velocity_vec = np.array([vx_i, vy_i, vz_i], dtype=np.float64)
        vel_norm = np.linalg.norm(velocity_vec)
        velocity_axis = velocity_vec / max(vel_norm, 1.0)

        # 加速度矢量 (m/s^2)
        accel_vec = self.N * max(v_closing, 0.0) * np.cross(los_omega_vec, velocity_axis)

        # 机体偏航轴与俯仰轴（单位向量）
        yaw_axis = np.array([-np.sin(intc.heading), np.cos(intc.heading), 0.0], dtype=np.float64)
        pitch_axis = np.array([
            -np.sin(intc.gamma) * np.cos(intc.heading),
            -np.sin(intc.gamma) * np.sin(intc.heading),
            np.cos(intc.gamma),
        ], dtype=np.float64)

        an_yaw_cmd = float(np.dot(accel_vec, yaw_axis))
        an_pitch_cmd = float(np.dot(accel_vec, pitch_axis)) + G * np.cos(intc.gamma)

        self.demanded_an_pitch = an_pitch_cmd
        self.demanded_an_yaw = an_yaw_cmd

        # 轴向加速度: 重力补偿 + 微小前向加速
        ax_cmd = G * np.sin(intc.gamma) + 5.0

        # 饱和保护
        params = intc.params
        ax_cmd = np.clip(ax_cmd, params["ax_min"], params["ax_max"])
        an_pitch_cmd = np.clip(an_pitch_cmd, -params["an_pitch_max"], params["an_pitch_max"])
        an_yaw_cmd = np.clip(an_yaw_cmd, -params["an_yaw_max"], params["an_yaw_max"])

        return ax_cmd, an_pitch_cmd, an_yaw_cmd

    def _patrol_action(self):
        """V5: 巡逻模式, 输出 (ax, an_pitch, an_yaw)"""
        intc = self.interceptor
        px = self.hvt.x + self.patrol_offset_x
        py = self.hvt.y + self.patrol_offset_y
        pz = self.hvt.z + 500.0
        dx = px - intc.x
        dy = py - intc.y
        heading_err = np.arctan2(dy, dx) - intc.heading
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        # 偏航校正加速度 (m/s²)
        an_yaw_cmd = np.clip(2.0 * heading_err, -2.0, 2.0) * G
        # 俯仰校正加速度 (m/s²) + 重力补偿
        dz = pz - intc.z
        an_pitch_cmd = np.clip(0.5 * dz / 100.0, -1.0, 1.0) * G + G * np.cos(intc.gamma)
        ax_cmd = G * np.sin(intc.gamma)  # 重力补偿, 巡航
        # 饱和
        params = intc.params
        an_pitch_cmd = np.clip(an_pitch_cmd, -params["an_pitch_max"], params["an_pitch_max"])
        an_yaw_cmd = np.clip(an_yaw_cmd, -params["an_yaw_max"], params["an_yaw_max"])
        ax_cmd = np.clip(ax_cmd, params["ax_min"], params["ax_max"])
        return ax_cmd, an_pitch_cmd, an_yaw_cmd

    def _goto_hvt(self):
        """V5: 飞向HVT, 输出 (ax, an_pitch, an_yaw)"""
        intc = self.interceptor
        dx = self.hvt.x - intc.x
        dy = self.hvt.y - intc.y
        heading_err = np.arctan2(dy, dx) - intc.heading
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        # 偏航校正加速度 (m/s²)
        an_yaw_cmd = np.clip(3.0 * heading_err, -4.0, 4.0) * G
        # 俯仰校正加速度 (m/s²) + 重力补偿
        dz = (self.hvt.z + 500.0) - intc.z
        an_pitch_cmd = np.clip(0.5 * dz / 100.0, -1.0, 1.0) * G + G * np.cos(intc.gamma)
        ax_cmd = G * np.sin(intc.gamma) + 5.0  # 重力补偿 + 微小前向加速
        # 饱和
        params = intc.params
        an_pitch_cmd = np.clip(an_pitch_cmd, -params["an_pitch_max"], params["an_pitch_max"])
        an_yaw_cmd = np.clip(an_yaw_cmd, -params["an_yaw_max"], params["an_yaw_max"])
        ax_cmd = np.clip(ax_cmd, params["ax_min"], params["ax_max"])
        return ax_cmd, an_pitch_cmd, an_yaw_cmd

    def _pursuit_guidance_3d(self, tx, ty, tz):
        """
        目标后半球时的纯追踪制导
        V5: 输出 (ax, an_pitch, an_yaw)
        V31: 飞越后大幅降低机动能力 (模拟真实导弹回头困难)
        """
        intc = self.interceptor
        dx = tx - intc.x
        dy = ty - intc.y
        dz = tz - intc.z

        heading_err = np.arctan2(dy, dx) - intc.heading
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))

        params = intc.params

        # === V31: 飞越后机动能力退化 ===
        if self._target_passed:
            pcfg = self.pursuit_cfg
            accel_fraction = pcfg.get("uturn_ay_fraction", 0.20)
            recovery_steps = pcfg.get("uturn_recovery_steps", 500)
            ax_brake = pcfg.get("uturn_ax_brake", -8.0)

            # 逐步恢复: 从 accel_fraction 线性恢复到 1.0
            steps_since_pass = max(self.tracking_steps - self._pass_step, 0)
            recovery_ratio = min(steps_since_pass / max(recovery_steps, 1), 1.0)
            current_fraction = accel_fraction + (1.0 - accel_fraction) * recovery_ratio

            # 限制加速度
            an_pitch_max_eff = params["an_pitch_max"] * current_fraction
            an_yaw_max_eff = params["an_yaw_max"] * current_fraction

            # 降低追踪增益 (回头更慢)
            gain_h = 1.0 * current_fraction   # 正常是 4.0
            gain_v = 0.8 * current_fraction   # 正常是 3.0

            an_yaw_cmd = np.clip(gain_h * heading_err, -2.0, 2.0) * G

            horiz = max(np.sqrt(dx**2 + dy**2), 1.0)
            pitch_err = np.arctan2(dz, horiz) - intc.gamma
            pitch_err = np.arctan2(np.sin(pitch_err), np.cos(pitch_err))
            an_pitch_cmd = np.clip(gain_v * pitch_err, -1.5, 1.5) * G + G * np.cos(intc.gamma)

            # 飞越后减速 → 回头更困难, 速度优势消失
            ax_cmd = ax_brake

            an_pitch_cmd = np.clip(an_pitch_cmd, -an_pitch_max_eff, an_pitch_max_eff)
            an_yaw_cmd = np.clip(an_yaw_cmd, -an_yaw_max_eff, an_yaw_max_eff)
            ax_cmd = np.clip(ax_cmd, params["ax_min"], params["ax_max"])
            return ax_cmd, an_pitch_cmd, an_yaw_cmd

        # 正常追踪制导 (未飞越)
        an_yaw_cmd = np.clip(4.0 * heading_err, -5.0, 5.0) * G

        horiz = max(np.sqrt(dx**2 + dy**2), 1.0)
        pitch_err = np.arctan2(dz, horiz) - intc.gamma
        pitch_err = np.arctan2(np.sin(pitch_err), np.cos(pitch_err))
        an_pitch_cmd = np.clip(3.0 * pitch_err, -4.0, 4.0) * G + G * np.cos(intc.gamma)

        ax_cmd = G * np.sin(intc.gamma)

        an_pitch_cmd = np.clip(an_pitch_cmd, -params["an_pitch_max"], params["an_pitch_max"])
        an_yaw_cmd = np.clip(an_yaw_cmd, -params["an_yaw_max"], params["an_yaw_max"])
        ax_cmd = np.clip(ax_cmd, params["ax_min"], params["ax_max"])
        return ax_cmd, an_pitch_cmd, an_yaw_cmd

    def is_overload_saturated(self):
        """
        检查当前PN要求的法向加速度是否超出拦截器极限
        返回: (pitch_saturated, yaw_saturated, max_saturation_ratio)
        """
        params = self.interceptor.params
        an_pitch_max = params["an_pitch_max"]
        an_yaw_max = params["an_yaw_max"]
        pitch_ratio = abs(self.demanded_an_pitch) / max(an_pitch_max, 0.1)
        yaw_ratio = abs(self.demanded_an_yaw) / max(an_yaw_max, 0.1)
        max_ratio = max(pitch_ratio, yaw_ratio)
        pitch_sat = abs(self.demanded_an_pitch) > an_pitch_max
        yaw_sat = abs(self.demanded_an_yaw) > an_yaw_max
        return pitch_sat or yaw_sat, pitch_sat or yaw_sat, max_ratio
