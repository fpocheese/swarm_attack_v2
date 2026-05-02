"""Terminal-phase PN action wrapper for FOV penetration policies.

This wrapper leaves the environment dynamics, hit logic, action space, and
observations unchanged. It only replaces policy pitch/yaw commands during the
terminal phase with a deterministic PN-like command computed from the two HVT
LOS-rate observation channels exposed by ``PhaseMaskedFOVWrapper``.
"""

from __future__ import annotations

import math
import numpy as np


class TerminalPNActionWrapper:
    """Apply LOS-rate terminal guidance after phase masking.

    Expected observation layout is the standard 23-D policy observation where
    ``obs[5]`` is normalized HVT azimuth LOS rate and ``obs[6]`` is normalized
    HVT elevation LOS rate. In strict phase mode these are the only non-zero
    terminal-phase guidance inputs.
    """

    def __init__(self, env, gain: float = 3.0, max_action: float = 0.8,
                 z_safety: float = 100.0, z_critical: float = 50.0,
                 anti_dive: float = 0.6, terminal_only: bool = True):
        self.env = env
        self.gain = float(gain)
        self.max_action = float(max_action)
        # Safety floor: when an offensive aircraft is below ``z_safety`` and
        # still descending (gamma < 0), we override the pitch action to pull
        # up. Below ``z_critical`` we apply a hard pull-up regardless of phase.
        # This wrapper-level guard prevents PN from driving a non-strike
        # aircraft into the ground after it has overflown the HVT region.
        self.z_safety = float(z_safety)
        self.z_critical = float(z_critical)
        self.anti_dive = float(anti_dive)
        self.terminal_only = bool(terminal_only)
        self.n_agents = env.n_agents
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self._last_obs = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        obs, share_obs, avail = self.env.reset()
        self._last_obs = self._copy_obs(obs)
        return obs, share_obs, avail

    def step(self, actions):
        guided_actions, guided_count = self.guide_actions(actions)
        obs, share_obs, rewards, costs, dones, infos, avail = self.env.step(guided_actions)
        self._last_obs = self._copy_obs(obs)
        for info in infos:
            if isinstance(info, dict):
                info["terminal_pn_guided_agents"] = guided_count
                info["terminal_pn_gain"] = self.gain
                info["terminal_pn_max_action"] = self.max_action
        return obs, share_obs, rewards, costs, dones, infos, avail

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def _copy_obs(self, obs):
        if isinstance(obs, list):
            return [np.asarray(item, dtype=np.float32).copy() for item in obs]
        return np.asarray(obs, dtype=np.float32).copy()

    def guide_actions(self, actions):
        if self._last_obs is None or not hasattr(self.env, "_terminal_flags"):
            return actions, 0

        obs_arr = np.asarray(self._last_obs, dtype=np.float32)
        action_arr = np.asarray(actions, dtype=np.float32).copy()
        flags = self.env._terminal_flags()
        guided_count = 0

        n = min(len(flags), action_arr.shape[0], obs_arr.shape[0])
        for agent_id in range(n):
            off = self.env.offensives[agent_id]
            if not off.alive or off.hit_hvt:
                continue
            obs_i = obs_arr[agent_id]
            in_terminal = bool(flags[agent_id])
            if in_terminal or (not self.terminal_only):
                # vector-projection PN guidance (target is HVT, stationary)
                # compute geometry
                hvt = self.env.hvt
                range_vec = np.array([hvt.x - off.x, hvt.y - off.y, hvt.z - off.z], dtype=np.float64)
                r = max(np.linalg.norm(range_vec), 1.0)
                # own velocity
                psi = float(off.heading); gam = float(off.gamma); V = float(off.v)
                vx = V * math.cos(gam) * math.cos(psi)
                vy = V * math.cos(gam) * math.sin(psi)
                vz = V * math.sin(gam)
                vel_vec = np.array([vx, vy, vz], dtype=np.float64)
                # relative velocity to stationary HVT
                rel_vel = -vel_vec
                # closing speed (dot of range and own_vel / r)
                closing_speed = -np.dot(range_vec, rel_vel) / r
                # back-half / non-closing fallback to pursuit guidance
                bearing = math.atan2(range_vec[1], range_vec[0])
                bearing_err = math.atan2(math.sin(bearing - psi), math.cos(bearing - psi))
                if abs(bearing_err) > math.pi / 2 or closing_speed <= 0.0:
                    # pure pursuit / tracking fallback
                    an_yaw_cmd = float(np.clip(4.0 * bearing_err, -5.0, 5.0) * (9.81))
                    los_el = math.atan2(range_vec[2], math.hypot(range_vec[0], range_vec[1]))
                    pitch_err = math.atan2(math.sin(los_el - gam), math.cos(los_el - gam))
                    an_pitch_cmd = float(np.clip(3.0 * pitch_err, -4.0, 4.0) * (9.81) + 9.81 * math.cos(gam))
                else:
                    # vector PN: los_omega_vec = cross(range, rel_vel) / r^2
                    los_omega_vec = np.cross(range_vec, rel_vel) / max(r * r, 1e-6)
                    vel_norm = np.linalg.norm(vel_vec)
                    vel_axis = vel_vec / max(vel_norm, 1.0)
                    accel_vec = self.gain * max(closing_speed, 0.0) * np.cross(los_omega_vec, vel_axis)
                    yaw_axis = np.array([-math.sin(psi), math.cos(psi), 0.0], dtype=np.float64)
                    pitch_axis = np.array([
                        -math.sin(gam) * math.cos(psi),
                        -math.sin(gam) * math.sin(psi),
                        math.cos(gam),
                    ], dtype=np.float64)
                    an_yaw_cmd = float(np.dot(accel_vec, yaw_axis))
                    an_pitch_cmd = float(np.dot(accel_vec, pitch_axis)) + 9.81 * math.cos(gam)
                # convert acceleration commands (m/s^2) to normalized action space
                params = getattr(off, "params", None)
                if params is not None:
                    an_pitch_max = float(params.get("an_pitch_max", 2.5 * 9.81))
                    an_yaw_max = float(params.get("an_yaw_max", 5.0 * 9.81))
                else:
                    an_pitch_max = 2.5 * 9.81
                    an_yaw_max = 5.0 * 9.81
                an_pitch_trim = 9.81
                # inverse mapping to action a[1]
                if an_pitch_cmd >= an_pitch_trim:
                    a1 = (an_pitch_cmd - an_pitch_trim) / max(an_pitch_max - an_pitch_trim, 1e-3)
                else:
                    a1 = (an_pitch_cmd - an_pitch_trim) / max(an_pitch_max + an_pitch_trim, 1e-3)
                a2 = an_yaw_cmd / max(an_yaw_max, 1e-3)
                a1 = float(np.clip(a1, -self.max_action, self.max_action))
                a2 = float(np.clip(a2, -self.max_action, self.max_action))
                action_arr[agent_id, 1] = a1
                action_arr[agent_id, 2] = a2
                guided_count += 1
            # --- altitude safety floor: anti-dive pull-up ---
            # Only triggered when NOT actively closing on HVT (i.e., either
            # not in terminal phase, or terminal phase but already past HVT).
            # Strike aircraft on terminal final approach must be allowed to
            # descend toward the ground-level HVT.
            z = float(off.z)
            gamma = float(off.gamma)
            hvt = self.env.hvt
            dx = float(hvt.x - off.x); dy = float(hvt.y - off.y); dz = float(hvt.z - off.z)
            psi = float(off.heading); gam = float(off.gamma); V = float(off.v)
            vx = V * math.cos(gam) * math.cos(psi)
            vy = V * math.cos(gam) * math.sin(psi)
            vz = V * math.sin(gam)
            # closing on HVT if velocity component toward HVT is positive
            closing = (dx * vx + dy * vy + dz * vz) > 0.0
            on_final = in_terminal and closing
            if on_final:
                continue
            if z < self.z_critical:
                action_arr[agent_id, 1] = float(self.anti_dive)
            elif z < self.z_safety and gamma < 0.0:
                cmd = float(action_arr[agent_id, 1])
                if cmd < self.anti_dive:
                    action_arr[agent_id, 1] = float(self.anti_dive)

        if isinstance(actions, list):
            return [action_arr[i].copy() for i in range(action_arr.shape[0])], guided_count
        return action_arr, guided_count