---
name: env-readonly-fov-penetration
description: '**FOV PENETRATION ENV READ-ONLY GUARD** — Use ALWAYS when modifying anything in `envs/fov_penetration/**` or related training/eval code. USE FOR: any change to dynamics, action mapping, observation, hit/kill thresholds, scenario layout, episode length, dt, agent count, FOV/lock rules, interceptor PN gains, autopilot config. The user has frozen the environment definition: do NOT modify dynamics.py, entities.py, scenarios, hit_hvt_range/collision_kill_range/hit_threshold, dt, max_steps, n_offensive/n_defensive, FOV angles, PN gains, or v_min/v_max/an_*_max. ONLY allowed targets for "make it learn" work: reward_cost.py, config.py["reward"] block, training scripts in scripts/, policy hyperparams (lr/entropy/std_*), MAPPO/MACPO algorithm code in third_party/. If a change appears to require touching env physics, STOP and ask the user.'
---

# FOV Penetration — Environment Read-Only Guard

This workspace's training environment is **frozen by the user**. Do not touch the
physics, scenario, hit/kill geometry, FOV/lock rules, or episode length while
fixing learning problems. Only reward shaping, training hyperparameters, and
policy/algorithm code may change.

## Frozen environment facts (do not modify)

### Dynamics (`envs/fov_penetration/dynamics.py`)

State `[x, y, z, v, psi(heading), gamma]`; control `[ax, an_pitch, an_yaw]`.

Equations (RK4 integrated, with rate-limit on each control):

```
x_dot     = v cos(gamma) cos(psi)
y_dot     = v cos(gamma) sin(psi)
z_dot     = v sin(gamma)
v_dot     = ax - g sin(gamma)
psi_dot   = an_yaw   / (v cos(gamma))
gamma_dot = (an_pitch - g cos(gamma)) / v
```

Action mapping `action_to_control_3d` (action ∈ [-1,1]^3):
- `action=[0,0,0]` is **trim** (level cruise): `ax=0`, `an_pitch=g` (gravity comp), `an_yaw=0`.
- `action[0]` → `ax` (asymmetric: 0 → 0; +1 → ax_max; -1 → ax_min).
- `action[1]` → `an_pitch` (offset: 0 → g; +1 → an_pitch_max; -1 → -an_pitch_max).
- `action[2]` → `an_yaw` (linear: 0 → 0; ±1 → ±an_yaw_max).

**Implication**: an actor outputting ≈0 still flies straight at v_nominal — it
does NOT crash, stall, or stop. "Trim flight" is the natural baseline.

### Scenario / geometry (`envs/fov_penetration/config.py` `DEFAULT_CONFIG`)

| Key | Value | Notes |
|---|---|---|
| `dt` | 0.01 s | step size; 1 step = 0.45 m at v=45 |
| `max_steps` | 8000 | episode = 80 s of sim time |
| `map_size` | 2000 m | half-width |
| `n_offensive` / `n_defensive` | 4 / 4 | |
| `hvt_position` | `[1200, 0, 0]` | static target |
| offensive init center | `(-1200, 0, 300)` | distance to HVT ≈ 2400 m |
| `hit_hvt_range` / `point_target.hit_threshold` | **5.0 m** | **frozen — both must always be 5.0** |
| `collision_kill_range` / `collision_range` | **5.0 m** | **frozen** |
| `fov_half_angle` | 30° | |
| `detection_range` | 2500 m | |

### Aircraft envelope (frozen)

| | offensive | defensive |
|---|---|---|
| v_min / v_nominal / v_max | 40 / 45 / 50 m/s | 50 / 55 / 60 m/s |
| `ax_min` / `ax_max` | -5 / 20 m/s² | -10 / 30 m/s² |
| `an_pitch_max` / `an_yaw_max` | 2.5 g | 5.0 g |
| `dax_max` | 60 m/s³ | 80 m/s³ |
| `dan_pitch_max` / `dan_yaw_max` | 120 m/s³ | 150 m/s³ |
| `gamma_min` / `gamma_max` | ±15° | ±45° |
| `action_scale` | 1.0 | n/a |

Min turn radius (offensive): R = v² / a_n = 45² / 24.5 ≈ 82.7 m.
Min turn radius (defensive): R = 55² / 49.1 ≈ 61.6 m.

Time-to-traverse 2400 m in straight flight: 2400 / 45 ≈ 53 s ≈ 5300 steps.

### Hit / kill physics (frozen — DO NOT relax)

`fov_penetration_env.py::_check_kills_escapes_and_hits`:

- Interceptor vs offensive: discrete-CPA at threshold
  `cfg["collision_kill_range"]` (= 5 m) → mutual kill.
- Offensive vs HVT: discrete + line-segment CPA at threshold
  `cfg["point_target"]["hit_threshold"]` (= 5 m) → `off.mark_hit_hvt()`,
  `self.hit_count += 1`.

Both thresholds are 5 m by user mandate. Relaxing either is forbidden.

### Defender autopilot

3D PN guidance, navigation gain `pn_nav_gain = 4`, FOV-trigger lock,
`uturn_ay_fraction = 0.20` after fly-by. Defender behavior is part of the
frozen environment.

## What is allowed to change

| Allowed | Not allowed |
|---|---|
| `config.py` → `reward` block (weights, new shaping terms) | `config.py` → scenario / dt / max_steps / agent counts / hit ranges / FOV / PN |
| `reward_cost.py` (entire file) | `dynamics.py`, `entities.py`, `fov_penetration_env.py` step / kill / hit logic |
| `policies_interceptor.py` if explicitly asked | `policies_interceptor.py` by default (defender behavior frozen) |
| Training scripts (`scripts/run_*.sh`, `train_*.py`) including `--lr`, `--entropy_coef`, `--std_x_coef`, `--std_y_coef`, `--clip_param`, `--ppo_epoch`, `--num_mini_batch`, `--gamma`, `--gae_lambda`, `--model_dir` (resume) | `--episode_length`, scenario name, `ap_config` semantics |
| MAPPO/MACPO core under `third_party/MACPO/` | environment registration / scenario constants |
| Diagnostic/eval scripts under `scripts/` | env import paths |

## Required workflow when "make it hit / make success > 0"

0. **Never pause training for testing.** Deterministic eval must run against the
  latest available checkpoint while ongoing training keeps running.
1. **Run a deterministic eval first** (`scripts/diag_v43.py` style): load latest
   actor weights, `deterministic=True`, log per-step `cur_dist`, `min_dist`,
   `avg_action`, `heading_err`, `hit_count` for ≥3 seeds. Distinguish four
   failure modes:
   - actor outputs ≈ 0 (trim) → reward shaping problem
   - actor flies past target then drifts → no end-game / overshoot signal
   - actor turns away early → mis-shaped heading reward
   - actor reaches < 5 m but no hit recorded → **investigate code, do NOT widen threshold**
2. **Pick reward / hyperparameter changes** that target the observed failure
   mode. Document the hypothesis in the run script header.
3. **Resume from latest weights** when possible (`--model_dir
   outputs/results/.../models`) instead of training from scratch.
4. **Use `conda run -n rlgpu`** for every Python command — never create venvs,
   never install new conda envs.
5. **Keep training continuous.** If eval verdict is bad, start an improved
  resume run immediately; do not leave the system with zero active training
  processes at any point.
6. **TensorBoard must stay online** on port `6006`. Treat it as a keepalive
  service for monitoring; if it dies, relaunch immediately to the active
  experiment logdir.

### Latest-checkpoint selection policy (for eval while training runs)

1. Prefer the most recently modified experiment under
  `outputs/results/fov_penetration/mappo/`.
2. Within that experiment, pick the newest `runN/models` that contains all
  four files `actor_agent{0,1,2,3}.pt`.
3. If the active run has no checkpoint yet, fallback to the newest older run
  with complete actor files. Keep current training running while evaluating.

## Common pitfalls (observed in this workspace)

- **Do not "fix" 0 hits by raising `hit_threshold`.** It's 5 m by mandate. If
  `min_dist` is stuck at 200–300 m the policy hasn't learned end-game
  manoeuvre — fix the reward, not the geometry.
- **Trim ≠ doing nothing.** `action=0` flies straight at 45 m/s; any reward
  that pays for "approach speed" or "closing speed" is paid for free during
  trim cruise. Dense shaping must penalise overshoot or reward end-game
  precision, otherwise actor converges to trim and never learns manoeuvre.
- **dt is 0.01 s — CRITICAL, NEVER CHANGE.** Per user mandate, `dt=0.01`
  directly governs whether a 5 m hit threshold is geometrically reachable.
  At v=45 m/s a single step advances 0.45 m, so the discrete-CPA test can
  resolve a 5 m miss; at dt=0.1 the per-step displacement is 4.5 m and
  the policy can never reliably catch a 5 m radius. Raising `dt` to "speed
  up training" or lowering it to "increase precision" both break this
  contract. Same for `max_steps=8000` (= 80 s, plenty of time to traverse
  the 2400 m corridor in ≈ 53 s and still leave 25 s for end-game).
- **Min turn radius ≈ 83 m** at full 2.5 g. To capture a 5 m point from a
  flyby trajectory the policy must commit to end-game pull-up while still
  several hundred metres out. Reward design should incentivise this.
- **HVT guidance features are observation-only**, not an autopilot. The actor
  must learn to act on `pn_hint`, `omega_los`, etc. itself.

## Forbidden quick fixes (prior incidents)

- Changing `dt` in any direction. The 5 m hit threshold is calibrated for
  `dt=0.01 s` per user mandate; any other dt invalidates the hit geometry.
- Widening `hit_hvt_range`, `hit_threshold`, or `collision_kill_range`
  beyond 5.0.
- Adding "near-target snap" or "auto-aim" inside the env step.
- Increasing `max_steps` past 8000 to "give more time" (verified unnecessary
  by diagnostics).
- Replacing PN guidance for defenders with simpler logic to make them
  miss more.

## Sanity checklist before any edit under `envs/fov_penetration/`

- [ ] Is the change limited to `reward_cost.py` or the `reward` block of
      `config.py`? If yes, proceed.
- [ ] If touching anything else under `envs/fov_penetration/`, **STOP and ask
      the user**.
- [ ] No edit to `hit_hvt_range`, `hit_threshold`, `collision_kill_range`,
      `dt`, `max_steps`, scenario layout, agent envelope, FOV, PN gains, or
      defender uturn config.
- [ ] Run `scripts/diag_v43.py` (or equivalent) before AND after to verify
      the change made the intended difference.
