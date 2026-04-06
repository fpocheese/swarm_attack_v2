# V34 Analysis & Fix Report
## Date: 2026-04-04

---

## 1. Run11 (V33) Evaluation Results

| Metric | Value |
|---|---|
| **Success rate** | **0.0%** |
| **Hit rate** | **0.0%** |
| **Timeout rate** | 70.0% |
| **All-killed rate** | 30.0% |
| **Avg min dist to HVT** | **887.6m** (worst was 1046.6m) |
| **Avg final dist** | 3649.9m |
| **Avg reward** | 3903.57 |
| **Avg episode length** | 5437 steps |

### Per-episode details:
| Ep | Reward | Steps | Hits | Reason | Min Dist |
|---|---|---|---|---|---|
| 1 | 3944.6 | 6000 | 0 | timeout | 915.5m |
| 2 | 4173.7 | 6000 | 0 | timeout | 562.7m |
| 3 | 4359.2 | 4562 | 0 | all_killed | 853.6m |
| 4 | 3539.3 | 4033 | 0 | all_killed | 1043.2m |
| 5 | 3471.1 | 6000 | 0 | timeout | 969.0m |
| 6 | 3569.5 | 6000 | 0 | timeout | 886.2m |
| 7 | 4476.1 | 6000 | 0 | timeout | 895.0m |
| 8 | 3569.5 | 3776 | 0 | all_killed | 1046.6m |
| 9 | 3910.3 | 6000 | 0 | timeout | 943.7m |
| 10 | 4022.4 | 6000 | 0 | timeout | 760.8m |

**V33 Training was a complete failure**: 0 hits in 10 episodes, agents never get closer than 562m to HVT.

---

## 2. Root Cause Analysis

### PRIMARY BUG: Wrong File Modified
**The env imports `reward_cost.py` but V33 changes were made to `reward_cost_v28.py`.**

```python
# fov_penetration_env.py line 19:
from .reward_cost import compute_rewards, compute_costs, compute_terminal_rewards
```

- `reward_cost.py` = V29 Optimized (unchanged since Apr 2)
- `reward_cost_v28.py` = where V33 exponential close-range code was added
- **Result**: All V33 reward changes (exponential amplification, close-range code) were NEVER USED

### SECONDARY: Reward Scale Imbalance
Even if V33 had been applied correctly, it wouldn't have worked:

| Component | Per-step magnitude | Notes |
|---|---|---|
| Approach reward (d=1500m) | **0.019** | Tiny signal |
| Approach reward (d=500m) | **0.030** | Still tiny |
| V33 approach (d=100m) | 0.116 | Only matters if agent gets there |
| Risk penalty (cone+fov) | **0.230** | Dominates approach 10-90x |
| Risk penalty (cone+fov+danger) | **0.263** | Even worse |

**Root cause**: approach reward normalized by `obs_range=2500m`, making each step's reward ~ `22 * 1.0/2500 = 0.009`. Risk penalties are unnormalized (0.2, 0.03, etc.) so they dominate.

**Agent learns**: "avoid interceptors" (reduces risk -0.23/step) >> "approach HVT" (gains +0.02/step)

### V33 Training Trajectory
```
Update  | Eval Reward | Trend
0       | 5576        | Initial (inherited from run10)
5       | 4702        | ↓ -16%
10      | 4575        | ↓ -18%
15      | 2925        | ↓ -48% (policy collapse)
20      | 4208        | ↑ partial recovery
25      | 4128        | ↓ stagnant
```
Policy was degrading because the risk-dominated reward taught "stay away from everything".

---

## 3. V34 Fix Summary

### 3.1 Wrong-File Bug Fix
Directly modified `reward_cost.py` (the file env actually imports).

### 3.2 Approach Signal Strengthened (5x)
```python
# OLD (V29): normalize by obs_range=2500
approach_r = lambda_rho * delta_rho / obs_range  # = 22 * 1.0/2500 = 0.0088
# NEW (V34): normalize by approach_norm_dist=500
approach_r = lambda_rho * delta_rho / approach_norm_dist  # = 22 * 1.0/500 = 0.044
```

### 3.3 Per-Step Proximity Drive (NEW)
```python
# New: direct per-step penalty for being far from HVT
prox_pen = 0.3 * (d / 2000.0)
# At d=1500m: -0.225/step (comparable to risk penalty!)
# At d=500m:  -0.075/step
# At d=50m:   -0.008/step (negligible near target)
```
Creates continuous gradient toward HVT at all distances.

### 3.4 V33 Exponential Close-Range (NOW ACTUALLY APPLIED)
500m threshold, quadratic curve, max 20x multiplier.
At d=100m: approach per step = 0.579 (was 0.034 in V29).

### 3.5 Risk Penalties Halved
| Param | V33 | V34 | Effect |
|---|---|---|---|
| cone | 0.2 | 0.1 | -50% |
| fov | 0.03 | 0.01 | -67% |
| danger | 0.1 | 0.05 | -50% |
| danger_radius | 150m | 120m | -20% |
| killed_penalty | -1.5 | -1.0 | -33% |
| **Total risk/step** | **~0.263** | **~0.118** | **-55%** |

### 3.6 Net Effect
| Distance | V29 approach | V34 approach | V34 risk | Balance |
|---|---|---|---|---|
| 1500m | 0.019 | **0.097** | 0.118 | Near parity |
| 1000m | 0.025 | **0.123** | 0.118 | **Approach wins** |
| 500m | 0.030 | **0.150** | 0.118 | Approach wins |
| 100m | 0.034 | **0.579** | 0.118 | Approach dominates 5x |

Plus proximity drive adds 0.225/step penalty for staying at 1500m.

### 3.7 Reward Budgets
| Scenario | V29 total | V34 total |
|---|---|---|
| Hit HVT (1500→50m, 3000 steps) | ~6033 | ~6700 |
| Orbit at 800m (6000 steps timeout) | ~3500 | **~2100** |
| **Difference (hit vs orbit)** | 2533 | **4600** |

V34 makes the reward difference between hitting and orbiting 82% larger.

---

## 4. Files Modified

### `envs/fov_penetration/reward_cost.py`
- Updated header to V34
- Section 3.1.2: approach normalization changed from `obs_range` to `approach_norm_dist`
- Section 3.1.2: V33 exponential close-range amplification added
- Section 3.1.2: closing speed amplification near target added
- Section 3.1.4 (NEW): per-step proximity drive
- Section: hit_hvt_bonus default restored to 6000 (V29 had 4500)

### `envs/fov_penetration/config.py`
- Reward section renamed to V34
- New params: `approach_norm_dist=500`, `lambda_proximity=0.3`, `proximity_norm_dist=2000`
- Risk reduced: cone 0.1, fov 0.01, danger 0.05, danger_radius 120
- killed_penalty: -1.0, step_penalty: -0.005
- lambda_no_retreat: 6 (down from 8, proximity handles retreat)
- lambda_hit_los_rate: 0.15 (down from 0.2)

### `scripts/run_v34_train.sh`
- New training script, matches V33 format (conda run)
- Resumes from run11/models

---

## 5. Training Status
- V33 training: **KILLED** (was PID 2342810)
- V34 training: **RUNNING** (PID 968066/1828389)
- Log: `outputs/v34_reward_scale_fix.log`
- Checkpoint: run12 (new)
- First update completed at FPS 537

## 6. Eval Data Saved
- JSON: `outputs/results/latest_model_eval/manual_20260404_185903/eval_10eps.json`
- CSV: `outputs/results/latest_model_eval/manual_20260404_185903/eval_10eps.csv`
- Plots: `outputs/results/latest_model_eval/manual_20260404_185903/`
