---
description: "One-click deterministic diagnostic for the FOV penetration training run. Loads the latest actor weights, runs 3 seeds with deterministic=True, prints per-step distances + heading errors + actions, then aggregates min_d / closed_distance / hits and reports likely failure mode (trim white-leech / overshoot / mis-heading / sub-5m miss)."
mode: "agent"
tools: ["codebase", "search", "edit", "runCommands", "terminalLastCommand", "githubRepo"]
---

# `/eval` — One-click FOV penetration diagnostic

You are running a deterministic evaluation of the currently-active FOV penetration training run.

## Inputs (figure out from context, do NOT ask the user)

1. The active experiment name. Default search order:
   - The most recently modified subdir of `outputs/results/fov_penetration/mappo/`
   - If user typed an arg like `/eval v44_reward_reshape`, use that exact name.
2. The diagnostic script: prefer `scripts/diag_v43.py`. If it points at a hard-coded older run dir, **patch the `MODEL_DIR` constant in-place** to the active experiment's `runX/models` (newest run number).
3. Never stop training for eval. If active run has no checkpoint yet, fallback to the newest older run with complete `actor_agent{0,1,2,3}.pt`.

## Hard constraints (must follow `.github/skills/env-readonly-fov-penetration/SKILL.md`)

- Use `conda run --no-capture-output -n rlgpu python ...` for every Python invocation.
- Do NOT modify anything under `envs/fov_penetration/` to "make the diag pass".
- Do NOT modify `dt`, `max_steps`, `hit_threshold`, `hit_hvt_range`, `collision_kill_range`.
- Eval must be **deterministic** (`deterministic=True` in the actor forward pass).

## Workflow

1. Verify at least one training process is alive (`pgrep -fa "train_fov_penetration_mappo.py"`).
   Do not kill or restart it.
2. Locate active experiment + newest `runN/models` dir with all four
   `actor_agentN.pt` files. If active run has none yet, fallback to the newest
   older runnable model dir.
3. Patch `scripts/diag_v43.py` `MODEL_DIR` to the selected model dir if needed.
4. Run:
   ```bash
   conda run --no-capture-output -n rlgpu python scripts/diag_v43.py 2>&1 | tail -120
   ```
5. From the aggregate output extract:
   - `total_hits` across the 3 seeds
   - per-agent `min_d`, `closed_distance`, `avg|heading_error|`, `avg_act`
6. Classify failure mode (one of):
   - **trim_white_leech** — `avg_act` magnitudes < 0.05 on all 3 channels and `min_d` > 200 → actor outputs trim, never manoeuvres.
   - **overshoot** — `min_d` < 200 but `end_d` > min_d * 2 → flies past HVT and drifts.
   - **mis_heading** — `avg|hErr|` > 60° → actor turning wrong way / spiralling.
   - **sub_5m_miss** — any agent with `min_d` < 10 m but `total_hits == 0` → STOP and ask user before any reward edit; this is an env/code bug, not a learning bug.
   - **converging** — `total_hits >= 1` and trending up vs prior eval → keep training.
7. Ensure TensorBoard keepalive on port `6006`:
   - If already listening on `6006`, keep it.
   - If down, start it immediately with active experiment logdir using:
   ```bash
   nohup conda run --no-capture-output -n rlgpu tensorboard --logdir outputs/results/fov_penetration/mappo/<active_exp> --port 6006 --host 0.0.0.0 --reload_interval 30 >/tmp/fov_tb.log 2>&1 &
   ```
8. Report a tight 6-line summary in this exact form:
   ```
   exp=<name> run=<N> step=<step_count_from_models_dir>
   hits=<H>/3 episodes
   min_d:  A0=<>, A1=<>, A2=<>, A3=<>
   |hErr|: A0=<>, A1=<>, A2=<>, A3=<>
   avg_act: A0=<...>, A1=<...>  (mean over agents)
   verdict: <failure_mode> -> <one-sentence next step>
   ```
9. If `verdict == sub_5m_miss`, output a warning block and **do not propose any env or reward edit**.
10. If `verdict` is `trim_white_leech` / `overshoot` / `mis_heading`, immediately hand off to `@Reward Tuner` for one optimization iteration and launch a resumed training run without interrupting existing training first.

## What NOT to do

- Don't kill or restart the running training.
- Don't stop all training processes at any point.
- Don't write a long report — only the 6-line summary plus warnings.
- Don't re-run if `total_hits >= 2` already; just report and keep training.
