# swarm_attack_v2 — Repo-wide instructions

When working in this repo, BEFORE editing anything under `envs/fov_penetration/`,
training scripts, eval scripts, or reward code, you MUST load and follow
[.github/skills/env-readonly-fov-penetration/SKILL.md](.github/skills/env-readonly-fov-penetration/SKILL.md).

Hard rules (do not violate):

1. The environment is **frozen** by the user. Never modify `dt` (=0.01),
   `max_steps` (=8000), `hit_hvt_range` / `hit_threshold` (=5.0),
   `collision_kill_range` (=5.0), agent counts, scenario layout, FOV/PN
   config, or aircraft envelope.
2. To "make it learn", only touch `envs/fov_penetration/reward_cost.py`,
   the `reward` block of `envs/fov_penetration/config.py`, training scripts
   under `scripts/`, and policy / MAPPO hyperparameters.
3. Always use `conda run -n rlgpu python ...`. Do not create new conda or
   virtualenv environments.
4. Run a deterministic eval (`scripts/diag_v43.py` or equivalent) before AND
   after a reward / hyperparameter change to verify the intended effect.
5. TensorBoard on port 6006, pointing at the active experiment's logdir.
