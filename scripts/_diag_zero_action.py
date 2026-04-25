"""Quick diagnostic: zero-action / direct-flight baseline + observation sanity."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "MACPO", "MACPO"))

import numpy as np
from envs.fov_penetration.fov_penetration_env import FOVPenetrationEnv

print("="*70)
print("QUICK DIAGNOSTIC — Zero action (trim flat flight) baseline")
print("="*70)

N_EP = int(os.environ.get("N_EP", "3"))
MAX_STEP = int(os.environ.get("MAX_STEP", "2000"))
hits_total = 0
done_reasons = {}

for seed in range(N_EP):
    env = FOVPenetrationEnv({'scenario': 'scenario_1'})
    env.seed(seed * 100)
    obs, share_obs, avail = env.reset()
    init_dists = [o.distance_to(env.hvt.x, env.hvt.y, env.hvt.z) for o in env.offensives]
    init_pos = [(o.x, o.y, o.z, o.heading, o.gamma) for o in env.offensives]
    print(f"\nseed={seed}: HVT=({env.hvt.x:.0f},{env.hvt.y:.0f},{env.hvt.z:.0f})")
    for k, p in enumerate(init_pos):
        print(f"  off[{k}] init pos=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) heading={np.degrees(p[3]):.1f}° gamma={np.degrees(p[4]):.1f}°  init_dist_to_hvt={init_dists[k]:.0f}")
    for d in env.defensives:
        print(f"  def: pos=({d.x:.0f},{d.y:.0f},{d.z:.0f})  alive={d.alive}")

    min_dist_seen = init_dists[:]
    step_done = None
    for step in range(MAX_STEP):
        out = env.step(np.zeros((4, 3)))
        obs, share_obs, rewards, costs, dones, infos, avail = out
        for k, o in enumerate(env.offensives):
            if o.alive:
                d = o.distance_to(env.hvt.x, env.hvt.y, env.hvt.z)
                if d < min_dist_seen[k]:
                    min_dist_seen[k] = d
        if np.all(dones):
            step_done = step
            break

    n_hit = sum(o.hit_hvt for o in env.offensives)
    hits_total += n_hit
    r = infos[0]['done_reason']
    done_reasons[r] = done_reasons.get(r, 0) + 1
    off_alive = sum(1 for o in env.offensives if o.alive)
    def_alive = sum(1 for d in env.defensives if d.alive)
    final_alive = [(o.x, o.y, o.z) for o in env.offensives if o.alive]
    print(f"  RESULT: hit={n_hit}/4, off_alive={off_alive}/4, def_alive={def_alive}/4, "
          f"reason={r}, ended_at_step={step_done}/{MAX_STEP}")
    print(f"  min_dist_seen per off: {[f'{d:.0f}' for d in min_dist_seen]}")
    print(f"  final alive positions: {[(int(p[0]),int(p[1]),int(p[2])) for p in final_alive]}")

print("\n" + "="*70)
print(f"Hit-rate (zero-action baseline): {hits_total}/{N_EP*4} = {hits_total/(N_EP*4)*100:.1f}%")
print(f"Done reasons: {done_reasons}")
print("="*70)
