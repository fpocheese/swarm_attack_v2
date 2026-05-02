"""V43 deterministic diagnosis: are agents flying toward HVT? hits? distances?"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from envs.fov_penetration import FOVPenetrationEnv
from eval_v28_10episodes import load_policies
from scripts.phase_obs_wrapper import PhaseMaskedFOVWrapper
from scripts.terminal_pn_action_wrapper import TerminalPNActionWrapper

MODEL_DIR = "outputs/results/fov_penetration/mappo/v44_reward_reshape/run1/models"
HIDDEN = 256
LAYER_N = 3
N_STEPS = 8000
N_EP = 3


def get_actions(policies, obs, device, hidden_size, rnn_states, masks):
    actions = []
    new_rnn = []
    obs_all = obs[0] if isinstance(obs, tuple) else obs
    for i, po in enumerate(policies):
        o = np.asarray(obs_all[i]).flatten()
        obs_t = torch.FloatTensor(o).unsqueeze(0).to(device)
        with torch.no_grad():
            a, _, h = po.actor(obs_t, rnn_states[i], masks[i], deterministic=True)
        actions.append(a.cpu().numpy().flatten())
        new_rnn.append(h)
    return actions, new_rnn


def run_episode(env, policies, device, seed):
    env.seed(seed)
    obs, _, _ = env.reset()
    n = env.n_agents
    hvt = env.hvt
    rnn = [torch.zeros(1, 1, HIDDEN).to(device) for _ in range(n)]
    masks = [torch.ones(1, 1).to(device) for _ in range(n)]

    init_d = [off.distance_to(hvt.x, hvt.y, hvt.z) for off in env.offensives]
    min_d = list(init_d)
    sum_act = np.zeros((n, 3))
    sum_herr = np.zeros(n)
    cnt_alive_steps = np.zeros(n)

    print(f"\n=== seed={seed} ===")
    print(f"HVT @ ({hvt.x:.0f},{hvt.y:.0f},{hvt.z:.0f})")
    for i, off in enumerate(env.offensives):
        bearing = np.arctan2(hvt.y - off.y, hvt.x - off.x)
        herr = (bearing - off.heading + np.pi) % (2*np.pi) - np.pi
        print(f"  Agent{i} init: pos=({off.x:.0f},{off.y:.0f},{off.z:.0f}) "
              f"d={init_d[i]:.0f} hdg={np.degrees(off.heading):.0f} hErr={np.degrees(herr):.0f}")

    final_step = 0
    for step in range(N_STEPS):
        actions, rnn = get_actions(policies, obs, device, HIDDEN, rnn, masks)
        executed_actions = actions
        if hasattr(env, 'guide_actions'):
            executed_actions, _ = env.guide_actions(actions)
        obs, _, _, _, dones, _, _ = env.step(executed_actions)
        final_step = step + 1

        for i, off in enumerate(env.offensives):
            if not off.alive:
                continue
            cnt_alive_steps[i] += 1
            sum_act[i] += np.array(executed_actions[i])
            d = off.distance_to(hvt.x, hvt.y, hvt.z)
            if d < min_d[i]:
                min_d[i] = d
            bearing = np.arctan2(hvt.y - off.y, hvt.x - off.x)
            herr = (bearing - off.heading + np.pi) % (2*np.pi) - np.pi
            sum_herr[i] += abs(herr)

        if step % 500 == 0 or step == N_STEPS-1:
            ds = [f"{off.distance_to(hvt.x,hvt.y,hvt.z):.0f}" if off.alive else "DEAD"
                  for off in env.offensives]
            alive = sum(off.alive for off in env.offensives)
            print(f"  step={step:5d} alive={alive}/{n} d={ds} "
                  f"hits={env.hit_count}")

        if all(dones):
            break

    print(f"--- Episode summary (steps={final_step}) ---")
    print(f"  hit_count={env.hit_count}, hit_idx={env.hit_indices}")
    for i in range(n):
        n_alive = max(cnt_alive_steps[i], 1)
        avg_a = sum_act[i] / n_alive
        avg_herr = np.degrees(sum_herr[i] / n_alive)
        delta_d = init_d[i] - min_d[i]
        off = env.offensives[i]
        end_d = off.distance_to(hvt.x, hvt.y, hvt.z) if off.alive else float('nan')
        print(f"  A{i}: init_d={init_d[i]:.0f} min_d={min_d[i]:.0f} end_d={end_d:.0f} "
              f"closed={delta_d:.0f}m | avg|hErr|={avg_herr:.1f} deg | "
              f"avg_act=[{avg_a[0]:+.3f},{avg_a[1]:+.3f},{avg_a[2]:+.3f}] | alive={off.alive}")
    return {
        'init_d': init_d, 'min_d': min_d,
        'hits': env.hit_count,
        'avg_herr_deg': [np.degrees(sum_herr[i]/max(cnt_alive_steps[i],1)) for i in range(n)],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = FOVPenetrationEnv(scenario='scenario_1')
    obs_mask = os.environ.get('FOV_OBS_PHASE_MASK', 'none').strip().lower()
    if obs_mask != 'none':
        env = PhaseMaskedFOVWrapper(env, mode=obs_mask)
        print(f"Using obs phase mask: {obs_mask}")
    terminal_guidance = os.environ.get('FOV_TERMINAL_GUIDANCE', 'none').strip().lower()
    if terminal_guidance == 'pn_los':
        pn_gain = float(os.environ.get('FOV_TERMINAL_PN_GAIN', '3.0'))
        pn_max_action = float(os.environ.get('FOV_TERMINAL_PN_MAX_ACTION', '0.8'))
        env = TerminalPNActionWrapper(env, gain=pn_gain, max_action=pn_max_action)
        print(f"Using terminal guidance: pn_los gain={pn_gain} max_action={pn_max_action}")
    print(f"Loading policies from {MODEL_DIR} (device={device})")
    policies = load_policies(MODEL_DIR, env, device, hidden_size=HIDDEN, layer_N=LAYER_N)
    results = []
    for s in range(N_EP):
        r = run_episode(env, policies, device, seed=1000+s)
        results.append(r)
    print("\n=========== AGGREGATE ===========")
    total_hits = sum(r['hits'] for r in results)
    print(f"Episodes: {len(results)}, total_hits={total_hits}")
    closed = []
    for r in results:
        for i in range(len(r['init_d'])):
            closed.append(r['init_d'][i] - r['min_d'][i])
    print(f"closed_distance per agent: mean={np.mean(closed):.0f}m, max={np.max(closed):.0f}m, min={np.min(closed):.0f}m")
    all_herr = []
    for r in results:
        all_herr.extend(r['avg_herr_deg'])
    print(f"avg|heading_error|: mean={np.mean(all_herr):.1f} deg (0=朝向HVT, 90=横飞, 180=背离)")


if __name__ == '__main__':
    main()
