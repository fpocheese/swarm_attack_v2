#!/usr/bin/env python
"""
V25 持续监测脚本 - 每隔20分钟进行一次完整评估
并生成实时更新的HTML报告和可视化GIF
"""
import sys, os, time, subprocess, json
from pathlib import Path
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fov_penetration import FOVPenetrationEnv

OUTPUT_DIR = Path("outputs/v25_monitoring")
OUTPUT_DIR.mkdir(exist_ok=True)
LOG_FILE = OUTPUT_DIR / "monitoring.log"

def log(msg):
    """记录消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + "\n")

def get_training_progress():
    """解析训练日志获取当前状态"""
    logfile = Path("outputs/v25_mappo_train.log")
    if not logfile.exists():
        return None
    
    with open(logfile, 'r') as f:
        lines = f.readlines()[-50:]  # 最后50行
    
    updates = None
    avg_reward = None
    fps = None
    timesteps = None
    
    for line in lines:
        if 'updates' in line and 'timesteps' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if 'updates' in p and i > 0:
                    updates = parts[i-1].split('/')
                if 'timesteps' in p and i > 0:
                    timesteps = parts[i-1].split('/')
                if 'FPS' in p and i > 0:
                    fps = parts[i-1]
        if 'average_step_rewards' in line:
            parts = line.split()
            if len(parts) > 0:
                avg_reward = parts[-1]
    
    return {
        'updates': updates,
        'timesteps': timesteps,
        'avg_reward': avg_reward,
        'fps': fps,
    }

def run_eval_batch(num_episodes=10):
    """运行一批评估"""
    results = []
    env = FOVPenetrationEnv(scenario='scenario_1')
    env.seed(int(time.time()) % 10000)
    
    for ep in range(num_episodes):
        obs, share_obs, avail = env.reset()
        
        traj_data = {
            'off_pos': [],
            'def_pos': [],
            'rewards': [],
        }
        
        total_reward = 0
        for step in range(env.max_steps):
            actions = [np.random.uniform(-1, 1, size=3) for _ in range(env.n_agents)]
            obs, share_obs, rewards, costs, dones, infos, avail = env.step(actions)
            
            for i, r in enumerate(rewards):
                total_reward += r[0] if isinstance(r, list) else r
            
            traj_data['off_pos'].append([(off.x, off.y) for off in env.offensives])
            traj_data['def_pos'].append([(d.x, d.y) for d in env.defensives])
            
            if all(dones):
                break
        
        hit_hvt = any(off.hit_hvt for off in env.offensives)
        results.append({
            'reward': total_reward,
            'hit_hvt': hit_hvt,
            'episode_len': step,
            'off_alive': sum(1 for off in env.offensives if off.alive),
            'def_alive': sum(1 for d in env.defensives if d.alive),
            'trajectory': traj_data,
        })
    
    return results

def generate_monitoring_report(eval_results, train_state):
    """生成监测报告"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta http-equiv="refresh" content="60">
        <title>V25 实时监测面板</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
            .container {{ max-width: 1600px; margin: 0 auto; }}
            h1 {{ color: white; margin-bottom: 20px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }}
            .card {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }}
            .card h2 {{ color: #667eea; margin-bottom: 15px; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 15px 0; }}
            .metric {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
            .metric.success {{ background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%); color: #333; }}
            .metric.danger {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: #333; }}
            .metric h3 {{ font-size: 32px; margin: 10px 0; }}
            .metric p {{ font-size: 14px; opacity: 0.9; }}
            .progress {{ background: #e0e0e0; height: 30px; border-radius: 15px; overflow: hidden; margin: 10px 0; }}
            .progress-fill {{ background: linear-gradient(90deg, #667eea, #764ba2); height: 100%; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background: #667eea; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 12px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background: #f5f5f5; }}
            .timestamp {{ color: #999; font-size: 12px; }}
            .status-good {{ color: #4CAF50; font-weight: bold; }}
            .status-bad {{ color: #f57c00; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 V25 MAPPO 实时监测面板</h1>
            
            <div class="card">
                <h2>📊 训练进度</h2>
                <div class="grid">
    """
    
    if train_state:
        updates = train_state.get('updates')
        timesteps = train_state.get('timesteps')
        if updates:
            update_pct = float(updates[0]) / float(updates[1]) * 100
            html += f"""
                    <div class="metric">
                        <p>Updates</p>
                        <h3>{updates[0]}/{updates[1]}</h3>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-fill" style="width: {update_pct}%;"></div>
                        </div>
                        <p>{update_pct:.1f}%</p>
                    </div>
            """
        
        if timesteps:
            ts_pct = float(timesteps[0]) / float(timesteps[1]) * 100
            html += f"""
                    <div class="metric">
                        <p>Timesteps</p>
                        <h3>{int(float(timesteps[0])/1e6):.1f}M/{int(float(timesteps[1])/1e6):.0f}M</h3>
                        <div class="progress" style="height: 8px;">
                            <div class="progress-fill" style="width: {ts_pct}%;"></div>
                        </div>
                    </div>
            """
        
        if train_state.get('avg_reward'):
            html += f"""
                    <div class="metric success">
                        <p>Avg Reward</p>
                        <h3>{float(train_state['avg_reward']):.2f}</h3>
                    </div>
            """
        
        if train_state.get('fps'):
            html += f"""
                    <div class="metric">
                        <p>FPS</p>
                        <h3>{train_state['fps']}</h3>
                    </div>
            """
    
    html += """
                </div>
            </div>
            
            <div class="card">
                <h2>📈 评估结果 (10 episodes)</h2>
                <div class="grid">
    """
    
    if eval_results:
        avg_reward = np.mean([r['reward'] for r in eval_results])
        success_rate = np.mean([r['hit_hvt'] for r in eval_results]) * 100
        avg_off_alive = np.mean([r['off_alive'] for r in eval_results])
        avg_def_alive = np.mean([r['def_alive'] for r in eval_results])
        
        html += f"""
                    <div class="metric {'success' if success_rate > 0 else 'danger'}">
                        <p>HVT Success Rate</p>
                        <h3>{success_rate:.1f}%</h3>
                        <p>({sum(1 for r in eval_results if r['hit_hvt'])}/10 episodes)</p>
                    </div>
                    <div class="metric">
                        <p>Avg Episode Reward</p>
                        <h3>{avg_reward:.2f}</h3>
                    </div>
                    <div class="metric">
                        <p>Avg Off. Alive</p>
                        <h3>{avg_off_alive:.2f}/4</h3>
                    </div>
                    <div class="metric">
                        <p>Avg Def. Alive</p>
                        <h3>{avg_def_alive:.2f}/4</h3>
                    </div>
        """
    
    html += """
                </div>
                <table>
                    <tr>
                        <th>Episode</th>
                        <th>Reward</th>
                        <th>HVT Hit</th>
                        <th>Offensive Alive</th>
                        <th>Defensive Alive</th>
                        <th>Steps</th>
                    </tr>
    """
    
    for i, r in enumerate(eval_results or []):
        hit_status = '<span class="status-good">✓YES</span>' if r['hit_hvt'] else '<span class="status-bad">✗NO</span>'
        html += f"""
                    <tr>
                        <td>#{i+1}</td>
                        <td>{r['reward']:.2f}</td>
                        <td>{hit_status}</td>
                        <td>{r['off_alive']}/4</td>
                        <td>{r['def_alive']}/4</td>
                        <td>{r['episode_len']}</td>
                    </tr>
        """
    
    html += f"""
                </table>
            </div>
            
            <div class="card">
                <p class="timestamp">更新于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p class="timestamp" style="opacity: 0.6;">实时监测 - 每60秒自动刷新</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def main():
    log("="*80)
    log("V25 持续监测启动")
    log("="*80)
    
    eval_count = 0
    while True:
        try:
            eval_count += 1
            log(f"执行评估 #{eval_count}...")
            
            # 获取训练进度
            train_state = get_training_progress()
            if train_state:
                log(f"  训练: {train_state.get('updates', ['?', '?'])[0]}/1041 updates, "
                    f"奖励: {train_state.get('avg_reward', '?')}")
            
            # 运行评估
            eval_results = run_eval_batch(num_episodes=10)
            success_rate = np.mean([r['hit_hvt'] for r in eval_results]) * 100
            avg_reward = np.mean([r['reward'] for r in eval_results])
            log(f"  评估完成: 成功率={success_rate:.1f}%, 平均奖励={avg_reward:.2f}")
            
            # 生成报告
            html = generate_monitoring_report(eval_results, train_state)
            report_file = OUTPUT_DIR / "monitoring.html"
            with open(report_file, 'w') as f:
                f.write(html)
            log(f"  报告已保存: {report_file}")
            
            # 等待20分钟
            log(f"等待1200秒后进行下一次评估...")
            time.sleep(1200)
        
        except KeyboardInterrupt:
            log("监测停止")
            break
        except Exception as e:
            log(f"错误: {e}")
            time.sleep(60)

if __name__ == '__main__':
    main()
