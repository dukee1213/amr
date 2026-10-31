import numpy as np
import os, math
import matplotlib.pyplot as plt
import cv2 as cv
import functools
from copy import copy
from gymnasium.spaces import Discrete, Box
from pettingzoo import ParallelEnv

class AMRStewardshipEnv(ParallelEnv):
    metadata = {"name": "amr_stewardship_v0"}
    def __init__(self, num_agents=200, ward_side_len=50.0, render_mode=None, w_beta_scale=1.0):
        self.render_mode = render_mode
        self.w_beta_scale = float(w_beta_scale)
        self.ward_side_len = float(ward_side_len)
        self.possible_agents = [f"prescriber_{i}" for i in range(int(num_agents))]
        
        self.timestep = None
        self.agents = None
        
        self.position = None        # [x, y]
        self.severity = None        # {1,2,3} From mild to severe 
        self.comorbidities = None   # [bool]*5
        self.prescriber_types = None # {0,1,2} Conservative, Aggressive, and Guideline 
        
        self.S = None               # bacterial population(S and R) measured by CFU
        self.R = None               
        
        self.ward_environmental_shedding = None # 2D float array as ward layout. Each value is bac density (0-1)
        self.surveillance_history = None # saves past `self.ward_environmental_shedding` as a list
        self.history = None
        self.ward_drug_usage = None
        self.infection_duration = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = copy(self.possible_agents)
        self.timestep = 0
        
        self.position = np.random.uniform(0, self.ward_side_len, size=(self.num_agents, 2))
        self.severity = np.random.choice([1, 2, 3], size=self.num_agents)
        self.comorbidities = np.random.randint(0, 2, size=(self.num_agents, 5))
        
        num_cons = int(0.3 * self.num_agents)
        num_agg = int(0.2 * self.num_agents)
        num_guide = self.num_agents - num_cons - num_agg
        types = [0] * num_cons + [1] * num_agg + [2] * num_guide
        self.prescriber_types = np.array(types)# Prescriber(30% / 20% / 50%) 
        
        self.S = np.zeros(self.num_agents, dtype=np.float64)
        self.R = np.zeros(self.num_agents, dtype=np.float64)
        initial_infected = np.random.choice(self.num_agents, size=max(1, int(0.20 * self.num_agents)), replace=False)
        # Create heterogeneous infections to force diverse drug usage
        for idx in initial_infected:
            if np.random.rand() < 0.5:
                self.S[idx] = 1e5  # S-dominant
                self.R[idx] = 1e3
            else:
                self.S[idx] = 1e3  # R-dominant
                self.R[idx] = 1e5
        # zero R bac in wards at the start
        self.ward_environmental_shedding = np.zeros((int(self.ward_side_len), int(self.ward_side_len)))
        self.surveillance_history = [] # the memory buffer

        self.history = np.zeros((self.num_agents, 2))
        self.ward_drug_usage = {0: 0, 1: 0, 2: 0}
        self.infection_duration = np.zeros(self.num_agents)
        
        observations = {}
        infos = {}
        for agent in self.agents:
            agent_idx = int(agent.split("_")[-1])
            observations[agent] = self._get_observation(agent_idx)
            infos[agent] = {"prescriber_type": int(self.prescriber_types[agent_idx])}
        return observations, infos

    def _get_observation(self, agent_idx):
        x, y = self.position[agent_idx]
        sev = float(self.severity[agent_idx])
        comorb = self.comorbidities[agent_idx].astype(np.float32)
        history = self.history[agent_idx].astype(np.float32)
        if len(self.surveillance_history) > 1:
            yesterday_grid = self.surveillance_history[-2] 
            x_idx, y_idx = min(int(x), int(self.ward_side_len)-1), min(int(y), int(self.ward_side_len)-1)
            raw_rho = yesterday_grid[x_idx, y_idx]
            noise = np.random.normal(0, 0.02)
            delayed_noisy_rho = max(0.0, float(raw_rho) + float(noise))
        else:
            delayed_noisy_rho = 0.0
        current_load = float(self.S[agent_idx] + self.R[agent_idx])
        log_load = float(math.log10(max(1.0, current_load)) / 9.0)
        obs = np.concatenate(([x/self.ward_side_len, y/self.ward_side_len, sev/3.0], comorb, history, [delayed_noisy_rho, log_load]))
        return obs.astype(np.float32)
        
    def step(self, actions):
        rewards = {a: 0.0 for a in self.agents}
        terminations = {a: False for a in self.agents}
        truncations = {a: False for a in self.agents}
        pre_step_total_load = self.S + self.R
        total_loads = pre_step_total_load.copy()
        total_loads[total_loads == 0] = 1.0 
        pop_resistance_ratios = self.R / total_loads
        mean_pop_resistance = np.mean(pop_resistance_ratios)
        phi_vals = [2.2, 2.4, 2.5]
        alpha_vals = [0.15, 0.08, 0.20]
        weight_maps = { 
            0: (1.8, 2.0, 0.3, 0.1), 
            1: (2.0, 0.2, 0.2, 0.2), 
            2: (1.9, 1.8, 0.3, 0.1)  
        }
        for agent in self.agents:
            i = int(agent.split("_")[-1])
            if terminations[agent] or pre_step_total_load[i] < 1e4:
                continue
            self.infection_duration[i] += 1
            c_drug = actions[agent]
            phi_c = phi_vals[c_drug]
            # explicitly control
            if self.severity[i] == 1 and c_drug == 2:
                phi_c *= 0.5
            alpha_c = alpha_vals[c_drug]
            for _ in range(10):
                current_S, current_R = self.S[i], self.R[i]
                N_total = current_S + current_R
                if N_total == 0: break
                growth_term = max(0, 1 - (N_total / 10**9))
                dS = 0.6 * current_S * growth_term - 0.1 * current_S - phi_c * current_S - 10**-6 * current_S
                dR = 0.55 * current_R * growth_term - 0.1 * current_R - (alpha_c * phi_c) * current_R + 10**-6 * current_S
                self.S[i] = max(0, current_S + dS * 0.1)
                self.R[i] = max(0, current_R + dR * 0.1)
        self._update_environmental_grid()
        for i in range(self.num_agents):
            x_idx, y_idx = min(int(self.position[i][0]), int(self.ward_side_len)-1), min(int(self.position[i][1]), int(self.ward_side_len)-1)
            local_env_density = self.ward_environmental_shedding[x_idx, y_idx]
            p_env = 1 - np.exp(-0.05 * local_env_density)
            if np.random.rand() < p_env:
                self.R[i] += 10**3
            for j in range(self.num_agents):
                if i == j: continue
                dist = np.linalg.norm(self.position[i] - self.position[j])
                if dist < 2.0: 
                    source_total = self.S[j] + self.R[j]
                    if source_total > 0:
                        source_R_ratio = self.R[j] / source_total
                        p_contact = 1 - np.exp(-0.15 * source_R_ratio)
                        if np.random.rand() < p_contact:
                            self.R[i] += 10**4 * source_R_ratio
                            self.S[i] += 10**4 * (self.S[j] / source_total)
        post_step_total_load = self.S + self.R
        infos = {}
        for agent in self.agents:
            i = int(agent.split("_")[-1])
            c_drug = int(actions[agent])
            w_alpha, w_beta, w_gamma, w_delta = weight_maps[int(self.prescriber_types[i])]
            w_beta = float(w_beta * self.w_beta_scale)
            pre_load = max(1.0, float(pre_step_total_load[i]))
            post_load = max(1.0, float(post_step_total_load[i]))
            log_reduction = float(math.log10(pre_load) - math.log10(post_load))
            is_failed = post_load >= 0.9 * 1e9
            is_cured = post_load < 1e3 and pre_load >= 1e3
            if is_failed or is_cured:
                terminations[agent] = True
                self.S[i] = 0.0
                self.R[i] = 0.0
            if is_cured:
                r_treat = 3.0 
                if self.infection_duration[i] <= 7: r_treat += 5.0
                elif self.infection_duration[i] <= 14: r_treat += 2.0
            elif is_failed:
                r_treat = -150.0
            else:
                r_treat = (10.0 * log_reduction) + 2.0

            # 2. Malpractice Penalties
            malpractice = 0.0
            if self.severity[i] == 1 and c_drug == 2:
                malpractice -= 25.0
            if self.severity[i] == 3 and c_drug == 0 and pre_load > 1e6:
                malpractice -= 20.0
            
            # 3. Resistance Wall (Calculated per agent based on global state)
            r_ratio = float(self.R[i] / max(1.0, float(post_step_total_load[i])))
            res_wall = 0.0
            if mean_pop_resistance > 0.30:
                res_wall = np.exp((mean_pop_resistance - 0.30) * 25.0)
            
            # Apply all components to the specific info keys
            r_resist_total = r_ratio + float(2.0 * mean_pop_resistance) + res_wall
            
            infos[agent] = {
                "r_treat": float(w_alpha * (r_treat + malpractice)), # Integrated Malpractice
                "r_resist": float(-w_beta * r_resist_total),         # Integrated Wall
                "r_cost": float(-w_gamma * (base_costs[c_drug] / 10.0)),
                "r_adverse": float(-w_delta * (0.0 if is_cured else 0.025))
            }
            # Summation of all signals
            rewards[agent] = sum(infos[agent].values())
        self.timestep += 1
        if self.timestep >= 30:
            for a in self.agents: truncations[a] = True
        observations = {a: self._get_observation(int(a.split("_")[-1])) for a in self.agents}
        self.agents = [a for a in self.agents if not (terminations[a] or truncations[a])]
        return observations, rewards, terminations, truncations, infos

    def _update_environmental_grid(self):
        """
        Spatial shedding grid
        """
        self.ward_environmental_shedding.fill(0.0)
        h = 2.0 # bandwidth
        omega_env = 10**-4 # shedding fraction
        
        grid_x, grid_y = np.mgrid[0:int(self.ward_side_len), 0:int(self.ward_side_len)]
        
        for i in range(self.num_agents):
            if self.R[i] > 0:
                agent_x, agent_y = self.position[i]
                # Gaussian kernel K_h(d)
                dist_sq = (grid_x - agent_x)**2 + (grid_y - agent_y)**2
                kernel = (1 / (2 * np.pi * h**2)) * np.exp(-dist_sq / (2 * h**2)) 
                
                self.ward_environmental_shedding += omega_env * self.R[i] * kernel 
        # Append to surveillance history for the 1-day lag 
        self.surveillance_history.append(self.ward_environmental_shedding.copy())
        if len(self.surveillance_history) > 5:
            self.surveillance_history.pop(0)
    def render(self, root="./", filename="amr_stewardship.mp4"):
        if not hasattr(self, '_frames'):
            self._frames = []
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            self.ward_environmental_shedding.T,
            origin='lower',
            extent=[0, float(self.ward_side_len), 0, float(self.ward_side_len)],
            cmap='Reds',
            alpha=0.5,
            vmin=0.0,
            vmax=0.05
        )
        total_loads = self.S + self.R
        colors = []
        for load in total_loads:
            load_val = float(load)
            if load_val < 1000.0:
                colors.append('green')
            elif load_val > 1000000000.0:
                colors.append('red')
            else:
                colors.append('orange')
        ax.scatter(self.position[:, 0], self.position[:, 1], c=colors, s=80, edgecolors='black')
        ax.set_xlim(0, float(self.ward_side_len))
        ax.set_ylim(0, float(self.ward_side_len))
        ax.set_title(f"Step: {int(self.timestep)}")
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())
        img_bgr = cv.cvtColor(img[:, :, :3], cv.COLOR_RGB2BGR)
        self._frames.append(img_bgr)
        plt.close(fig)
        if int(self.timestep) >= 30 or len(self.agents) == 0:
            out_path = os.path.join(root, filename)
            height, width, _ = self._frames[0].shape
            fourcc = cv.VideoWriter_fourcc(*'mp4v')
            video_writer = cv.VideoWriter(out_path, fourcc, 5.0, (width, height))
            for frame in self._frames:
                video_writer.write(frame)   
            video_writer.release()
            self._frames = []
    # def _run_random_baseline(self, num_episodes=30, num_days=30):

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(3)

def plot_stewardship_analysis(sev_stats, load_stats, rho_stats, temp_data):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
    width = 0.25
    x_indices = np.arange(3)
    drug_labels = ['Drug 0', 'Drug 1', 'Drug 2']
    colors = ['#5dade2', '#58d68d', '#ec7063']

    def draw_bars(ax, data_dict, x_tick_labels, title):
        keys = list(data_dict.keys())
        for i in range(3):
            vals = [data_dict[k][i] for k in keys]
            totals = [max(1, sum(data_dict[k])) for k in keys]
            pcts = [(v/t)*100 for v, t in zip(vals, totals)]
            ax.bar(x_indices + (i*width), pcts, width, label=drug_labels[i], color=colors[i])
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_ylabel('Usage Percentage (%)')
        ax.set_xticks(x_indices + width, x_tick_labels)
        ax.legend(loc='best')
        ax.set_ylim(0, 110)

    draw_bars(axs[0,0], sev_stats, ['Mild (1)', 'Moderate (2)', 'Severe (3)'], 'Policy by Patient Severity')
    draw_bars(axs[0,1], load_stats, ['Low (<1e6)', 'Mid', 'High (>1e8)'], 'Policy by Bacterial Load')
    draw_bars(axs[1,0], rho_stats, ['Clean (<0.2)', 'Mid', 'Hot (>0.4)'], 'Policy by Local Resistance Density')

    # Temporal Dynamics Subplot
    ax4 = axs[1,1]
    ax4_res = ax4.twinx()
    days = np.arange(30)
    ax4.plot(days, temp_data['infections'], color='black', label='Infections', linewidth=2.5)
    ax4_res.plot(days, temp_data['res']*100, color='#e67e22', linestyle='--', label='Resistance %', linewidth=2)
    ax4.set_title('Temporal Ward Dynamics', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Active Infection Count')
    ax4_res.set_ylabel('Avg Ward Resistance (%)', color='#e67e22')
    
    # Combined Legend
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_res.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.savefig("stewardship_stratified_analysis.png", dpi=300)
    plt.show()

def run_policy_evaluation(model, num_agents=15, ward_side_len=15.0, num_episodes=30, w_beta_scale=1.0):
    num_days = 30
    env = AMRStewardshipEnv(num_agents=int(num_agents), ward_side_len=float(ward_side_len), w_beta_scale=float(w_beta_scale))
    
    # Standard Metrics
    daily_cured = np.zeros(num_days)
    daily_failed = np.zeros(num_days)
    episode_rewards, episode_resistance = [], []
    action_counts = {0: 0, 1: 0, 2: 0}
    total_ever_infected = 0
    total_agent_count = len(env.possible_agents)

    # Stratified Statistics
    sev_stats = {1: [0,0,0], 2: [0,0,0], 3: [0,0,0]}
    load_stats = {"low": [0,0,0], "mid": [0,0,0], "high": [0,0,0]} 
    rho_stats = {"low": [0,0,0], "mid": [0,0,0], "high": [0,0,0]}  
    
    # Temporal Dynamics
    temp_infections = np.zeros(num_days)
    temp_resistance = np.zeros(num_days)
    temp_actions = np.zeros((num_days, 3))

    for ep in range(num_episodes):
        obs_dict, _ = env.reset()
        ep_reward = 0
        cured_set, failed_set = set(), set()
        
        ever_infected = set(int(idx) for idx in np.where((env.S + env.R) >= 1e4)[0])
        active_infections = copy(ever_infected)
        
        for day in range(num_days):
            if not env.agents: break
            
            actions = {}
            for agent in env.agents:
                idx = int(agent.split('_')[-1])
                act, _ = model.predict(obs_dict[agent], deterministic=True)
                act = int(act)
                actions[agent] = act
                
                # Update Global/Temporal Counts
                action_counts[act] += 1
                temp_actions[day][act] += 1
                
                # Capture State Context
                curr_load = float(env.S[idx] + env.R[idx])
                curr_sev = int(env.severity[idx])
                ax_pos, ay_pos = env.position[idx]
                # Boundary check for grid indexing
                gx, gy = min(int(ax_pos), int(ward_side_len)-1), min(int(ay_pos), int(ward_side_len)-1)
                local_rho = float(env.ward_environmental_shedding[gx, gy])

                # Log Stratified Data
                sev_stats[curr_sev][act] += 1
                l_bin = "low" if curr_load < 1e6 else ("high" if curr_load > 1e8 else "mid")
                load_stats[l_bin][act] += 1
                r_bin = "low" if local_rho < 0.2 else ("high" if local_rho > 0.4 else "mid")
                rho_stats[r_bin][act] += 1

            obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            ep_reward += sum(rewards.values())
            
            # Record Daily Status
            temp_infections[day] += len(active_infections)
            total_loads_snap = env.S + env.R
            total_loads_snap[total_loads_snap == 0] = 1.0
            temp_resistance[day] += np.mean(env.R / total_loads_snap)
            
            total_load = env.S + env.R
            for i in range(env.num_agents):
                if i in active_infections:
                    if float(total_load[i]) < 1e4:
                        active_infections.remove(i)
                        cured_set.add(i)
                        daily_cured[day] += 1
                    elif float(total_load[i]) >= 1e8:
                        active_infections.remove(i)
                        failed_set.add(i)
                        daily_failed[day] += 1
                elif i not in cured_set and i not in failed_set and float(total_load[i]) >= 1e4:
                    active_infections.add(i)
                    ever_infected.add(i)
        
        total_ever_infected += len(ever_infected)
        total_loads_final = env.S + env.R
        total_loads_final[total_loads_final == 0] = 1.0 
        episode_resistance.append(float(np.mean(env.R / total_loads_final)))
        episode_rewards.append(float(ep_reward / total_agent_count))

    # Visualization
    temp_data = {
        'infections': temp_infections / num_episodes,
        'res': temp_resistance / num_episodes,
        'actions': temp_actions / num_episodes
    }
    plot_stewardship_analysis(sev_stats, load_stats, rho_stats, temp_data)

    # Console Summary
    total_actions = sum(action_counts.values())
    dist = {k: (v / max(1, total_actions)) * 100 for k, v in action_counts.items()}
    total_cured = float(np.sum(daily_cured))
    cure_rate = (total_cured / max(1, total_ever_infected)) * 100.0

    print(f"\n=== {num_episodes}-Episode Stratified Evaluation ===")
    print(f"Terminating Resistance Prevalence : {np.mean(episode_resistance)*100:.2f}%")
    print(f"Mean Reward (Per Agent)           : {np.mean(episode_rewards):.2f}")
    print(f"True Clinical Cure Rate           : {cure_rate:.2f}%")
    print(f"Action Distribution: Drug 0: {dist[0]:.1f}% | Drug 1: {dist[1]:.1f}% | Drug 2: {dist[2]:.1f}%")
    
    print("\n[Severity-Action Split %]")
    for s in [1, 2, 3]:
        row = sev_stats[s]
        tot = max(1, sum(row))
        print(f" Sev {s}: D0: {row[0]/tot*100:.1f}% | D1: {row[1]/tot*100:.1f}% | D2: {row[2]/tot*100:.1f}%")