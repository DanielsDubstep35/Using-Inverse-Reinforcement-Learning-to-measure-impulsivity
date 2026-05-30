# Imports

# General
import socket
import json
import os
import pandas as pd
import numpy as np
import torch
import tqdm
import matplotlib
matplotlib.use('Agg')  # Prevents crashes if no display is connected
import matplotlib.pyplot as plt
import seaborn as sns

# Gym environment
import gymnasium as gym
from gymnasium import spaces

# Reinforcement Learning
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
import imitation.data.types as types
from stable_baselines3.common.logger import configure

# Global structural normalization boundaries
NORM_BOUNDS = {
    "speed": {"min": 0.0, "max": 120.0},
    "distance": {"min": 0.0, "max": 500.0}, # Updated to match Unreal 500 Max
    "lane": {"min": -4.0, "max": 16.0},
}

# ====================
# Driving Environment: Pure Functions
# ====================

def create_state(speed: float, distance: float, lane: float):
    return np.array([speed, distance, lane], dtype=np.float32)

def move_step_count(step_count: int, by_value: int):
    return step_count + by_value

def is_terminated(dist: float, step_count: int):
    return dist <= 0 or step_count >= 100

def get_normalized_vector(speed, distance, lane):
    s = (speed - NORM_BOUNDS["speed"]["min"]) / (
        NORM_BOUNDS["speed"]["max"] - NORM_BOUNDS["speed"]["min"]
    )
    d = (distance - NORM_BOUNDS["distance"]["min"]) / (
        NORM_BOUNDS["distance"]["max"] - NORM_BOUNDS["distance"]["min"]
    )
    l = (lane - NORM_BOUNDS["lane"]["min"]) / (
        NORM_BOUNDS["lane"]["max"] - NORM_BOUNDS["lane"]["min"]
    )
    return np.array(
        [np.clip(s, 0.0, 1.0), np.clip(d, 0.0, 1.0), np.clip(l, 0.0, 1.0)],
        dtype=np.float32,
    )

# ====================
# Fake Driving Environment
# ====================
class DummyDrivingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5) 
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_speed, raw_distance, raw_lane = 20.0, 105.0, 1.0
        self.state = create_state(raw_speed, raw_distance, raw_lane)
        self.step_count = 0
        
        normalized_state = get_normalized_vector(raw_speed, raw_distance, raw_lane)
        return normalized_state, {}

    def step(self, action):
        self.step_count = move_step_count(self.step_count, 1)
        speed, dist, lane = self.state
        
        self.state = create_state(speed, dist, lane)
        normalized_state = get_normalized_vector(speed, dist, lane)
        
        terminated = is_terminated(dist, self.step_count)
        return normalized_state, 0.0, terminated, False, {}


# ====================
# Unreal Engine Bridge
# ====================
class UnrealPostGameBridge:
    def __init__(self, hostname="127.0.0.1", port=3000, blended_baseline_path="airl_reward_model_production_theirs.pth"):
        self.host = hostname
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.trajectory_buffer = []  
        self.episode_count = 0
        self.blended_baseline_path = blended_baseline_path

    def start_server(self):
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(1)
            print(f"Ready for Game Session on {self.host}:{self.port}...")

            while True:
                conn, addr = self.sock.accept()
                print(f"\nGame Client Connected: {addr}")
                with conn:
                    self.handle_session(conn)
                print("Session Ended. Waiting for next player...")
                self.trajectory_buffer = []  

        except Exception as e:
            print(f"Server Error: {e}")

    def handle_session(self, conn):
        buffer = ""
        step_idx = 0
        while True:
            try:
                data = conn.recv(1048576)
                if not data:
                    break

                buffer += data.decode("utf-8")

                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    if not message.strip():
                        continue

                    packet = json.loads(message)
                    msg_type = packet.get("type", "state")

                    if msg_type == "state":
                        if isinstance(packet.get("data"), list):
                            if len(packet["data"]) >= 3:
                                raw_data = packet["data"]
                                speed = float(raw_data[0])
                                # FIX: Removed dividing by 100 to protect scale mapping to 500
                                dist = float(raw_data[1])  
                                lane = float(raw_data[2])
                                
                                self.trajectory_buffer.append([float(step_idx), speed, dist, lane])
                                step_idx += 1

                    elif msg_type == "level_complete":
                        print(f"Level Complete! Processing pipeline matrix on {len(self.trajectory_buffer)} frames...")
                        
                        df_trajectory = pd.DataFrame(
                            self.trajectory_buffer, 
                            columns=["time", "speed", "distance", "lane"]
                        )
                        
                        score, weights, extra_metrics = self.run_airl_training(df_trajectory)

                        result = {
                            "type": "result",
                            "impulsivity_score": float(score),
                            "details": f"Spd:{weights[0]:.2f} Safe:{weights[1]:.2f}",
                            "paradigm_a_grid_impulsivity": float(extra_metrics["paradigm_a_grid_impulsivity"]),
                            "paradigm_b_anomaly_ratio": float(extra_metrics["paradigm_b_anomaly_ratio"]),
                            "paradigm_b_safety_plunge": float(extra_metrics["paradigm_b_max_plunge"])
                        }
                        
                        conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
                        print(f"Sent Final Analytical Profile Payload back to Unreal dashboard: {result}\n")
                        return

            except Exception as e:
                import traceback
                print(f"Session Error: {e}")
                traceback.print_exc()
                break

    def run_airl_training(self, df_trajectory: pd.DataFrame):
        ep_length = len(df_trajectory)
        if ep_length < 10:
            print("Not enough tracking frames captured within dataframe to train weights.")
            return 0.0, [0.0, 0.0, 0.0], {"paradigm_a_grid_impulsivity": 0, "paradigm_b_anomaly_ratio": 0, "paradigm_b_max_plunge": 0}

        self.episode_count += 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        expert_obs, acts, trajectory = format_expert_data_from_dataframe(df_trajectory)
        
        raw_venv = DummyVecEnv([lambda: DummyDrivingEnv()])
        venv = VecNormalize(raw_venv, norm_obs=False, norm_reward=False) 
        
        temp_result = build_airl_trainer(venv, trajectory)
        if temp_result is None:
            print("CRITICAL ERROR: build_airl_trainer returned None.")
            return 0.0, [0.0, 0.0, 0.0], {"paradigm_a_grid_impulsivity": 0, "paradigm_b_anomaly_ratio": 0, "paradigm_b_max_plunge": 0}

        learner, reward_net, airl_trainer = temp_result
        reward_net.to(device)
        
        print(f"\nTraining AIRL optimization loops for Participant Episode #{self.episode_count}...")
        for i in range(2): 
            airl_trainer.train(total_timesteps=4096)
            acc, loss = get_disc_stats(airl_trainer, expert_obs, acts)
            print(f" [Optimizer Step {i+1}] Discriminator Loss: {loss:.4f} | Accuracy: {acc*100:.1f}%")

        transient_model_path = f"transient_reward_model_ep{self.episode_count}.pth"
        torch.save(reward_net.state_dict(), transient_model_path)

        weights = extract_weights(reward_net)
        score = calculate_impulsivity_score(reward_net, expert_obs)
        heatmap = generate_reward_heatmap(reward_net, self.episode_count)
        
        extra_metrics = {"paradigm_a_grid_impulsivity": 0.0, "paradigm_b_anomaly_ratio": 0.0, "paradigm_b_max_plunge": 0.0}
        
        if os.path.exists(self.blended_baseline_path):
            try:
                safe_net = BasicRewardNet(venv.observation_space, venv.action_space, use_state=True, use_action=False, use_next_state=False, use_done=False, hid_sizes=(512, 512, 512))
                safe_net.load_state_dict(torch.load(self.blended_baseline_path, map_location=device))
                safe_net.to(device).eval()
                reward_net.eval()

                t_obs = torch.as_tensor(expert_obs[:-1], dtype=torch.float32, device=device)
                t_acts = torch.as_tensor(acts, dtype=torch.int64, device=device)
                
                with torch.no_grad():
                    driver_rewards = reward_net.predict_processed(t_obs, t_acts, t_obs, torch.zeros(len(t_obs), dtype=torch.bool, device=device)).cpu().numpy().flatten()
                    baseline_rewards = safe_net.predict_processed(t_obs, t_acts, t_obs, torch.zeros(len(t_obs), dtype=torch.bool, device=device)).cpu().numpy().flatten()
                    
                    s_space = np.linspace(NORM_BOUNDS["speed"]["min"], NORM_BOUNDS["speed"]["max"], 25)
                    d_space = np.linspace(NORM_BOUNDS["distance"]["min"], NORM_BOUNDS["distance"]["max"], 25)
                    m_speed, m_dist = np.meshgrid(s_space, d_space)
                    
                    flat_grid_obs = np.array([get_normalized_vector(s, d, 4.0) for s, d in zip(m_speed.flatten(), m_dist.flatten())], dtype=np.float32)
                    t_grid_obs = torch.as_tensor(flat_grid_obs, dtype=torch.float32, device=device)
                    t_grid_acts = torch.ones(len(flat_grid_obs), dtype=torch.int64, device=device)
                    grid_rewards = reward_net.predict_processed(t_grid_obs, t_grid_acts, t_grid_obs, torch.zeros(len(t_grid_obs), dtype=torch.bool, device=device)).cpu().numpy().reshape(m_speed.shape)
                    
                    high_risk_mask = (m_speed > 90.0) & (m_dist < 50.0)
                    low_risk_mask = (m_speed > 40.0) & (m_speed < 70.0) & (m_dist > 200.0)
                    extra_metrics["paradigm_a_grid_impulsivity"] = float(np.mean(grid_rewards[high_risk_mask]) - np.mean(grid_rewards[low_risk_mask])) if np.any(high_risk_mask) and np.any(low_risk_mask) else 0.0

                    deltas = driver_rewards - baseline_rewards
                    extra_metrics["paradigm_b_max_plunge"] = float(np.min(deltas))
                    outlier_threshold = np.mean(baseline_rewards) - (2 * np.std(baseline_rewards))
                    extra_metrics["paradigm_b_anomaly_ratio"] = float(np.sum(driver_rewards < outlier_threshold) / len(driver_rewards))
                    
            except Exception as profile_err:
                print(f"[PIPELINE ERROR] Multi-paradigm profiling pass calculation failed: {profile_err}")
        else:
            print(f"[WARNING] Blended Master Baseline Model not found at '{self.blended_baseline_path}'. Skipping Paradigm evaluation comparisons.")

        raw_venv.close()
        if os.path.exists(transient_model_path):
            os.remove(transient_model_path)
            print(f"[CLEANUP] Deleted transient model layer artifact tracking weights file: {transient_model_path}")

        return score, weights, extra_metrics


# ====================
# Useful Functions
# ====================
def map_raw_to_discrete(current_obs, prev_obs):
    curr_spd, curr_dist, curr_lane = current_obs
    prev_spd, prev_dist, prev_lane = prev_obs
    eps = 1e-4
    
    if (prev_lane - curr_lane) > eps: return 3   # LANE_LEFT
    if (curr_lane - prev_lane) > eps: return 4   # LANE_RIGHT
    
    if curr_spd > prev_spd: return 1             # FASTER
    if curr_spd < prev_spd: return 2             # SLOWER
    
    return 0                                     # MAINTAIN/STAY


def get_reward_for_state(reward_net, speed, dist, lane):
    device = next(reward_net.parameters()).device
    norm_state = get_normalized_vector(speed, dist, lane)
    state = torch.tensor([norm_state], dtype=torch.float32, device=device)
    act = torch.zeros((1,), dtype=torch.long, device=device) 
    done = torch.zeros(1, dtype=torch.bool, device=device)

    with torch.no_grad():
        rew = reward_net.forward(state, act, state, done)
    return rew.item()


def calculate_impulsivity_score(reward_net, expert_obs):
    device = next(reward_net.parameters()).device
    reward_net.eval()

    with torch.no_grad():
        norm_safe = get_normalized_vector(20.0, 105.0, 1.0)
        norm_impulsive = get_normalized_vector(120.0, 0.0, 1.0)
        
        safe_state = torch.tensor([norm_safe], device=device, dtype=torch.float32)
        impulsive_state = torch.tensor([norm_impulsive], device=device, dtype=torch.float32)
        
        act = torch.zeros((1,), dtype=torch.long, device=device)
        done = torch.zeros(1, dtype=torch.bool, device=device)

        r_min = reward_net.predict_processed(safe_state, act, safe_state, done).item()
        r_max = reward_net.predict_processed(impulsive_state, act, impulsive_state, done).item()

        exp_tensor = torch.tensor(expert_obs, dtype=torch.float32, device=device)
        exp_acts = torch.zeros((len(exp_tensor),), dtype=torch.long, device=device)
        exp_dones = torch.zeros(len(exp_tensor), dtype=torch.bool, device=device)
        
        r_expert = reward_net.predict_processed(exp_tensor, exp_acts, exp_tensor, exp_dones).mean().item()

    if abs(r_max - r_min) < 1e-5:
        percentage = 0.0
    else:
        percentage = ((r_expert - r_min) / (r_max - r_min)) * 100.0

    final_score = np.clip(percentage, 0.0, 100.0)
    return round(float(final_score), 2)


def format_expert_data_from_dataframe(df_trajectory: pd.DataFrame):
    raw_speeds = df_trajectory["speed"].to_numpy()
    raw_distances = df_trajectory["distance"].to_numpy()
    raw_lanes = df_trajectory["lane"].to_numpy()
    
    expert_obs = np.array([
        get_normalized_vector(s, d, l) for s, d, l in zip(raw_speeds, raw_distances, raw_lanes)
    ], dtype=np.float32)
    
    discrete_actions = []
    for i in range(1, len(df_trajectory)):
        prev_raw = [raw_speeds[i-1], raw_distances[i-1], raw_lanes[i-1]]
        curr_raw = [raw_speeds[i], raw_distances[i], raw_lanes[i]]
        action = map_raw_to_discrete(curr_raw, prev_raw)
        discrete_actions.append(action)
    
    acts = np.array(discrete_actions, dtype=np.int64)
    infos = np.array([{}] * len(acts))
    
    trajectory = types.Trajectory(
        obs=expert_obs, 
        acts=acts, 
        infos=infos, 
        terminal=True
    )
    
    return expert_obs, acts, trajectory


def build_airl_trainer(venv, trajectory):
    # FIX: Increased n_steps to 2048 to allow a healthy mini-batch split of 512
    learner = PPO(
        env=venv, policy=MlpPolicy, batch_size=512, n_steps=2048,
        ent_coef=0.1, learning_rate=5e-4, n_epochs=5, verbose=0,
    )
    
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True, use_action=False, use_next_state=False, use_done=False,
        hid_sizes=(512, 512, 512),
    )
    
    airl_trainer = AIRL(
        demonstrations=[trajectory], demo_batch_size=32,
        gen_replay_buffer_capacity=2000, n_disc_updates_per_round=1,
        venv=venv, gen_algo=learner, reward_net=reward_net,
        allow_variable_horizon=True,
    )
    
    return learner, reward_net, airl_trainer


def extract_weights(reward_net):
    reward_net.eval()
    device = next(reward_net.parameters()).device
    
    norm_base = get_normalized_vector(70.0, 50.0, 1.0)
    base_state = torch.tensor([norm_base], device=device, dtype=torch.float32, requires_grad=True)
    
    try:
        act = torch.zeros((1,), dtype=torch.long, device=device)
        done = torch.tensor([False], device=device)
        rew = reward_net.forward(base_state, act, base_state, done)
        rew.backward()
        sensitivities = base_state.grad.detach().cpu().numpy().flatten()
        return sensitivities
    except Exception as e:
        print(f"Extraction Error: {e}")
        return np.array([0.0, 0.0, 0.0])


def evaluate_expert(reward_net, expert_obs, acts):
    device = next(reward_net.parameters()).device
    with torch.no_grad():
        states = torch.tensor(expert_obs[:-1], dtype=torch.float32, device=device)
        actions = torch.tensor(acts, dtype=torch.long, device=device)
        dones = torch.zeros(len(acts), dtype=torch.bool, device=device)
        return reward_net.predict_processed(states, actions, states, dones).mean().item()


def get_disc_stats(airl_trainer, expert_obs, acts):
    reward_net = airl_trainer._reward_net
    policy = airl_trainer.gen_algo.policy
    device = next(reward_net.parameters()).device

    states = torch.tensor(expert_obs[:-1], dtype=torch.float32, device=device)
    actions = torch.tensor(acts, dtype=torch.long, device=device)
    dones = torch.zeros(len(acts), dtype=torch.bool, device=device)

    with torch.no_grad():
        r = reward_net.forward(states, actions, states, dones)
        dist = policy.get_distribution(states)
        log_prob = dist.log_prob(actions)

        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(-1)

        logits = r - log_prob
        probs = torch.sigmoid(logits)
        acc = ((probs > 0.5).float().mean()).item()
        loss = torch.nn.functional.binary_cross_entropy(probs, torch.ones_like(probs))

    return acc, loss.item()


def generate_reward_heatmap(reward_net, episode_num=0):
    speeds = np.linspace(20, 120, 10)
    distances = np.linspace(0, 500, 10)  # FIX: Scaled to 500 to match new bounds
    device = next(reward_net.parameters()).device
    reward_net.eval()
    
    heatmap_data = []
    with torch.no_grad():
        for d in reversed(distances):
            row_rewards = []
            for s in speeds:
                norm_v = get_normalized_vector(s, d, 1.0)
                state = torch.tensor([norm_v], dtype=torch.float32, device=device)
                act = torch.zeros((1,), dtype=torch.long, device=device)
                done = torch.zeros(1, dtype=torch.bool, device=device)
                rew = reward_net.predict_processed(state, act, state, done)
                row_rewards.append(rew.item())
            heatmap_data.append(row_rewards)
            
    data_array = np.array(heatmap_data)
    plt.figure(figsize=(8, 6))
    sns.heatmap(data_array, annot=True, xticklabels=np.round(speeds).astype(int), yticklabels=np.round(distances[::-1]).astype(int), cmap="coolwarm")
    plt.title(f"AIRL Reward Heatmap - Episode {episode_num}")
    plt.savefig(f"reward_heatmap_ep{episode_num}.png")
    plt.close()
    return data_array


# ====================
# Code Start
# ====================
if __name__ == "__main__":
    server = UnrealPostGameBridge(blended_baseline_path=r"src\PythonAIRL\Reward Networks\airl_reward_model_production_theirs_5Epochs.pth")
    server.start_server()