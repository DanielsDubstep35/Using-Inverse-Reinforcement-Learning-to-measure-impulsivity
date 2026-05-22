import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from typing import Tuple, Dict, Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet

# Import your validated environment wrapper and normalization map directly
from Game_Code import DummyDrivingEnv, get_normalized_vector, NORM_BOUNDS

# ==============================================================================
# 1. LIVE UTILITY INTERFACE (Pure Functional Layer)
# ==============================================================================

def pure_extract_live_trajectories(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms a raw live player dataframe into normalized trajectory states and discrete tokens."""
    df = df.sort_values(by="time")
    eps = 1e-4
    
    raw_speeds = df["speed"].to_numpy()
    raw_distances = df["distance"].to_numpy()
    raw_lanes = df["lane"].to_numpy()
    
    obs_matrix = np.array([
        get_normalized_vector(s, d, l) for s, d, l in zip(raw_speeds, raw_distances, raw_lanes)
    ], dtype=np.float32)
    
    actions = []
    for i in range(1, len(obs_matrix)):
        prev_spd, prev_lane = raw_speeds[i - 1], raw_lanes[i - 1]
        curr_spd, curr_lane = raw_speeds[i], raw_lanes[i]
        
        if (prev_lane - curr_lane) > eps:
            actions.append(3)  # LANE_LEFT
        elif (curr_lane - prev_lane) > eps:
            actions.append(4)  # LANE_RIGHT
        elif (curr_spd - prev_spd) > eps:
            actions.append(2)  # FASTER
        elif (prev_spd - curr_spd) > eps:
            actions.append(0)  # SLOWER
        else:
            actions.append(1)  # MAINTAIN
            
    return obs_matrix[:-1], np.array(actions, dtype=np.int64)


def pure_compute_metrics(
    driver_rewards: np.ndarray, 
    baseline_rewards: np.ndarray,
    mesh_speed: np.ndarray,
    mesh_dist: np.ndarray,
    grid_rewards: np.ndarray
) -> Dict[str, Any]:
    """Pure mathematical transformation to generate unified profile scores."""
    # Paradigm A Calculation: Fixed Grid Impulsivity
    rewards_grid = grid_rewards.reshape(mesh_speed.shape)
    high_risk_mask = (mesh_speed > 90.0) & (mesh_dist < 50.0)
    low_risk_mask = (mesh_speed > 40.0) & (mesh_speed < 70.0) & (mesh_dist > 200.0)
    
    risk_yield = np.mean(rewards_grid[high_risk_mask]) if np.any(high_risk_mask) else float(rewards_grid.min())
    safe_yield = np.mean(rewards_grid[low_risk_mask]) if np.any(low_risk_mask) else float(rewards_grid.mean())
    impulsivity_score = float(risk_yield - safe_yield)
    
    # Paradigm B Calculation: Anomaly Detection vs Blended Safe Driver
    step_deltas = driver_rewards - baseline_rewards
    max_safety_plunge = float(np.min(step_deltas))
    outlier_threshold = np.mean(baseline_rewards) - (2 * np.std(baseline_rewards))
    anomaly_ratio = float(np.sum(driver_rewards < outlier_threshold) / len(driver_rewards))
    
    return {
        "impulsivity_score": impulsivity_score,
        "max_safety_plunge": max_safety_plunge,
        "anomaly_ratio": anomaly_ratio,
        "risk_classification": "High Risk" if (impulsivity_score > 0.1 or anomaly_ratio > 0.15) else "Normal/Safe"
    }


# ==============================================================================
# 2. RUNTIME PIPELINE ORCHESTRATOR (Imperative Shell Layer)
# ==============================================================================

class LiveSessionProfiler:
    """Manages on-the-fly training, scoring, and cleanup of transient driver models."""
    def __init__(self, blended_baseline_path: str, temp_output_dir: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blended_baseline_path = blended_baseline_path
        self.temp_output_dir = temp_output_dir
        os.makedirs(temp_output_dir, exist_ok=True)
        
    def _instantiate_airl(self, venv: SubprocVecEnv) -> Tuple[AIRL, BasicRewardNet]:
        """Builds standard matching architecture structures."""
        reward_net = BasicRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            use_state=True, use_action=True, hid_sizes=(512, 512, 512)
        )
        gen_algo = PPO(
            "MlpPolicy", env=venv, verbose=0, n_steps=512, batch_size=256,
            learning_rate=5e-4, device=self.device
        )
        trainer = AIRL(
            demonstrations=None, demo_batch_size=256, venv=venv,
            gen_algo=gen_algo, reward_net=reward_net, allow_variable_horizon=True
        )
        return trainer, reward_net

    def run_post_session_pipeline(self, active_player_csv: str) -> Dict[str, Any]:
        """Trains a reward network for the current player, scores it, and clears files."""
        print(f"\n[PIPELINE] New player log detected: {active_player_csv}")
        
        # 1. Ingest telemetry logs
        df = pd.read_csv(active_player_csv)
        live_obs, live_acts = pure_extract_live_trajectories(df)
        
        # Convert to imitation format
        from imitation.data.types import TrajectoryWithRew
        player_trajectory = [TrajectoryWithRew(
            obs=live_obs, acts=live_acts, infos=None, terminal=True, rews=np.zeros(len(live_acts), dtype=np.float32)
        )]
        
        # 2. Spin up isolated local background environment processes
        num_envs = 8
        venv = SubprocVecEnv([lambda: DummyDrivingEnv() for _ in range(num_envs)])
        
        temp_model_path = os.path.join(self.temp_output_dir, "transient_player_model.pt")
        
        try:
            # 3. Fast localized adversarial training pass
            airl_trainer, reward_net = self._instantiate_airl(venv)
            airl_trainer.optimizer = optim.Adam(reward_net.parameters(), lr=1e-4)
            airl_trainer.set_demonstrations(player_trajectory)
            
            print("[PIPELINE] Optimizing driver-specific behavioral reward networks...")
            airl_trainer.train(total_timesteps=100_000)  # High-speed local convergence threshold
            torch.save(reward_net.state_dict(), temp_model_path)
            
            # 4. Evaluation Calculations
            # Load your collective master baseline profile
            safe_net = BasicRewardNet(venv.observation_space, venv.action_space, use_state=True, use_action=True, hid_sizes=(512, 512, 512))
            safe_net.load_state_dict(torch.load(self.blended_baseline_path, map_location=self.device))
            safe_net.to(self.device).eval()
            reward_net.eval()
            
            # Form inference arrays
            t_obs = torch.as_tensor(live_obs, dtype=torch.float32, device=self.device)
            t_acts = torch.as_tensor(live_acts, dtype=torch.int64, device=self.device)
            t_next = torch.zeros_like(t_obs)
            t_done = torch.zeros(len(t_obs), dtype=torch.bool, device=self.device)
            
            with torch.no_grad():
                driver_rewards = reward_net(t_obs, t_acts, t_next, t_done).cpu().numpy().flatten()
                baseline_rewards = safe_net(t_obs, t_acts, t_next, t_done).cpu().numpy().flatten()
                
                # Paradigm A Grid Sweep
                s_space = np.linspace(NORM_BOUNDS["speed"]["min"], NORM_BOUNDS["speed"]["max"], 30)
                d_space = np.linspace(NORM_BOUNDS["distance"]["min"], NORM_BOUNDS["distance"]["max"], 30)
                m_speed, m_dist = np.meshgrid(s_space, d_space)
                
                flat_obs = np.array([get_normalized_vector(s, d, 4.0) for s, d in zip(m_speed.flatten(), m_dist.flatten())], dtype=np.float32)
                t_flat_obs = torch.as_tensor(flat_obs, dtype=torch.float32, device=self.device)
                t_flat_acts = torch.ones(len(flat_obs), dtype=torch.int64, device=self.device)
                
                grid_rewards = reward_net(t_flat_obs, t_flat_acts, torch.zeros_like(t_flat_obs), torch.zeros(len(t_flat_obs), dtype=torch.bool, device=self.device)).cpu().numpy().flatten()
            
            # Run pure processing scoring transformations
            metrics = pure_compute_metrics(driver_rewards, baseline_rewards, m_speed, m_dist, grid_rewards)
            return metrics
            
        finally:
            # 5. Resource Takedown & File Cleanup Firewall
            venv.close()
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                print("[PIPELINE] Cleaned up transient player model weights from memory workspace.")


# ==============================================================================
# 3. AMENDMENT TARGET FOR YOUR MAIN SOCKET LOOP (Game_Code.py Integration Hook)
# ==============================================================================

def unreal_engine_socket_listener_loop():
    """Conceptual loop layout showing exactly where this hooks into your socket file."""
    BLENDED_MODEL_PATH = "airl_reward_model_production_theirs.pth"
    TEMP_DIR = "./temp_session_cache"
    
    # Instantiate the processing pipeline asset
    pipeline = LiveSessionProfiler(blended_baseline_path=BLENDED_MODEL_PATH, temp_output_dir=TEMP_DIR)
    
    # Mocking standard continuous game listening architecture loop logic
    while True:
        # data = conn.recv()
        # if message_indicates_end_of_episode:
        
        # Example triggering event:
        player_data_csv = "latest_runtime_session_telemetry.csv"
        
        try:
            # Run the combined evaluation pass
            computed_metrics = pipeline.run_post_session_pipeline(player_data_csv)
            
            # Format as clean JSON string matching your socket configuration expectations
            json_payload = {
                "type": "profiling_result",
                "impulsivity_score": computed_metrics["impulsivity_score"],
                "safety_deviation_plunge": computed_metrics["max_safety_plunge"],
                "anomaly_ratio": computed_metrics["anomaly_ratio"],
                "classification": computed_metrics["risk_classification"]
            }
            print(f"[SOCKET SEND] Telemetry ready for Unreal dashboard: {json_payload}")
            # conn.sendall((json.dumps(json_payload) + "\n").encode('utf-8'))
            
        except Exception as e:
            print(f"Error handling post-session profiling pipeline: {e}")
        break

if __name__ == "__main__":
    # Internal execution test route loop verification entry point
    unreal_engine_socket_listener_loop()