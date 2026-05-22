import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
import gymnasium as class_gymnasium

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper

# Import our safe vectorized simulation interface wrappers
from Game_Code import DummyDrivingEnv, get_normalized_vector
from Evaluate_Driver_Profile import (
    DriverEvaluationShell, 
    pure_generate_evaluation_meshgrid, 
    pure_calculate_grid_impulsivity, 
    pure_calculate_anomaly_deviation
)

# Enforce clean typing specifications
class UnrealGymInterface(DummyDrivingEnv):
    """Gymnasium bridging wrapper ensuring normalized bounds match our mathematical core."""
    def __init__(self, df=None):
        super().__init__(df)
        self.observation_space = class_gymnasium.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = class_gymnasium.spaces.Discrete(5)

    def _get_obs(self) -> np.ndarray:
        if not hasattr(self, 'raw_state') or self.raw_state is None:
            self.raw_state = [50.0, 250.0, 4.0]
        obs = get_normalized_vector(*self.raw_state)
        return np.clip(np.asarray(obs, dtype=np.float32).flatten(), 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._get_obs(), {}

    def step(self, action):
        out = super().step(action)
        obs = self._get_obs()
        reward = float(out[1])
        done = bool(out[2]) if len(out) == 4 else bool(out[2] or out[3])
        return obs, reward, done, False, {}

# ==============================================================================
# 1. LIVE RECONSTRUCTION ENGINE (Functional Core Operations)
# ==============================================================================

def pure_parse_unreal_telemetry(df: pd.DataFrame) -> List[TrajectoryWithRew]:
    """Converts a live stream of Unreal Engine telemetry rows into strict discrete trajectories."""
    df = df.sort_values(by="time")
    if len(df) < 3:
        raise ValueError("Insufficient tracking states found within live Unreal sequence.")
        
    raw_speeds = df["speed"].to_numpy()
    raw_distances = df["distance"].to_numpy()
    raw_lanes = df["lane"].to_numpy()

    # Apply global scale normalization maps
    obs_matrix = np.array([
        get_normalized_vector(s, d, l) for s, d, l in zip(raw_speeds, raw_distances, raw_lanes)
    ], dtype=np.float32)

    # Reconstruct true historical action choices sequentially
    eps = 1e-4
    actions = []
    for i in range(1, len(obs_matrix)):
        prev_s, prev_l = raw_speeds[i - 1], raw_lanes[i - 1]
        curr_s, curr_l = raw_speeds[i], raw_lanes[i]

        if (prev_l - curr_l) > eps:
            actions.append(3)   # LANE_LEFT
        elif (curr_l - prev_l) > eps:
            actions.append(4)   # LANE_RIGHT
        elif (curr_s - prev_s) > eps:
            actions.append(2)   # FASTER
        elif (prev_s - curr_s) > eps:
            actions.append(0)   # SLOWER
        else:
            actions.append(1)   # MAINTAIN

    discrete_actions = np.array(actions, dtype=np.int64)
    
    trajectory = TrajectoryWithRew(
        obs=obs_matrix,
        acts=discrete_actions,
        infos=np.array([{} for _ in range(len(discrete_actions))]),
        terminal=True,
        rews=np.zeros(len(discrete_actions), dtype=np.float32)
    )
    return [trajectory]

# ==============================================================================
# 2. RUNTIME ORCHESTRATION PIPELINE (Imperative Shell Operations)
# ==============================================================================

def run_live_pipeline_evaluation(
    unreal_csv_path: str,
    output_model_path: str,
    blended_baseline_path: str,
    use_fixed_grid_comparison: bool,
    num_envs: int = 8,
    training_steps: int = 150_000
):
    """
    Executes automated training over a fresh human gameplay sequence, saves weights,
    and runs a dual-paradigm evaluation slice immediately.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nLAUNCHING LIVE UNREAL REWARD MODEL PIPELINE\n{'='*70}")
    
    # 1. Ingest fresh player actions from your local file asset paths
    print(f"[1/4] Ingesting live runtime trajectory asset from: {unreal_csv_path}")
    if not os.path.exists(unreal_csv_path):
        raise FileNotFoundError(f"No recent live gameplay track found at {unreal_csv_path}. Go play in Unreal first!")
        
    live_df = pd.read_csv(unreal_csv_path)
    unreal_trajectories = pure_parse_unreal_telemetry(live_df)
    print(f" ✓ Reconstructed human profile footprint containing {len(unreal_trajectories[0].acts)} sequential decisions.")

    # 2. Spin up localized parallel processing pools 
    print(f"[2/4] Initializing training environments across {num_envs} virtual cores...")
    def make_env():
        return RolloutInfoWrapper(UnrealGymInterface())
        
    parallel_venv = SubprocVecEnv([make_env for _ in range(num_envs)])
    parallel_venv.seed(42)

    try:
        # 3. Instantiate and train your custom AIRL configuration architecture
        print(f"[3/4] Optimizing personalized reward network weights over live sequence...")
        reward_net = BasicRewardNet(
            observation_space=parallel_venv.observation_space,
            action_space=parallel_venv.action_space,
            use_state=True,
            use_action=True,
            hid_sizes=(512, 512, 512) # Match "THEIRS" specifications exactly
        )

        gen_algo = PPO(
            "MlpPolicy", env=parallel_venv, verbose=0, n_steps=512, batch_size=256,
            learning_rate=5e-4, ent_coef=0.05, device=device
        )

        airl_trainer = AIRL(
            demonstrations=unreal_trajectories,
            demo_batch_size=min(128, len(unreal_trajectories[0].acts)),
            venv=parallel_venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            allow_variable_horizon=True
        )
        
        airl_trainer.optimizer = optim.Adam(reward_net.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Fast fine-tuning optimization pass
        airl_trainer.train(total_timesteps=training_steps)
        
        # Save model weights cleanly
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        torch.save(reward_net.state_dict(), output_model_path)
        print(f" ✓ Customized Reward Network saved directly to: {output_model_path}")

    finally:
        parallel_venv.close()

    # 4. Invoke the Driver Evaluation Shell 
    print(f"[4/4] Executing comparative paradigm evaluations...")
    evaluator = DriverEvaluationShell(device="cpu")
    
    if use_fixed_grid_comparison:
        print("\n>>> RESULTS paradigm A: FIXED SCIENTIFIC GRID EXPLORATION <<<")
        obs_grid, act_grid, mesh_speed, mesh_dist = pure_generate_evaluation_meshgrid(resolution=40)
        grid_rewards = evaluator.evaluate_inference(reward_net, obs_grid, act_grid)
        metrics = pure_calculate_grid_impulsivity(grid_rewards, mesh_speed, mesh_dist)
        
        print(f" - Personal Driving Impulsivity Delta:   {metrics['impulsivity_score']:.4f}")
        print(f" - Latent Preference Speed Target:       {metrics['preferred_speed_mph']:.2f} MPH")
        print(f" - Latent Preference Safe Headway Gap:   {metrics['preferred_safety_gap_ft']:.2f} FT")
    else:
        print("\n>>> RESULTS paradigm B: BLENDED ANOMALY BASELINE DEVIATION <<<")
        if not os.path.exists(blended_baseline_path):
            raise FileNotFoundError(f"Master safe model profile missing at {blended_baseline_path}")
            
        safe_net = evaluator.load_reward_network(blended_baseline_path)
        live_obs = unreal_trajectories[0].obs[:-1]
        live_acts = unreal_trajectories[0].acts
        
        driver_rewards = evaluator.evaluate_inference(reward_net, live_obs, live_acts)
        baseline_rewards = evaluator.evaluate_inference(safe_net, live_obs, live_acts)
        metrics = pure_calculate_anomaly_deviation(driver_rewards, baseline_rewards)
        
        print(f" - Consolidated Behavioral Deviation Score: {metrics['deviation_score']:.4f}")
        print(f" - Maximum Negative Safety Plunge Value:   {metrics['max_safety_plunge']:.4f}")
        print(f" - Outlier Anomaly Step Horizon Ratio:     {metrics['anomaly_ratio']*100:.2f}%")

    print(f"\n{'='*70}\nPIPELINE PROCESSING SEQUENCE TERMINATED SUCCESSFULY\n{'='*70}")


if __name__ == "__main__":
    # Absolute paths pointing directly to your local workspace files
    LIVE_UNREAL_CSV      = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Original_Data\master_airl_expert_dataset.csv"
    FRESH_PLAYER_MODEL   = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Reward Networks\fresh_unreal_player_model.pt"
    BLENDED_MASTER_MODEL = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Reward Networks\airl_reward_model_production_theirs.pth"

    # Toggle this flag to test both paradigms!
    run_live_pipeline_evaluation(
        unreal_csv_path=LIVE_UNREAL_CSV,
        output_model_path=FRESH_PLAYER_MODEL,
        blended_baseline_path=BLENDED_MASTER_MODEL,
        use_fixed_grid_comparison=False,  # Flipped to False to match Paradigm B anomaly detection against your baseline
        num_envs=8,
        training_steps=100_000            # Swift pass for presentation demonstrations
    )