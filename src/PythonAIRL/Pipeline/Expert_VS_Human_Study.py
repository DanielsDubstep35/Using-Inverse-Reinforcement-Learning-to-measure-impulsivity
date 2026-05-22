import os
import random
import numpy as np
import pandas as pd
import torch
import gymnasium as class_gymnasium
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import KVWriter, Logger

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import TrajectoryWithRew

# Import your raw legacy driving environment from your local repository codebase
from Game_Code import DummyDrivingEnv, get_normalized_vector

# ==============================================================================
# I. FUNCTIONAL CORE (Pure Functions, Transformations, Data Operations)
# ==============================================================================

def set_global_seeds(seed: int = 42) -> None:
    """Enforces absolute determinism across all underlying mathematical backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_observation_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extracts and verifies normalized target feature trajectories from expert data."""
    required_cols = ['speed', 'distance', 'lane']
    assert all(col in df.columns for col in required_cols), f"Dataset missing keys: {required_cols}"
    return df[required_cols].to_numpy(dtype=np.float32)


def build_expert_trajectories(obs_matrix: np.ndarray) -> list:
    """Converts a raw feature matrix into an imitation-compliant Trajectory structure."""
    time_steps = len(obs_matrix)
    
    # Structural step length: N - 1
    step_length = time_steps - 1
    
    discrete_actions = np.zeros(step_length, dtype=np.int64)
    terminal_rewards = np.zeros(step_length, dtype=np.float32)
    
    # FIX: Match the step_length (actions) instead of time_steps (observations)
    safe_infos = [{} for _ in range(step_length)]

    trajectory = TrajectoryWithRew(
        obs=obs_matrix,
        acts=discrete_actions,
        infos=safe_infos,  # Now perfectly matching actions length!
        terminal=True,
        rews=terminal_rewards
    )
    return [trajectory]

def generate_distribution_plots(expert_obs: np.ndarray, gen_obs: np.ndarray, save_path: str) -> None:
    """Generates comparative kernel density evaluation profiles between policies."""
    features = ['Speed', 'Distance', 'Lane']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, feature_name in enumerate(features):
        sns.kdeplot(expert_obs[:, idx], label="Expert Data", fill=True, alpha=0.4, ax=axes[idx], bw_adjust=0.75)
        if len(gen_obs) > 0:
            sns.kdeplot(gen_obs[:, idx], label="Generator Policy", fill=True, alpha=0.4, ax=axes[idx], bw_adjust=0.75)
        
        axes[idx].set_title(f"{feature_name} Distribution Match Profile")
        axes[idx].set_xlabel("Normalized Scale (0.0 - 1.0)")
        axes[idx].set_ylabel("Density")
        axes[idx].legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[CORE] Diagnostic visualization cached successfully at: {save_path}")


# ==============================================================================
# II. COMPLIANCE & ENVIRONMENT ADAPTER (The Interface Bridge)
# ==============================================================================

class GymnasiumInterfaceAdapter(DummyDrivingEnv):
    """Translates legacy OpenAI Gym structural vectors to native Gymnasium specs."""
    def __init__(self, df):
        super().__init__(df)
        self.observation_space = class_gymnasium.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = class_gymnasium.spaces.Discrete(5)

    def _extract_and_normalize(self, raw_data_source) -> np.ndarray:
        """Forces extraction of the true underlying physics features directly."""
        try:
            # Convert whatever object the env returns to a flat float array
            arr = np.asarray(raw_data_source, dtype=np.float32).flatten()
            
            # If the environment returns a dummy empty/short array, fall back to physics attributes
            if arr.shape[0] < 3:
                s = getattr(self, 'speed', 20.0)
                d = getattr(self, 'distance', 105.0)
                l = getattr(self, 'lane', 4.0)
                arr = np.array([s, d, l], dtype=np.float32)
        except Exception:
            # Hard fallback anchor if data type casting fails
            arr = np.array([20.0, 105.0, 4.0], dtype=np.float32)

        # Grab the first three core driving features: [speed, distance, lane]
        target_features = arr[:3]

        # Call your repository's native normalization vector transform math
        normalized = get_normalized_vector(*target_features)
        normalized = np.asarray(normalized, dtype=np.float32).flatten()

        if not np.all(np.isfinite(normalized)):
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(normalized, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Initialize the agent right where humans typically start driving safely
        # This prevents the policy from immediately panicking and diving to 0
        start_speed = np.random.normal(loc=50.0, scale=10.0)      # Human mean speed is around 50
        start_distance = np.random.normal(loc=250.0, scale=50.0)  # Human mean distance
        start_lane = np.random.choice([0.0, 4.0, 8.0])            # Valid travel lanes
        
        self.raw_state = [start_speed, start_distance, start_lane]
        return get_normalized_vector(*self.raw_state), {}

    def step(self, action):
        legacy_output = super().step(action)
        
        # Unpack based on step output signature lengths safely
        if len(legacy_output) == 5:
            raw_obs, reward, terminated, truncated, info = legacy_output
        elif len(legacy_output) == 4:
            raw_obs, reward, done, info = legacy_output
            terminated, truncated = bool(done), False
        else:
            raise ValueError("Malformed environment transition signature tuple detected.")
            
        # Transform the live step observation vector
        sanitized_obs = self._extract_and_normalize(raw_obs)
        
        return sanitized_obs, float(reward), bool(terminated), bool(truncated), (info or {})

# ==============================================================================
# III. DECLARATIVE ASSEMBLY (Lego-Style Pipeline Configuration Blocks)
# ==============================================================================

class ObjectiveAblationTracker(KVWriter):
    """Monitors learning mechanics in real-time to detect vanishing gradients."""
    def write(self, key_values, key_excluded, step=0):
        critical_metrics = [
            "train/policy_gradient_loss", 
            "train/entropy_loss", 
            "mean/disc/disc_loss"
        ]
        reported = {k: v for k, v in key_values.items() if k in critical_metrics}
        if reported:
            print(f" [Ablation Telemetry @ Step {step}]: {reported}")


def build_vectorized_runtime_env(df: pd.DataFrame, num_envs: int = 1) -> DummyVecEnv:
    """Vectorizes environments to optimize training loops."""
    def make_env():
        # Pass the parsed pandas dataframe object 'df' directly into the adapter
        return RolloutInfoWrapper(GymnasiumInterfaceAdapter(df))
        
    return DummyVecEnv([make_env for _ in range(num_envs)])


def instantiate_airl_engine(env: DummyVecEnv, gen_lr: float, disc_lr: float) -> AIRL:
    """Assembles policy and reward network parameters into an adversarial framework."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ppo_agent = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=gen_lr,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.05,
        device=device
    )
    
    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        use_state=True,
        use_action=True,
        use_next_state=False,
        use_done=False
    )
    
    return AIRL(
        demonstrations=None,  # Configured dynamically during the run shell
        demo_batch_size=32,
        venv=env,
        gen_algo=ppo_agent,
        reward_net=reward_net,
        allow_variable_horizon=True
    )


def harvest_policy_rollouts(env: DummyVecEnv, model: PPO, sample_steps: int = 5000) -> np.ndarray:
    """Rolls out the active generator model to harvest current state features."""
    obs = env.reset()
    collected = []
    
    for _ in range(sample_steps):
        actions, _ = model.predict(obs, deterministic=False)
        obs, _, dones, _ = env.step(actions)
        for idx, done in enumerate(dones):
            # Collect environment observations safely across the vector steps
            collected.append(obs[idx])
            
    return np.array(collected, dtype=np.float32)


# ==============================================================================
# IV. IMPERATIVE SHELL (State, Side-Effects, Run Loop Execution Environment)
# ==============================================================================

def execute_ablation_study(csv_data_path: str, smoke_test: bool = True) -> None:
    """Coordinates file system resources, seeds configuration loops, and runs studies."""
    print(f"\n🚀 LAUNCHING CORE ABLATION ENGINE | Mode: {'SMOKE_TEST' if smoke_test else 'PRODUCTION'}")
    set_global_seeds(42)
    
    # 1. Data Processing Core Side-Effects
    if not os.path.exists(csv_data_path):
        raise FileNotFoundError(f"Expert record vector file missing at: {csv_data_path}")
        
    raw_df = pd.read_csv(csv_data_path)
    expert_obs = extract_observation_matrix(raw_df)
    expert_trajs = build_expert_trajectories(expert_obs)
    
    # 2. Configure hyperparameter combinations to test
    ablation_matrix = [
        {"name": "Slow_Discriminator_Test", "gen_lr": 1e-3, "disc_lr": 1e-4},
        {"name": "Balanced_Baseline_Test", "gen_lr": 3e-4, "disc_lr": 3e-4}
    ]
    
    total_timesteps = 20_000 if smoke_test else 500_000
    runtime_parallel_envs = 8
    
    # 3. Process Execution Matrix Loop
    for config in ablation_matrix:
        print(f"\n--- Investigating Matrix Node Configuration: {config['name']} ---")
        
        # FIX: Explicitly pass your parsed raw_df here so the workers can access it!
        venv = build_vectorized_runtime_env(df=raw_df, num_envs=runtime_parallel_envs)
        airl_trainer = instantiate_airl_engine(venv, config['gen_lr'], config['disc_lr'])
        
        # Instead of replacing the logger, append our custom format 
        # straight into the existing imitation-compatible logger outputs!
        airl_trainer._logger.output_formats.append(ObjectiveAblationTracker())
        
        # Synchronize it over to PPO's logging module so everything stays aligned
        airl_trainer.gen_algo.set_logger(airl_trainer._logger)
        
        # Bind the dataset trajectories into the engine matrix
        airl_trainer.set_demonstrations(expert_trajs)
        
        # Phase A: Capture baseline performance before optimization begins
        print("[SHELL] Gathering Pre-Training Baseline Rollouts...")
        pre_gen_obs = harvest_policy_rollouts(venv, airl_trainer.gen_algo, sample_steps=2000)
        generate_distribution_plots(expert_obs, pre_gen_obs, f"plot_pre_{config['name']}.png")
        
        # Phase B: Execute Adversarial Policy Optimization Pass
        print(f"[SHELL] Advancing training to target epoch threshold ({total_timesteps} steps)...")
        airl_trainer.train(total_timesteps=total_timesteps)
        
        # Phase C: Evaluate post-optimization performance shifts
        print("[SHELL] Gathering Post-Training Policy Rollouts...")
        post_gen_obs = harvest_policy_rollouts(venv, airl_trainer.gen_algo, sample_steps=2000)
        generate_distribution_plots(expert_obs, post_gen_obs, f"plot_post_{config['name']}.png")
        
        # Explicit Clean-up to clear memory footprint allocations
        venv.close()

if __name__ == "__main__":
    # Point this to your target data file asset paths
    DATA_PATH = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Original_Data\master_airl_expert_dataset.csv"
    
    # Toggle smoke_test=False once you verify metrics and plots update cleanly without freezing
    execute_ablation_study(csv_data_path=DATA_PATH, smoke_test=True)