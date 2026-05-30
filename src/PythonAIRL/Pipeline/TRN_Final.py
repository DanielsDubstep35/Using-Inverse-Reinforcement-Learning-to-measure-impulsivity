import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import KVWriter

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper

import gymnasium as class_gymnasium
from Game import DummyDrivingEnv, get_normalized_vector

# ==============================================================================
# 1. COMPLIANCE & ENVIRONMENT ADAPTER (Gymnasium Interface Bridge)
# ==============================================================================

class GymnasiumInterfaceAdapter(DummyDrivingEnv):
    """
    Translates trace-driven legacy simulation vectors to Gymnasium 5-tuple specs,
    applying native normalization maps and a foolproof Inf/NaN firewall.
    """
    def __init__(self, df):
        super().__init__(df)
        self.observation_space = class_gymnasium.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = class_gymnasium.spaces.Discrete(5)

    def _get_sanitized_native_obs(self) -> np.ndarray:
        if not hasattr(self, 'raw_state') or self.raw_state is None:
            self.raw_state = [50.0, 250.0, 4.0]
            
        obs = get_normalized_vector(*self.raw_state)
        obs = np.asarray(obs, dtype=np.float32).flatten()
        
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=0.0)
            
        return np.clip(obs, 0.0, 1.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return self._get_sanitized_native_obs(), {}

    def step(self, action):
        legacy_output = super().step(action)
        
        if len(legacy_output) == 5:
            _, reward, terminated, truncated, info = legacy_output
        elif len(legacy_output) == 4:
            _, reward, done, info = legacy_output
            terminated, truncated = bool(done), False
        else:
            raise ValueError("Malformed environment transition signature tuple detected.")

        return self._get_sanitized_native_obs(), float(reward), bool(terminated), bool(truncated), (info or {})


# ==============================================================================
# 2. EXPERT DATA INGESTION & DISCRETE ACTION RECONSTRUCTION
# ==============================================================================

def load_and_group_expert_csv(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Target expert dataset missing at: {filepath}")
    return pd.read_csv(filepath)


def build_trajectories_from_dataframe(df: pd.DataFrame) -> List[TrajectoryWithRew]:
    """ Reconstructs perfectly aligned trajectories with explicit action parsing. """
    trajectories = []
    grouped = df.groupby(["participant", "episode"])
    eps = 1e-4

    print(f"[DATA] Packaging {len(grouped)} human driving runs into episodic vectors...")
    for (_, _), run_df in grouped:
        run_df = run_df.sort_values(by="time")
        if len(run_df) < 2:
            continue

        raw_speeds = run_df["speed"].to_numpy()
        raw_distances = run_df["distance"].to_numpy()
        raw_lanes = run_df["lane"].to_numpy()

        # Build structural state-space arrays
        obs_matrix = []
        for i in range(len(run_df)):
            obs_matrix.append(get_normalized_vector(raw_speeds[i], raw_distances[i], raw_lanes[i]))
        obs_matrix = np.array(obs_matrix, dtype=np.float32)

        # Reverse-engineer discrete actions matching environment modifier maps
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
                actions.append(1)  # MAINTAIN / STAY

        discrete_actions = np.array(actions, dtype=np.int64)

        traj = TrajectoryWithRew(
            obs=obs_matrix,
            acts=discrete_actions,
            infos=np.array([{} for _ in range(len(discrete_actions))]),
            terminal=True,
            rews=np.zeros(len(discrete_actions), dtype=np.float32),
        )
        trajectories.append(traj)

    print(f"[DATA] Extraction finalized. Retained {len(trajectories)} valid expert trajectories.")
    return trajectories


# ==============================================================================
# 3. FIXED TUNING FACTORY (THEIRS ARCHITECTURE ASSEMBLY)
# ==============================================================================

def instantiate_theirs_architecture(
    venv: DummyVecEnv,
    device: torch.device,
    hidden_sizes: Tuple[int, ...],
    lr: float,
    batch_size: int,
    demo_batch_size: int,
) -> Tuple[AIRL, BasicRewardNet]:
    """ Assembles deep multilayer perceptrons explicitly mapped to 'THEIRS' specs. """
    
    # 1. Initialize Discriminator/Reward network structure
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True,
        use_action=True,  # Fully functional now that actions are clean discrete keys
        use_next_state=False,
        use_done=False,
        hid_sizes=hidden_sizes,  # Mapped to (512, 512, 512)
    )

    def linear_schedule(initial_value: float):
        return lambda progress_remaining: progress_remaining * initial_value

    # 2. Establish Generator Policy configuration parameters
    gen_algo = PPO(
        "MlpPolicy",
        env=venv,
        verbose=0,
        n_steps=1024,
        batch_size=batch_size,               # Mapped to 512
        learning_rate=linear_schedule(lr),   # Mapped to 5e-4
        ent_coef=0.03,
        n_epochs=10,
        gae_lambda=0.95,
        clip_range=0.15,
        device=device,
        policy_kwargs=dict(net_arch=dict(pi=list(hidden_sizes), vf=list(hidden_sizes))), # (512, 512, 512)
    )

    # 3. Encapsulate into adversarial pipeline
    airl_trainer = AIRL(
        demonstrations=None,
        demo_batch_size=demo_batch_size,     # Mapped to 500
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    return airl_trainer, reward_net


# ==============================================================================
# 4. DIAGNOSTICS & LOSS PROFILERS
# ==============================================================================

class ProductionLossTracker(KVWriter):
    def __init__(self):
        self.disc_losses = []
        self.gen_losses = []

    def write(self, key_values, key_excluded, step=0):
        if "mean/disc/disc_loss" in key_values:
            self.disc_losses.append(key_values["mean/disc/disc_loss"])
        if "train/policy_gradient_loss" in key_values:
            self.gen_losses.append(key_values["train/policy_gradient_loss"])


def generate_distribution_plots(expert_trajectories, gen_algo, venv, step_name):
    expert_obs = np.vstack([traj.obs for traj in expert_trajectories])
    rng = np.random.default_rng(seed=42)

    gen_trajectories = rollout.rollout(
        gen_algo,
        venv,
        rollout.make_sample_until(min_timesteps=min(len(expert_obs), 30000), min_episodes=15),
        rng=rng,
    )
    gen_obs = np.vstack([traj.obs for traj in gen_trajectories])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    features = ["Speed", "Distance", "Lane"]
    
    for i, feature in enumerate(features):
        sns.kdeplot(expert_obs[:, i], ax=axes[i], label="Expert Data", fill=True, color="#1f77b4", alpha=0.4, bw_adjust=0.75)
        sns.kdeplot(gen_obs[:, i], ax=axes[i], label="Generator Policy", fill=True, color="#ff7f0e", alpha=0.4, bw_adjust=0.75)
        axes[i].set_title(f"{feature} Distribution Match")
        axes[i].set_xlim(0.0, 1.0)
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"production_{step_name}.png", dpi=150)
    plt.close()


# ==============================================================================
# 5. PIPELINE PRODUCTION SHELL
# ==============================================================================

def execute_production_pipeline(data_file: str, save_model_path: str, num_envs: int = 16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}\nRUNNING DEFINITIVE PRODUCTION RUN: [ THEIRS SPECIFICATIONS ]\n{'='*70}")

    # Ingest data matrix
    raw_df = load_and_group_expert_csv(data_file)
    expert_trajectories = build_trajectories_from_dataframe(raw_df)

    # Establish worker pools
    def make_env():
        return RolloutInfoWrapper(GymnasiumInterfaceAdapter(df=raw_df))
    
    base_venv = DummyVecEnv([make_env for _ in range(num_envs)])
    base_venv.seed(42)

    # Explicitly pull parameters from THEIRS configuration matrix node
    THEIRS_CONFIG = {
        "lr": 5e-4,
        "lr_disc": 1e-4,
        "hidden_sizes": (512, 512, 512),
        "batch_size": 512,
        "demo_batch": 500
    }

    try:
        # Construct framework
        airl_trainer, reward_net = instantiate_theirs_architecture(
            venv=base_venv,
            device=device,
            hidden_sizes=THEIRS_CONFIG["hidden_sizes"],
            lr=THEIRS_CONFIG["lr"],
            batch_size=THEIRS_CONFIG["batch_size"],
            demo_batch_size=THEIRS_CONFIG["demo_batch"]
        )

        # Inject optimizer hyperparameters with standard weight decay regularizations
        airl_trainer.optimizer = optim.Adam(
            reward_net.parameters(), 
            lr=THEIRS_CONFIG["lr_disc"], 
            weight_decay=1e-5
        )
        airl_trainer.set_demonstrations(expert_trajectories)

        # Pre-training checkpoint metric printout
        generate_distribution_plots(expert_trajectories, airl_trainer.gen_algo, base_venv, "pre_train")

        # Set up active loss tracking telemetry
        tracker = ProductionLossTracker()
        airl_trainer.logger.default_logger.output_formats.append(tracker)
        airl_trainer.gen_algo.set_logger(airl_trainer.logger.default_logger)

        # Production iteration dimensions
        PRODUCTION_EPOCHS = 20
        STEPS_PER_EPOCH = 500_000

        print(f"\n🚀 Execution Loop Initialized. Target Profile: {PRODUCTION_EPOCHS} Epochs x {STEPS_PER_EPOCH:,} steps.")
        for epoch in range(1, PRODUCTION_EPOCHS + 1):
            print(f"--- Processing Optimization Sequence Block: Epoch {epoch}/{PRODUCTION_EPOCHS} ---")
            airl_trainer.train(total_timesteps=STEPS_PER_EPOCH)
            
            # Checkpoint backup intervals every 5 epochs
            if epoch % 5 == 0:
                torch.save(reward_net.state_dict(), f"checkpoint_theirs_epoch_{epoch}.pth")
                print(f"[CHECKPOINT] Network params exported successfully.")

        # Post-training alignment profiles
        generate_distribution_plots(expert_trajectories, airl_trainer.gen_algo, base_venv, "post_train")

        # Serialize complete output reward function weight parameters
        print("\n" + "="*70 + "\nEXPORTING EXPLICIT DECOUPLED CRITICAL UTILITY MATRIX\n" + "="*70)
        torch.save(reward_net.state_dict(), save_model_path)
        print(f"💥 Production pipeline fully complete! Target serialized model saved: '{save_model_path}'")

    finally:
        base_venv.close()


if __name__ == "__main__":
    # Handle environment seeding anchors
    import random
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Asset paths
    DATA_FILE_PATH = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Original_Data\master_airl_expert_dataset.csv"
    PRODUCTION_REWARD_OUTPUT = "airl_reward_model_production_theirs.pth"

    execute_production_pipeline(
        data_file=DATA_FILE_PATH,
        save_model_path=PRODUCTION_REWARD_OUTPUT,
        num_envs=16
    )