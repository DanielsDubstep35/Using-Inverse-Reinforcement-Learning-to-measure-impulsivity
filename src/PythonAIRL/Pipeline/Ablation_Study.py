import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.logger import KVWriter

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.data import rollout
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper

import gymnasium as class_gymnasium
from Game_Code import DummyDrivingEnv, get_normalized_vector

# ==============================================================================
# 1. VIRTUALLY VECTORIZED COMPLIANCE LAYER
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
    """Reconstructs episodic vectors and maps structural discrete choice tracks."""
    trajectories = []
    grouped = df.groupby(["participant", "episode"])
    eps = 1e-4

    print(f"[DATA] Parsing dataset into sequential trajectories...")
    for (_, _), run_df in grouped:
        run_df = run_df.sort_values(by="time")
        if len(run_df) < 2:
            continue

        raw_speeds = run_df["speed"].to_numpy()
        raw_distances = run_df["distance"].to_numpy()
        raw_lanes = run_df["lane"].to_numpy()

        obs_matrix = []
        for i in range(len(run_df)):
            obs_matrix.append(get_normalized_vector(raw_speeds[i], raw_distances[i], raw_lanes[i]))
        obs_matrix = np.array(obs_matrix, dtype=np.float32)

        # Reconstruct structural choice history
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

        discrete_actions = np.array(actions, dtype=np.int64)

        traj = TrajectoryWithRew(
            obs=obs_matrix,
            acts=discrete_actions,
            infos=np.array([{} for _ in range(len(discrete_actions))]),
            terminal=True,
            rews=np.zeros(len(discrete_actions), dtype=np.float32),
        )
        trajectories.append(traj)

    return trajectories


# ==============================================================================
# 3. COMPUTATIONAL NETWORK FACTORY BLOCK
# ==============================================================================

def instantiate_airl_architecture(
    venv: SubprocVecEnv,
    device: torch.device,
    hidden_sizes: Tuple[int, ...],
    lr: float,
    batch_size: int,
    demo_batch_size: int,
) -> Tuple[AIRL, BasicRewardNet]:
    
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True,
        use_action=True,  # Secure now that discrete tracks are parsed cleanly
        use_next_state=False,
        use_done=False,
        hid_sizes=hidden_sizes,
    )

    gen_algo = PPO(
        "MlpPolicy",
        env=venv,
        verbose=0,
        n_steps=1024,
        batch_size=batch_size,
        learning_rate=lr,
        ent_coef=0.05,
        n_epochs=5,
        gae_lambda=0.95,
        clip_range=0.2,
        device=device,
        policy_kwargs=dict(net_arch=dict(pi=list(hidden_sizes), vf=list(hidden_sizes))),
    )

    airl_trainer = AIRL(
        demonstrations=None,
        demo_batch_size=demo_batch_size,
        venv=venv,
        gen_algo=gen_algo,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )

    return airl_trainer, reward_net


# ==============================================================================
# 4. TELEMETRY MONITOR HOOKS
# ==============================================================================

class LossTracker(KVWriter):
    """Intercepts internal training logs to extract real-time discriminator performance."""
    def __init__(self):
        self.disc_losses = []
        self.gen_losses = []

    def write(self, key_values, key_excluded, step=0):
        if "mean/disc/disc_loss" in key_values:
            self.disc_losses.append(key_values["mean/disc/disc_loss"])
        if "train/policy_gradient_loss" in key_values:
            self.gen_losses.append(key_values["train/policy_gradient_loss"])


# ==============================================================================
# 5. EXECUTION CORE CONTROL
# ==============================================================================

def run_hyperparameter_ablation_study(
    data_file: str,
    configs_dict: dict,
    sample_fraction: float = 0.20,
    num_envs: int = 24,
    ablation_epochs: int = 1,
    timesteps_per_epoch: int = 500_000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLAUNCHING RE-VALIDATED PARALLEL ABLATION GRID [{device.type.upper()}]")

    raw_df = load_and_group_expert_csv(data_file)
    
    # Stratified downsampling by participant mapping
    participants = raw_df["participant"].unique()
    sampled_p = np.random.choice(
        participants, size=max(1, int(len(participants) * sample_fraction)), replace=False
    )
    small_df = raw_df[raw_df["participant"].isin(sampled_p)]
    print(f"[DATA] Retaining {len(small_df):,} lines across {len(sampled_p)} sampled subjects.")

    expert_trajectories = build_trajectories_from_dataframe(small_df)

    def make_worker_env():
        return RolloutInfoWrapper(GymnasiumInterfaceAdapter(df=raw_df))

    print(f"[WORKERS] Starting {num_envs} clean interface workers via SubprocVecEnv...")
    parallel_venv = SubprocVecEnv([make_worker_env for _ in range(num_envs)])
    parallel_venv.seed(42)

    results = {}

    try:
        for run_name, cfg in configs_dict.items():
            print(f"\n>>> PROCESSING CONFIGURATION TRACK: {run_name}")
            
            airl_trainer, reward_net = instantiate_airl_architecture(
                venv=parallel_venv,
                device=device,
                hidden_sizes=cfg["hidden_sizes"],
                lr=cfg["lr"],
                batch_size=cfg["batch_size"],
                demo_batch_size=cfg["demo_batch"],
            )

            airl_trainer.optimizer = optim.Adam(
                reward_net.parameters(), lr=cfg["lr_disc"], weight_decay=1e-5
            )
            airl_trainer.set_demonstrations(expert_trajectories)

            tracker = LossTracker()
            airl_trainer.logger.default_logger.output_formats.append(tracker)
            airl_trainer.gen_algo.set_logger(airl_trainer.logger.default_logger)

            for epoch in range(1, ablation_epochs + 1):
                print(f" -> Execution Step Pass: Epoch {epoch}/{ablation_epochs} ({timesteps_per_epoch:,} steps)...")
                airl_trainer.train(total_timesteps=timesteps_per_epoch)

            results[run_name] = {
                "disc_loss": list(tracker.disc_losses),
                "gen_loss": list(tracker.gen_losses),
            }
            print(f" ✓ Track Finalized. Captured {len(tracker.disc_losses)} updates.")

        # ==============================================================================
        # 6. PRESENTATION DIAGNOSTICS GENERATION
        # ==============================================================================
        print("\n[DIAGNOSTICS] Constructing thesis-grade optimization charts...")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Use a professional color palette
        colors = sns.color_palette("hud_gradient", len(results)) if hasattr(sns, "hud_gradient") else sns.color_palette("Set2", len(results))

        for idx, (run_name, metrics) in enumerate(results.items()):
            # Smooth out noisy gradient paths for presentation clarity using a rolling average
            disc_s = pd.Series(metrics["disc_loss"]).rolling(window=max(1, len(metrics["disc_loss"])//10), min_periods=1).mean()
            gen_s = pd.Series(metrics["gen_loss"]).rolling(window=max(1, len(metrics["gen_loss"])//10), min_periods=1).mean()

            axes[0].plot(disc_s, label=f"{run_name}", color=colors[idx], lw=2)
            axes[1].plot(gen_s, linestyle="--", label=f"{run_name}", color=colors[idx], lw=2)

        axes[0].set_title("Discriminator Convergence Profile\n(Stability Verification Across Architectures)", fontsize=11, fontweight='bold', pad=10)
        axes[0].set_xlabel("Adversarial Iteration Passes", fontsize=10)
        axes[0].set_ylabel("Binary Cross Entropy Loss Magnitude", fontsize=10)
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].set_title("Policy Generator Optimization Curve\n(PPO Objective Traking Profile)", fontsize=11, fontweight='bold', pad=10)
        axes[1].set_xlabel("Adversarial Iteration Passes", fontsize=10)
        axes[1].set_ylabel("Surrogate Policy Loss Magnitude", fontsize=10)
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="lower left", fontsize=8)

        plt.tight_layout()
        output_plot_path = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Reward Networks\ablation_study_results_9_experiment.png"
        os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
        plt.savefig(output_plot_path, dpi=200)
        plt.close()
        print(f"💥 Complete success! Presentation charts compiled and exported to: {output_plot_path}")

    finally:
        parallel_venv.close()


if __name__ == "__main__":
    import random
    import multiprocessing
    multiprocessing.freeze_support()

    # Anchor determinism
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    DATA_FILE = r"C:\Users\Danie\Desktop\PracticumRepository\src\PythonAIRL\Original_Data\master_airl_expert_dataset.csv"

    ABLATION_GRID = {
        "THEIRS (Deep 512)": {
            "lr": 5e-4,
            "lr_disc": 1e-4,
            "hidden_sizes": (512, 512, 512),
            "batch_size": 512,
            "demo_batch": 500,
        },
        "BEST ATTEMPT (Bottleneck)": {
            "lr": 5e-4,
            "lr_disc": 5e-4,
            "hidden_sizes": (512, 256, 128),
            "batch_size": 1024,
            "demo_batch": 2048,
        },
        "Thin Network 64": {
            "lr": 5e-4,
            "lr_disc": 1e-4,
            "hidden_sizes": (64, 64, 64),
            "batch_size": 512,
            "demo_batch": 500,
        },
        "Shallow Compact 2-Layer": {
            "lr": 5e-4,
            "lr_disc": 1e-4,
            "hidden_sizes": (64, 64),
            "batch_size": 512,
            "demo_batch": 500,
        },
        "Aggressive Disc (Unstable Check)": {
            "lr": 1e-4,
            "lr_disc": 5e-4,
            "hidden_sizes": (512, 512, 512),
            "batch_size": 512,
            "demo_batch": 500,
        }
    }

    run_hyperparameter_ablation_study(
        data_file=DATA_FILE,
        configs_dict=ABLATION_GRID,
        sample_fraction=0.15,   # Balanced slice for rigorous evaluation speed
        num_envs=28,            # Set to your system thread preference
        ablation_epochs=3,      # 1 Epoch is enough to confirm gradient stability patterns
        timesteps_per_epoch=250_000
    )