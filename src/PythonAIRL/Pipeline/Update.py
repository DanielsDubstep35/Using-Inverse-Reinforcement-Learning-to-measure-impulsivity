# Imports

# General
import socket
import json
import pandas as pd
import numpy as np
import torch
import tqdm
import matplotlib
matplotlib.use('Agg') # Prevents crashes if no display is connected
import matplotlib.pyplot as plt
import seaborn as sns

# Gym (Fake) environment
import gymnasium as gym
from gymnasium import spaces

# Reinforcement Learning
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
import imitation.data.types as types
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ====================
# Driving Environment: Pure Functions
# ====================

#  Create a new state
def create_state(speed: float, distance: float, lane: float):
    return np.array([speed, distance, lane], dtype=np.float32)

# Create box observation space lower bounds
def create_box_low(speed: float, distance: float, lane: float):
    return np.array([speed, distance, lane])

# Create box observation space higher bounds
def create_box_high(speed: float, distance: float, lane: float):
    return np.array([speed, distance, lane])

# Move time forward by a value  
def move_step_count(step_count: int, by_value: int):
    return step_count + by_value

def is_terminated(dist:int, step_count: int):
    return dist <= 0 or step_count >= 100
    return

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
        
        # State: [Speed, Distance, LaneID] 
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5) 
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Handles RNG seeding if needed
        
        # Initialize state
        self.state = create_state(20.0, 105.0, 1.0)
        self.step_count = 0
        
        # CRITICAL: Return (observation, info)
        # return self.state, {}
        
        normalized_state = get_normalized_vector(self.speed, self.distance, self.lane)
        return normalized_state, {}

    def step(self, action):
        # ... your existing step logic ...
        move_step_count(self.step_count, 1)
        speed, dist, lane = self.state
        
        self.state = create_state(speed, dist, lane)
        
        # Return (obs, reward, terminated, truncated, info)
        return self.state, 0.0, is_terminated(dist, self.step_count), False, {}


# ====================
# Unreal Engine Bridge
# ====================
class UnrealPostGameBridge:
    def __init__(self, hostname="127.0.0.1", port=3000):
        self.host = hostname
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.trajectory_buffer = []
        self.episode_count = 0
        self.training_history = []

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
        while True:
            try:
                data = conn.recv(1048576)
                if not data:
                    break

                buffer += data.decode("utf-8") + "\n"

                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    if not message.strip():
                        continue

                    packet = json.loads(message)
                    msg_type = packet.get("type", "state")

                    if msg_type == "state":
                        if isinstance(packet.get("data"), list):
                            if len(packet["data"]) >= 3:
                                # Scale distance if it's in Unreal Units (e.g., / 100.0)
                                raw_data = packet["data"]
                                speed = raw_data[0]
                                dist = raw_data[1] / 100.0 # <--- Adjust this divisor based on Unreal's scale
                                lane = raw_data[2]
                                
                                self.trajectory_buffer.append([speed, dist, lane])

                    elif msg_type == "level_complete":
                        print(
                            f"Level Complete! Analyzing {len(self.trajectory_buffer)} frames..."
                        )
                        score, weights = self.run_airl_training()

                        result = {
                            "type": "result",
                            "impulsivity_score": float(score),
                            "details": f"Spd:{weights[0]:.2f} Safe:{weights[1]:.2f}",
                        }
                        conn.sendall((json.dumps(result) + "\n").encode("utf-8"))
                        print(f"Sent Final Score: {score}\n")
                        return

            except Exception as e:
                import traceback
                print(f"Session Error: {e}")
                traceback.print_exc() # <--- ADD THIS LINE
                break

    # Executes training and extracts metrics
    def run_airl_training(self):
        ep_length = len(self.trajectory_buffer)
        if ep_length < 10:
            print("Not enough frames to train.")
            return 0.0, [0.0, 0.0, 0.0]

        self.episode_count += 1

        expert_obs, acts, trajectory = format_expert_data(self.trajectory_buffer)
        
        # Inside run_airl_training:
        raw_venv = DummyVecEnv([lambda: DummyDrivingEnv()])
        # This wraps the env to scale rewards and observations to a small range (e.g., -1 to 1)
        venv = VecNormalize(raw_venv, norm_obs=True, norm_reward=True, clip_obs=10.)
        
        # learner, reward_net, airl_trainer = build_airl_trainer(venv, trajectory)
        
        # temp_result = build_airl_trainer(venv, trajectory)
        # print(f"DEBUG: build_airl_trainer returned: {temp_result}")
        # learner, reward_net, airl_trainer = temp_result
        
        # Only call it once
        temp_result = build_airl_trainer(venv, trajectory)

        if temp_result is None:
            print("CRITICAL ERROR: build_airl_trainer returned None. Check the return statement in that function.")
            return 0.0, [0.0, 0.0, 0.0]

        learner, reward_net, airl_trainer = temp_result
        print(f"DEBUG: Successfully initialized AIRL Trainer components.")
        
        # airl_trainer.logger = configure(None, ["stdout"])

        metrics_history = []
        print("\nTraining AIRL (Generator vs. Discriminator)...")
        for i in tqdm.tqdm(range(5), desc="AIRL Training", unit="chunk"):
            airl_trainer.train(total_timesteps=32768)
            # airl_trainer.train(total_timesteps=2048)

            acc, loss = get_disc_stats(airl_trainer, expert_obs, acts)

            metrics_history.append(
                {
                    "Chunk": i + 1,
                    "Disc Acc": acc,
                    "Disc Loss": loss,
                }
            )

        print("Reward net device:", next(reward_net.parameters()).device)
        print("Policy device:", next(learner.policy.parameters()).device)


        weights = extract_weights(reward_net)
        score = calculate_impulsivity_score(reward_net, expert_obs)
        heatmap = generate_reward_heatmap(reward_net, self.episode_count)
        mean_reward = evaluate_expert(reward_net, expert_obs, acts)

        # Collect Generator Observations:
        gen_samples = airl_trainer.gen_algo.rollout_buffer.observations
        # Flatten buffer: (returns): [Speed, Distance, Lane]
        gen_obs = gen_samples.reshape(-1, gen_samples.shape[-1])

        avg_expert = expert_obs.mean(axis=0)
        avg_gen = gen_obs.mean(axis=0)

        print(get_reward_for_state(reward_net, 20, 0.0, 0.0))

        # print("\n=== ADVERSARIAL TRAINING LOGS ===")
        # print(df_metrics.to_string(index=False))
        # print("=" * 50)

        # ... (rest of your existing evaluation and weight extraction) ...

        print("\n=== AIRL Comparison Table ===")
        print(f"{'METRIC':<15} | {'EXPERT MEAN':<15} | {'GEN MEAN':<15}")
        print("-" * 50)
        print(f"{'Speed (x)':<15} | {avg_expert[0]:<15.4f} | {avg_gen[0]:<15.4f}")
        print(f"{'Distance (y)':<15} | {avg_expert[1]:<15.4f} | {avg_gen[1]:<15.4f}")
        print(f"{'Lane Pos (z)':<15} | {avg_expert[2]:<15.4f} | {avg_gen[2]:<15.4f}")
        print("=" * 50 + "\n")

        return score, weights


# ====================
# Useful Functions
# ====================
def map_raw_to_discrete(current_obs, prev_obs):
    curr_spd, curr_dist, curr_lane = current_obs
    prev_spd, prev_dist, prev_lane = prev_obs
    
    # Check Lane first (highest priority)
    if curr_lane < prev_lane: return 3 # Left
    if curr_lane > prev_lane: return 4 # Right
    
    # Check Speed
    if curr_spd > prev_spd: return 1 # Accel
    if curr_spd < prev_spd: return 2 # Decel
    
    return 0 # Stay


def get_reward_for_state(reward_net, speed, dist, lane):
    device = next(reward_net.parameters()).device
    state = torch.tensor([[speed, dist, lane]], dtype=torch.float32, device=device)
    # Action must be a LongTensor for Discrete spaces
    act = torch.zeros((1,), dtype=torch.long, device=device) 
    done = torch.zeros(1, dtype=torch.bool, device=device)

    with torch.no_grad():
        rew = reward_net.forward(state, act, state, done)
    return rew.item()


def calculate_impulsivity_score(reward_net, expert_obs):
    device = next(reward_net.parameters()).device
    reward_net.eval()

    with torch.no_grad():
        # 1. Define extreme states
        # [Speed, Distance, Lane]
        safe_state = torch.tensor([[20.0, 105.0, 1.0]], device=device)
        impulsive_state = torch.tensor([[120.0, 0.0, 1.0]], device=device)
        
        # Dummy actions/dones for the reward net
        act = torch.zeros((1,), dtype=torch.long, device=device)
        done = torch.zeros(1, dtype=torch.bool, device=device)

        # 2. Get baseline rewards
        r_min = reward_net.predict_processed(safe_state, act, safe_state, done).item()
        r_max = reward_net.predict_processed(impulsive_state, act, impulsive_state, done).item()

        # 3. Get Expert's average reward
        # Convert expert_obs to tensor if it isn't already
        exp_tensor = torch.tensor(expert_obs, dtype=torch.float32, device=device)
        exp_acts = torch.zeros((len(exp_tensor),), dtype=torch.long, device=device)
        exp_dones = torch.zeros(len(exp_tensor), dtype=torch.bool, device=device)
        
        r_expert = reward_net.predict_processed(exp_tensor, exp_acts, exp_tensor, exp_dones).mean().item()

    # 4. Calculate Percentage (Linear Interpolation)
    # Formula: (Expert - Min) / (Max - Min)
    if abs(r_max - r_min) < 1e-5:
        percentage = 0.0
    else:
        percentage = ((r_expert - r_min) / (r_max - r_min)) * 100.0

    final_score = np.clip(percentage, 0.0, 100.0)
    
    print(f"--- Scoring Debug ---")
    print(f"Safe Anchor Reward: {r_min:.2f}")
    print(f"Impulse Anchor Reward: {r_max:.2f}")
    print(f"Expert Avg Reward: {r_expert:.2f}")
    print(f"Final Impulsivity: {final_score:.1f}%")
    
    return round(float(final_score), 2)


def format_expert_data(trajectory_buffer):
    # Convert list to numpy array: [Speed, Distance, LaneID]
    expert_obs = np.array(trajectory_buffer, dtype=np.float32)
    
    # Initialize list for discrete actions
    discrete_actions = []
    
    # Loop from the second frame to the end to compare with previous
    for i in range(1, len(expert_obs)):
        current_frame = expert_obs[i]
        previous_frame = expert_obs[i-1]
        
        # Call the mapping function
        action = map_raw_to_discrete(current_frame, previous_frame)
        discrete_actions.append(action)
    
    # AIRL/imitation library requires actions to be a numpy array
    # For Discrete space, the shape must be (N,)
    acts = np.array(discrete_actions, dtype=np.int64)
    
    # The observations in a trajectory must include the final state,
    # so len(obs) == len(acts) + 1. This matches our loop above.
    infos = np.array([{}] * len(acts))
    
    trajectory = types.Trajectory(
        obs=expert_obs, 
        acts=acts, 
        infos=infos, 
        terminal=True
    )
    
    return expert_obs, acts, trajectory


# Creates generator and reward networks
def build_airl_trainer(venv, trajectory):
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,
        n_steps=2048,
        ent_coef=0.1,
        learning_rate=3e-4,
        n_epochs=10, 
        verbose=0,
    )
    # Inside build_airl_trainer
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space, # This is now Discrete(5)
        use_state=True,
        use_action=False,      # Paper uses state-only reward g(s)
        use_next_state=False,
        use_done=False,
        hid_sizes=(64, 64),
    )
    airl_trainer = AIRL(
        demonstrations=[trajectory],
        demo_batch_size=64,
        gen_replay_buffer_capacity=5000,
        n_disc_updates_per_round=1,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )
    
    return learner, reward_net, airl_trainer


def extract_weights(reward_net):
    reward_net.eval()
    device = next(reward_net.parameters()).device

    # Standardized state for sensitivity analysis
    base_state = torch.tensor([[70.0, 50.0, 1.0]], device=device, requires_grad=True)
    
    # We need tensors for AIRL's forward pass
    # Note: Using .forward() directly bypasses the "processed" wrapper but gives us the raw tensor
    try:
        # Create dummy tensors for the other required inputs
        act = torch.zeros((1,), dtype=torch.long, device=device)
        next_obs = base_state.clone()
        done = torch.tensor([False], device=device)

        # Call forward to keep it in the Torch domain
        rew = reward_net.forward(base_state, act, next_obs, done)
        rew.backward()

        sensitivities = base_state.grad.detach().cpu().numpy().flatten()
        return sensitivities
    except Exception as e:
        print(f"Extraction Error: {e}")
        return np.array([0.0, 0.0, 0.0])


# Evaluates mean reward of expert
def evaluate_expert(reward_net, expert_obs, acts):
    with torch.no_grad():
        states = torch.tensor(expert_obs[:-1], dtype=torch.float32)
        actions = torch.tensor(acts, dtype=torch.float32)
        next_states = torch.tensor(expert_obs[1:], dtype=torch.float32)
        dones = torch.zeros(len(acts), dtype=torch.bool)
        return (
            reward_net.predict_processed(states, actions, next_states, dones)
            .mean()
            .item()
        )


def get_disc_stats(airl_trainer, expert_obs, acts):
    reward_net = airl_trainer._reward_net
    policy = airl_trainer.gen_algo.policy

    device = next(reward_net.parameters()).device  # 👈 key line

    states = torch.tensor(expert_obs[:-1], dtype=torch.float32, device=device)
    actions = torch.tensor(acts, dtype=torch.float32, device=device)
    next_states = torch.tensor(expert_obs[1:], dtype=torch.float32, device=device)
    dones = torch.zeros(len(acts), dtype=torch.bool, device=device)

    with torch.no_grad():
        r = reward_net.forward(states, actions, next_states, dones)

        dist = policy.get_distribution(states)
        log_prob = dist.log_prob(actions)

        if log_prob.dim() == 1:
            log_prob = log_prob.unsqueeze(-1)

        logits = r - log_prob
        probs = torch.sigmoid(logits)

        acc = ((probs > 0.5).float().mean()).item()

        loss = torch.nn.functional.binary_cross_entropy(probs, torch.ones_like(probs))

        print(
            {
                "reward_mean": r.mean().item(),
                "log_prob_mean": log_prob.mean().item(),
                "disc_prob_mean": probs.mean().item(),
            }
        )

    return acc, loss.item()


def generate_reward_heatmap(reward_net, episode_num=0):
    speeds = np.linspace(20, 120, 10)
    distances = np.linspace(0, 105, 10)
    lane = 1.0 
    
    device = next(reward_net.parameters()).device
    reward_net.eval()
    
    heatmap_data = []
    with torch.no_grad():
        for d in reversed(distances):
            row_rewards = []
            for s in speeds:
                state = torch.tensor([[s, d, lane]], dtype=torch.float32, device=device)
                act = torch.zeros((1,), dtype=torch.long, device=device)
                done = torch.zeros(1, dtype=torch.bool, device=device)
                rew = reward_net.predict_processed(state, act, state, done)
                row_rewards.append(rew.item())
            heatmap_data.append(row_rewards)
            
    data_array = np.array(heatmap_data)

    # --- TERMINAL "PICTURE" VISUALIZATION ---
    print("\n=== TERMINAL REWARD MAP (Red=High, Blue=Low) ===")
    min_val, max_val = data_array.min(), data_array.max()
    
    for row in data_array:
        line = ""
        for val in row:
            # Calculate color intensity (0-255)
            ratio = (val - min_val) / (max_val - min_val + 1e-6)
            r = int(255 * ratio)
            b = int(255 * (1 - ratio))
            # Use ANSI escape codes for background color
            line += f"\033[48;2;{r};0;{b}m {val:>5.2f} \033[0m"
        print(line)
    print("===============================================\n")

    # --- SAVE AS PLOT ---
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data_array, 
        annot=True, 
        xticklabels=np.round(speeds).astype(int), 
        yticklabels=np.round(distances[::-1]).astype(int),
        cmap="coolwarm"
    )
    plt.title(f"AIRL Reward Heatmap - Episode {episode_num}")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Distance to Lead Car (m)")
    
    filename = f"reward_heatmap_ep{episode_num}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Heatmap saved as: {filename}")
            
    return data_array



# ====================
# Code Start
# ====================
if __name__ == "__main__":
    server = UnrealPostGameBridge()
    server.start_server()