# Imports

# General
import socket
import json
import pandas as pd
import numpy as np
import torch
import tqdm
import matplotlib

matplotlib.use("Agg")  # Prevents crashes if no display is connected
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

NORM_BOUNDS = {
    "speed": {"min": 0.0, "max": 120.0},
    "distance": {"min": 0.0, "max": 500.0},
    "lane": {"min": -4.0, "max": 16.0},
}


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


# Functional Core:

# ====================
# ========================================
# Driving Environment
# ========================================
# ====================


def create_state(speed: float, distance: float, lane: float) -> np.ndarray:
    return np.array([speed, distance, lane], dtype=np.float32)


def create_box_low(speed: float, distance: float, lane: float) -> np.ndarray:
    return np.array([speed, distance, lane], dtype=np.float32)


def create_box_high(speed: float, distance: float, lane: float) -> np.ndarray:
    return np.array([speed, distance, lane], dtype=np.float32)


def move_step_count(step_count: int, by_value: int) -> int:
    return step_count + by_value

def is_terminated(distance, step_count, max_steps=1000):
    """Checks for crashes or episode timeouts."""
    # A crash occurs if the agent drives directly through a vehicle ahead
    if distance <= 2.0:  # Threshold representing physical car length contact bounds
        return True
    if step_count >= max_steps:
        return True
    return False

# ====================
# ========================================
# Useful Evaluation & Training Functions
# ========================================
# ====================


def map_raw_to_discrete(current_obs, prev_obs, eps=1e-3):
    curr_spd, curr_dist, curr_lane = current_obs
    prev_spd, prev_dist, prev_lane = prev_obs

    if (prev_lane - curr_lane) > eps: return 3  # Left
    if (curr_lane - prev_lane) > eps: return 4  # Right
    if (curr_spd - prev_spd) > eps:   return 1  # Accel
    if (prev_spd - curr_spd) > eps:   return 2  # Decel
    return 0  # Stay


def get_reward_for_state(reward_net, speed, dist, lane):
    device = next(reward_net.parameters()).device
    state = torch.tensor([[speed, dist, lane]], dtype=torch.float32, device=device)
    act = torch.zeros((1,), dtype=torch.long, device=device)
    done = torch.zeros(1, dtype=torch.bool, device=device)

    with torch.no_grad():
        rew = reward_net.forward(state, act, state, done)
    return rew.item()


def calculate_impulsivity_score(reward_net, expert_obs):
    device = next(reward_net.parameters()).device
    reward_net.eval()

    with torch.no_grad():
        # Compute normalized versions of your physical anchors
        safe_norm = get_normalized_vector(20.0, 105.0, 4.0)
        impulse_norm = get_normalized_vector(120.0, 0.0, 4.0)

        safe_state = torch.tensor([safe_norm], device=device)
        impulsive_state = torch.tensor([impulse_norm], device=device)

        act = torch.zeros((1, 1), dtype=torch.float32, device=device)
        done = torch.zeros(1, dtype=torch.bool, device=device)

        # r_min = reward_net.predict_processed(safe_state, act, safe_state, done).item()
        r_min = reward_net.forward(safe_state, act, safe_state, done).item()
        r_max = reward_net.forward(
            impulsive_state, act, impulsive_state, done
        ).item()

        exp_tensor = torch.tensor(expert_obs, dtype=torch.float32, device=device)
        exp_acts = torch.zeros((len(exp_tensor), 1), dtype=torch.float32, device=device)
        exp_dones = torch.zeros(len(exp_tensor), dtype=torch.bool, device=device)

        r_expert = (
            reward_net.forward(exp_tensor, exp_acts, exp_tensor, exp_dones)
            .mean()
            .item()
        )

    percentage = (
        0.0
        if abs(r_max - r_min) < 1e-5
        else ((r_expert - r_min) / (r_max - r_min)) * 100.0
    )
    return round(float(np.clip(percentage, 0.0, 100.0)), 2)


# ====================
# Useful Evaluation: Formatting Human Data
# ====================
def get_expert_observations(traj_buffer: list):
    return np.array(traj_buffer, dtype=np.float32)


def get_actions(expert_observations):
    actions = []

    for i in range(1, len(expert_observations)):
        action = map_raw_to_discrete(expert_observations[i], expert_observations[i - 1])
        actions.append(action)

    return np.array(actions, dtype=np.int64)


def get_infos(num_steps):
    return np.array([{} for _ in range(num_steps)])


def get_trajectory(expert_obs, acts):
    return types.Trajectory(
        obs=expert_obs, acts=acts, infos=get_infos(len(acts)), terminal=True
    )

def format_expert_data(trajectory_buffer):
    expert_obs = get_expert_observations(trajectory_buffer)

    # Calculate clean discrete proxy steps [0-4]
    discrete_acts = get_actions(expert_obs)

    # --- REMOVE THE NOISE CODES ENTIRELY ---
    # Cast acts straight to np.int64 so imitation knows they are discrete classes
    acts = discrete_acts.astype(np.int64) 

    trajectory = get_trajectory(expert_obs, acts)
    return expert_obs, acts, trajectory

def format_expert_data_from_dataframe(run_df):
    # 1. Pull observations: Speed, Distance, Lane
    expert_obs = run_df[["speed", "distance", "lane"]].to_numpy(dtype=np.float32)

    # 2. Pull actions directly from your discrete control column (cast to int64 for Discrete spaces)
    # Drop the last element to keep length at N-1 relative to N observations
    acts = run_df["control"].to_numpy(dtype=np.int64)[:-1]

    # 3. Package into standard imitation trajectory type
    trajectory = types.Trajectory(
        obs=expert_obs,
        acts=acts,
        infos=np.array([{} for _ in range(len(acts))]),
        terminal=True,
    )
    return expert_obs, acts, trajectory


def extract_weights(reward_net):
    reward_net.eval()
    device = next(reward_net.parameters()).device

    # Standard baseline observation normalized
    base_norm = get_normalized_vector(70.0, 50.0, 4.0)
    base_state = torch.tensor([base_norm], device=device, requires_grad=True)

    try:
        act = torch.zeros((1, 1), dtype=torch.float32, device=device)
        rew = reward_net.forward(
            base_state, act, base_state.clone(), torch.tensor([False], device=device)
        )
        rew.backward()
        return base_state.grad.detach().cpu().numpy().flatten()
    except Exception as e:
        print(f"Extraction Error: {e}")
        return np.array([0.0, 0.0, 0.0])


def evaluate_expert(reward_net, expert_obs, acts):
    # FIX: Map tensors safely to the device reward_net is living on
    device = next(reward_net.parameters()).device
    with torch.no_grad():
        states = torch.tensor(expert_obs[:-1], dtype=torch.float32, device=device)
        actions = torch.tensor(acts, dtype=torch.float32, device=device)
        next_states = torch.tensor(expert_obs[1:], dtype=torch.float32, device=device)
        dones = torch.zeros(len(acts), dtype=torch.bool, device=device)
        return (
            reward_net.forward(states, actions, next_states, dones)
            .mean()
            .item()
        )


def get_disc_stats(airl_trainer, expert_obs, acts):
    reward_net = airl_trainer._reward_net
    policy = airl_trainer.gen_algo.policy
    device = next(reward_net.parameters()).device

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


def save_heatmap(plt, filename):
    plt.savefig(f"{filename}.png")


def create_heatmap_axes():
    speeds = np.linspace(20, 120, 10)
    distances = np.linspace(0, 105, 10)

    return speeds, distances


def build_reward_data_array(reward_net, speeds, distances, lane=4.0):
    device = next(reward_net.parameters()).device
    reward_net.eval()
    heatmap_data = []

    with torch.no_grad():
        for d in reversed(distances):
            row_rewards = []
            for s in speeds:
                # Keep heatmap axes in clean units, but evaluate on normalized state values
                norm_state = get_normalized_vector(s, d, lane)
                state = torch.tensor([norm_state], dtype=torch.float32, device=device)
                act = torch.zeros((1, 1), dtype=torch.float32, device=device)
                done = torch.zeros(1, dtype=torch.bool, device=device)

                rew = reward_net.forward(state, act, state, done)
                row_rewards.append(rew.item())
            heatmap_data.append(row_rewards)
    return np.array(heatmap_data)


def normalize_score(value, min_val, max_val):
    if abs(max_val - min_val) < 1e-5:
        return 0.0

    percentage = ((value - min_val) / (max_val - min_val)) * 100.0
    return np.clip(percentage, 0.0, 100.0)

# ==============================================================================
# III. ENVIRONMENT AND DETACHED PHYSICS SANDBOX (Trace-Driven Playback Engine)
# ==============================================================================

def transition_state(current_raw_state, action_idx):
    """
    Computes a deterministic physical transition over dt = 0.04s (25 Hz).
    Ensures clear cause-and-effect transitions for AIRL gradient stability.
    """
    speed, distance, lane = current_raw_state[0], current_raw_state[1], current_raw_state[2]
    
    target_speed = speed
    target_lane = lane
    
    # 1. Map discrete action spaces to relative state modifications
    if action_idx == 0:    # SLOWER (Brake)
        target_speed = max(10.0, speed - 12.0)  # Safe low buffer to prevent instant freeze termination
    elif action_idx == 2:  # FASTER (Accelerate)
        target_speed = min(120.0, speed + 12.0)
    elif action_idx == 3:  # LANE_LEFT
        target_lane = max(-4.0, lane - 4.0)
    elif action_idx == 4:  # LANE_RIGHT
        target_lane = min(16.0, lane + 4.0)
        
    # 2. Physics Integration Loop via Relative Velocity Tracking
    # Ambient traffic stream moves at a constant baseline reference speed of 30.0 units/sec
    ambient_traffic_speed = 30.0
    dt = 0.04  
    
    relative_speed = speed - ambient_traffic_speed
    
    # Distance closes if agent speed > ambient speed
    new_distance = distance - (relative_speed * dt)
    
    # Procedural Traffic Loop: If agent drives past or crashes, respawn target vehicle
    if new_distance <= 0.0:
        new_distance = 450.0  # Teleport target car out ahead to simulate continuous highway tracking
        
    return [target_speed, new_distance, target_lane]

class DummyDrivingEnv(gym.Env):
    """
    A trace-driven Sandbox Environment that directly streams expert states from 
    the master dataset to ensure perfectly bounded and aligned trajectories.
    """
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.step_count = 0
        
        # Keep track of where we are in our dataset playback pointer
        self.current_row_index = 0
        self.max_rows = len(self.df)
        
        # Continuous observation space mapping back to [speed, distance, lane]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Discrete control matching your data collection setup
        self.action_space = spaces.Discrete(5)
        
        # Warm initialization fallback state
        self.raw_state = [50.0, 250.0, 4.0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        # Pick a random starting point in the CSV trajectory to start playback
        # Leave a buffer of 200 steps so the episode has room to play out
        if self.max_rows > 200:
            self.current_row_index = np.random.randint(0, self.max_rows - 200)
        else:
            self.current_row_index = 0
            
        # Initialize our state vector to the CSV row data
        row_data = self.df.iloc[self.current_row_index]
        self.raw_state = [
            float(row_data['speed']),
            float(row_data['distance']),
            float(row_data['lane'])
        ]
        
        return get_normalized_vector(*self.raw_state), {}

    def step(self, action):
        self.step_count += 1
        self.current_row_index += 1  # Advance the playback pointer forward 1 frame
        
        # --- 1. Edge Case: Check if playback hit the end of the file ---
        if self.current_row_index >= self.max_rows:
            # Loop around to the start if we overflow the dataset bounds
            self.current_row_index = 0
            
        # --- 2. Read state directly from the next historical row ---
        row_data = self.df.iloc[self.current_row_index]
        csv_speed = float(row_data['speed'])
        csv_distance = float(row_data['distance'])
        csv_lane = float(row_data['lane'])
        
        # --- 3. Let actions exert subtle control over the streamed timeline ---
        # Instead of ignoring the agent's choice, let its actions slightly shift 
        # the baseline trajectory to keep the MDP optimization active.
        action_idx = int(action)
        speed_modifier = 0.0
        lane_modifier = 0.0
        
        if action_idx == 0:    # SLOWER
            speed_modifier = -5.0
        elif action_idx == 2:  # FASTER
            speed_modifier = 5.0
        elif action_idx == 3:  # LANE_LEFT
            lane_modifier = -2.0
        elif action_idx == 4:  # LANE_RIGHT
            lane_modifier = 2.0
            
        # Combine historical expert trends with real-time generator action modifications
        self.raw_state[0] = np.clip(csv_speed + speed_modifier, 0.0, 120.0)
        self.raw_state[1] = np.clip(csv_distance, 0.0, 500.0) # Keep distance locked to reality
        self.raw_state[2] = np.clip(csv_lane + lane_modifier, -4.0, 16.0)
        
        # --- 4. Structural Termination Rules ---
        terminated = False
        
        if self.raw_state[1] <= 5.0:    # Collision event found in historical log
            terminated = True
        if self.step_count >= 150:      # Cap episode lengths to build tight trajectory cuts
            terminated = True
            
        return get_normalized_vector(*self.raw_state), 0.0, terminated, False, {}

# ====================
# Unreal Engine Bridge (Main Server Object)
# ====================
class UnrealPostGameBridge:
    def __init__(self, hostname="127.0.0.1", port=3000):
        self.host = hostname
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.trajectory_buffer = []
        self.episode_count = 0

    def start_server(self):
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(1)

        except Exception as e:
            print(f"Server Error: {e}")

        return

    # Do we have enough training data?
    @staticmethod
    def check_episode_length_viability(traj_buffer: list) -> bool:
        if len(traj_buffer) < 10:
            return False
        else:
            return True

    @staticmethod
    def build_airl_trainer(venv, trajectory, hidden_sizes=(64, 64), lr=5e-4, batch_size=512):
        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            batch_size=batch_size,
            n_steps=128,
            ent_coef=0.1,
            learning_rate=lr,
            n_epochs=10,
            verbose=0,
            policy_kwargs=dict(net_arch=dict(pi=list(hidden_sizes), vf=list(hidden_sizes)))
        )
        reward_net = BasicRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            use_state=True,
            use_action=True,
            use_next_state=False,
            use_done=False,
            hid_sizes=hidden_sizes,
        )
        airl_trainer = AIRL(
            demonstrations=[trajectory],
            demo_batch_size=12,
            gen_replay_buffer_capacity=1000,
            n_disc_updates_per_round=1,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True,
        )
        return learner, reward_net, airl_trainer

    @staticmethod
    def run_airl_training(airl_trainer, expert_obs, acts):
        for _ in tqdm.tqdm(range(1), desc="AIRL Training", unit="chunk"):
            airl_trainer.train(total_timesteps=int(32768 / 16))
            acc, loss = get_disc_stats(airl_trainer, expert_obs, acts)
        return acc, loss


# ====================
# Code Start : Imperative Shell
# ====================
if __name__ == "__main__":
    # Setup Server
    hostname = "127.0.0.1"
    port = 3000
    server = UnrealPostGameBridge()
    server.start_server()

    # Record data + trigger AIRL afterwards
    while True:
        print()
        print(f"Ready for Game Session on {hostname}:{port}...")
        print(f"Please start the game.")
        conn, addr = server.sock.accept()
        with conn:
            print(f"\nGame Client Connected: {addr}")
            buffer = ""
            json_decoder = json.JSONDecoder()

            # Handle Session
            while True:
                try:
                    data = conn.recv(1048576)
                    if not data:
                        break

                    buffer += data.decode("utf-8") + "\n"

                    buffer = buffer.strip()

                    # While new line exists...
                    while buffer:
                        # Split the buffer into multiple lines

                        # raw_decode reads only up to the closing bracket of the FIRST valid JSON object
                        packet, idx = json_decoder.raw_decode(buffer)

                        # Slice out what we just parsed, leave the rest in the buffer
                        buffer = buffer[idx:].strip()

                        msg_type = packet.get("type", "state")

                        # Locate inside your `while True:` loop where `msg_type == "state"` is verified:
                        if msg_type == "state":
                            if (
                                isinstance(packet.get("data"), list)
                                and len(packet["data"]) >= 3
                            ):
                                raw_data = packet["data"]
                                speed = raw_data[0]
                                dist = (
                                    raw_data[1] / 100.0
                                )  # Scale down Unreal centimeters to meters
                                lane = raw_data[2]

                                # Apply normalization directly before committing to the buffer
                                normalized_state = get_normalized_vector(
                                    speed, dist, lane
                                )
                                server.trajectory_buffer.append(
                                    normalized_state.tolist()
                                )

                        # Else if field type is "level_complete"
                        elif msg_type == "level_complete":
                            # Increment the episode_count by 1
                            server.episode_count += 1

                            print(
                                f"Level Complete! Analyzing {len(server.trajectory_buffer)} frames..."
                            )

                            # If we have enough training data...
                            if server.check_episode_length_viability(
                                server.trajectory_buffer
                            ):
                                expert_obs, acts, trajectory = format_expert_data(
                                    server.trajectory_buffer
                                )

                                # Strip away the VecNormalize wrapper to maintain consistent, unscaled values
                                venv = DummyVecEnv([lambda: DummyDrivingEnv()])

                                temp_result = server.build_airl_trainer(venv, trajectory)
                                if temp_result is None:
                                    print(
                                        "CRITICAL ERROR: build_airl_trainer returned None."
                                    )

                                learner, reward_net, airl_trainer = temp_result

                                print(
                                    "\nTraining AIRL (Generator vs. Discriminator)..."
                                )

                                # Train an AIRL model on data
                                score, weights = server.run_airl_training(airl_trainer, expert_obs, acts)
                            else:
                                score, weights = 0.0, [0.0, 0.0, 0.0]

                            weights = extract_weights(reward_net)
                            score = calculate_impulsivity_score(reward_net, expert_obs)

                            # generate_reward_heatmap(reward_net, server.episode_count)

                            speeds, distances = create_heatmap_axes()

                            data_array = build_reward_data_array(
                                reward_net, speeds, distances
                            )

                            print("\n=== TERMINAL REWARD MAP ===")
                            min_val, max_val = data_array.min(), data_array.max()
                            for row in data_array:
                                line = ""
                                for val in row:
                                    ratio = (val - min_val) / (max_val - min_val + 1e-6)
                                    r = int(255 * ratio)
                                    b = int(255 * (1 - ratio))
                                    line += f"\033[48;2;{r};0;{b}m {val:>5.2f} \033[0m"
                                print(line)

                            plt.figure(figsize=(10, 8))
                            sns.heatmap(
                                data_array,
                                annot=True,
                                xticklabels=np.round(speeds).astype(int),
                                yticklabels=np.round(distances[::-1]).astype(int),
                                cmap="coolwarm",
                            )
                            plt.title(
                                f"AIRL Reward Heatmap - Episode {server.episode_count}"
                            )
                            plt.xlabel("Speed (km/h)")
                            plt.ylabel("Distance to Lead Car (m)")

                            save_heatmap(
                                plt, f"reward_heatmap_ep{server.episode_count}"
                            )

                            plt.close()

                            # Generator vs Expert Comparison Data Parsing
                            gen_samples = (
                                airl_trainer.gen_algo.rollout_buffer.observations
                            )
                            gen_obs = gen_samples.reshape(-1, gen_samples.shape[-1])

                            avg_expert = expert_obs.mean(axis=0)
                            avg_gen = gen_obs.mean(axis=0)

                            print("\n=== AIRL Comparison Table ===")
                            print(
                                f"{'METRIC':<15} | {'EXPERT MEAN':<15} | {'GEN MEAN':<15}"
                            )
                            print("-" * 50)
                            print(
                                f"{'Speed (x)':<15} | {avg_expert[0]:<15.4f} | {avg_gen[0]:<15.4f}"
                            )
                            print(
                                f"{'Distance (y)':<15} | {avg_expert[1]:<15.4f} | {avg_gen[1]:<15.4f}"
                            )
                            print(
                                f"{'Lane Pos (z)':<15} | {avg_expert[2]:<15.4f} | {avg_gen[2]:<15.4f}"
                            )
                            print("=" * 50 + "\n")

                            # Create result object
                            result = {
                                "type": "result",
                                "impulsivity_score": float(score),
                                "details": f"Spd:{weights[0]:.2f} Safe:{weights[1]:.2f}",
                            }

                            # Send the results back to game
                            conn.sendall((json.dumps(result) + "\n").encode("utf-8"))

                            print(f"Sent Final Score: {score}\n")

                except Exception as e:
                    import traceback

                    print(f"Session Error: {e}")
                    traceback.print_exc()
                    break

        print("Session Ended. Waiting for next player...")
