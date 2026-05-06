# Imports

# General
import socket
import json
import pandas as pd
import numpy as np
import torch
import tqdm

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


# Gym state: Speed, Distance, LaneDev
class DummyDrivingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0

    # Reset environment to starting state
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([50.0 / 150.0, 20.0 / 100.0, 0.0], dtype=np.float32)
        self.state += np.random.normal(0, 0.02, size=3)
        self.step_count = 0
        return self.state, {}

    # Run one timestep with agent actions
    def step(self, action):
        self.step_count += 1

        # 1. Update Speed
        self.state[0] = np.clip(self.state[0] + (action[0] * 0.05), 0.16, 1.0)

        # 2. Update Distance
        relative_speed = self.state[0] - 0.25
        dist_change = (relative_speed * 240.0 * 0.2) / 105.0

        # Calculate new distance
        new_dist = self.state[1] - dist_change

        # 3. Collision Logic: The "Stuck" State
        if new_dist <= 0:
            self.state[1] = 0.0  # Bumper to bumper
            self.state[0] = 0.25  # Force speed to match traffic (cannot pass)
            terminated = True  # End episode
        else:
            self.state[1] = np.clip(new_dist, 0.0, 1.0)
            terminated = bool(self.step_count >= 100)

        self.state[2] = np.clip(self.state[2] + action[1] * 0.1, -1.0, 1.0)

        return self.state, 0.0, terminated, False, {}


def get_reward_for_state(reward_net, speed, dist, lane):
    # Create a state tensor [Speed, Distance, Lane]
    state = torch.tensor([[speed, dist, lane]], dtype=torch.float32)
    # Dummy action and next_state for the prediction call
    act = torch.zeros((1, 2))
    done = torch.zeros(1, dtype=torch.bool)

    with torch.no_grad():
        # predict_processed returns the output of the Reward Network
        rew = reward_net.predict_processed(state, act, state, done)
    return rew.item()


def calculate_impulsivity_score(reward_net):
    # Use the sensitivity (weights) we already extracted
    weights = extract_weights(reward_net)

    # In the paper, impulsivity is linked to valuing Speed (weight[0])
    # more than Safety/Distance (weight[1]).
    # We look for a high ratio of Speed weight to Distance weight.
    speed_impulse = weights[0]
    distance_safety = weights[1]

    # We calculate the score based on how much the agent 'ignores' distance
    # to maintain speed.
    raw_score = speed_impulse - distance_safety

    # Normalize to a 0.0 - 10.0 scale for Unreal
    final_score = np.clip(raw_score * 100.0, 0.0, 10.0)
    return round(float(final_score), 2)


def format_expert_data(trajectory_buffer):
    expert_obs = np.array(trajectory_buffer, dtype=np.float32)

    # 1. Speed: Paper uses max 120
    # expert_obs[:, 0] = np.clip(expert_obs[:, 0], 20, 120) / 240.0
    # expert_obs[:, 0] = expert_obs[:, 0] / 200.0
    
    # expert_obs[:, 0] = expert_obs[:, 0] 

    # 2. Distance: Paper uses max 105 (crucial for weight calculation!)
    # expert_obs[:, 1] = np.clip(expert_obs[:, 1], 0, 105) / 10.50
    # expert_obs[:, 1] = expert_obs[:, 1] / 150.0
    
    # expert_obs[:, 1] = expert_obs[:, 1] 

    # 3. Position: Normalize -375/0/375 to -1.0/0.0/1.0
    # expert_obs[:, 2] /= 375.0
    
    # expert_obs[:, 2] = expert_obs[:, 2] 

    # Note: Actions in AIRL are shifts between states
    # acts = expert_obs[1:, :2] - expert_obs[:-1, :2]
    acts = expert_obs[1:, [0, 2]] - expert_obs[:-1, [0, 2]]
    infos = np.array([{}] * len(acts))
    trajectory = types.Trajectory(obs=expert_obs, acts=acts, infos=infos, terminal=True)
    return expert_obs, acts, trajectory


# Creates generator and reward networks
def build_airl_trainer(venv, trajectory):
    learner = PPO(
        env=venv,
        policy=MlpPolicy,
        batch_size=128,
        n_steps=1024,
        ent_coef=0.05,
        learning_rate=1e-3,
        n_epochs=5, 
        verbose=0,
    )
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True,
        use_action=False,
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
    # Find where the model lives (CPU or CUDA)
    device = next(reward_net.parameters()).device

    # Create test state on the SAME device
    base_state = torch.tensor([[0.5, 0.9, 0.0]], device=device, requires_grad=True)
    act = torch.zeros((1, 2), device=device)
    done = torch.tensor([False], device=device)

    try:
        # Get raw reward and backpropagate to find "Sensitivity"
        rew = reward_net.forward(base_state, act, base_state, done)
        rew.backward()

        # Pull gradient back to CPU for the score calculation
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
                            # Ensure data format matches Gym
                            # Unreal order: Speed, Distance, LaneDev
                            if len(packet["data"]) >= 3:
                                formatted_data = packet["data"][:3]
                                self.trajectory_buffer.append(formatted_data)

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
                print(f"Session Error: {e}")
                break

    # Executes training and extracts metrics
    def run_airl_training(self):
        ep_length = len(self.trajectory_buffer)
        if ep_length < 10:
            print("Not enough frames to train.")
            return 0.0, [0.0, 0.0, 0.0]

        self.episode_count += 1

        expert_obs, acts, trajectory = format_expert_data(self.trajectory_buffer)
        venv = DummyVecEnv([lambda: DummyDrivingEnv()])
        learner, reward_net, airl_trainer = build_airl_trainer(venv, trajectory)

        # airl_trainer.logger = configure(None, ["stdout"])

        metrics_history = []
        print("\nTraining AIRL (Generator vs. Discriminator)...")
        for i in tqdm.tqdm(range(3), desc="AIRL Training", unit="chunk"):
            # airl_trainer.train(total_timesteps=8192 * 2)
            airl_trainer.train(total_timesteps=2048)

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

        # Change this line in run_airl_training:
        weights = extract_weights(reward_net)
        score = calculate_impulsivity_score(reward_net)  # Now passing reward_net
        mean_reward = evaluate_expert(reward_net, expert_obs, acts)

        # Collect Generator Observations:
        gen_samples = airl_trainer.gen_algo.rollout_buffer.observations
        # Flatten buffer: (returns): [Speed, Distance, Lane]
        gen_obs = gen_samples.reshape(-1, gen_samples.shape[-1])

        avg_expert = expert_obs.mean(axis=0)
        avg_gen = gen_obs.mean(axis=0)

        print(get_reward_for_state(reward_net, 0.5, 0.5, 0.0))

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


if __name__ == "__main__":
    server = UnrealPostGameBridge()
    server.start_server()
