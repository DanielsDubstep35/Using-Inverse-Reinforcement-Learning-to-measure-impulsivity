import socket
import json
import pandas as pd
import numpy as np
import torch
import tqdm
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicRewardNet
import imitation.data.types as types



# 1. Dummy Gym Environment for the Generator
class DummyDrivingEnv(gym.Env):
    """
    SB3 PPO needs a live environment to generate fake data. 
    Since Unreal only sends offline post-game data, we mock a sandbox here.
    """
    def __init__(self):
        super().__init__()
        # Obs: [Speed, Distance, LaneDev]
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(3,), dtype=np.float32)
        # Action: [Throttle/Brake, Steering]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.state = np.zeros(3, dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start with normalized values (50/150 for speed, 20/100 for distance)
        self.state = np.array([50.0 / 150.0, 20.0 / 100.0, 0.0], dtype=np.float32)
        self.state += np.random.normal(0, 0.02, size=3)
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.step_count += 1
        
        # Apply physics changes in the NORMALIZED scale
        self.state[0] += (action[0] * 2.0) / 150.0  # Normalized speed change
        self.state[1] += (action[0] * 1.0) / 100.0  # Normalized distance change
        self.state[2] += action[1] * 0.1            # Lane dev (already small)
        # self.state[3] += action[1] * 0.1            # Crashes (already small)
        # self.state[4] += action[1] * 0.1            # Overtakes (already small)
        
        terminated = bool(self.step_count >= 100)
        truncated = False
        reward = 0.0 # Handled entirely by AIRL
        
        return self.state, reward, terminated, truncated, {}



# 2. AIRL Setup and Utilities
def calculate_impulsivity_score(learned_weights):
    """
    Interpret the weights to get a score.
    Weights order assumed: [Speed, DistanceToLead, LaneDeviation]
    """
    if len(learned_weights) < 3:
        return 0.0

    w_speed = learned_weights[0]
    w_dist = learned_weights[1] 
    
    # Avoid div by zero
    if abs(w_dist) < 0.001: 
        w_dist = -0.001 if w_dist <= 0 else 0.001

    raw_score = w_speed / abs(w_dist)
    score_hud = min(max(raw_score * 50, 0), 100)
    return float(score_hud)



# 3. Unreal Engine Bridge
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
                if not data: break

                buffer += data.decode('utf-8') + "\n"

                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    if not message.strip(): continue

                    packet = json.loads(message)
                    msg_type = packet.get("type", "state")

                    if msg_type == "state":
                        if isinstance(packet.get("data"), list):
                            
                            # time, speed, score (CURRENTLY)
                            # print(packet["data"])
                            
                            self.trajectory_buffer.append(packet["data"])

                    elif msg_type == "level_complete":
                        print(f"Level Complete! Analyzing {len(self.trajectory_buffer)} frames...")
                        score, weights = self.run_airl_training()

                        result = {
                            "type": "result",
                            "impulsivity_score": float(score),
                            "details": f"Spd:{weights[0]:.2f} Safe:{weights[1]:.2f}"
                        }
                        conn.sendall((json.dumps(result) + "\n").encode('utf-8'))
                        print(f"Sent Final Score: {score}\n")
                        
                        
                        
                        return  

            except Exception as e:
                print(f"Session Error: {e}")
                break

    def run_airl_training(self):
        ep_length = len(self.trajectory_buffer)
        if ep_length < 10:
            print("Not enough frames to train.")
            return 0.0, [0.0, 0.0, 0.0] 
            
        self.episode_count += 1



        # 1. Format Expert Data for imitation library
        expert_obs = np.array(self.trajectory_buffer, dtype=np.float32)
        
        # Normalize features like the original script
        expert_obs[:, 0] /= 150.0  
        expert_obs[:, 1] /= 100.0  

        # imitation requires dummy actions and infos for the trajectory
        # acts = np.zeros((len(expert_obs) - 1, 2), dtype=np.float32) 
        acts = expert_obs[1:, :2] - expert_obs[:-1, :2]
        
        infos = np.array([{}] * len(acts))

        trajectory = types.Trajectory(
            obs=expert_obs,
            acts=acts,
            infos=infos,
            terminal=True
        )



        # 2. Setup Vectorized Sandbox Environment
        venv = DummyVecEnv([lambda: DummyDrivingEnv()])

        # 3. Setup PPO Generator
        # Setting verbose=0 to let tqdm handle the console output cleanly
        learner = PPO(
            env=venv,
            policy=MlpPolicy,
            batch_size=64,
            ent_coef=0.0,
            learning_rate=0.0003,
            n_epochs=5,
            verbose=0 
        )
        
        

        # 4. Setup Reward Network (Discriminator Core)
        # hid_sizes=() forces it to be a single linear layer so we can extract exact weights
        reward_net = BasicRewardNet(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            use_state=True,
            use_action=True,
            use_next_state=False,
            use_done=False,
            hid_sizes=(32, 32)
        )



        # 5. Initialize AIRL
        airl_trainer = AIRL(
            demonstrations=[trajectory],
            demo_batch_size=64,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=8,
            venv=venv,
            gen_algo=learner,
            reward_net=reward_net,
            allow_variable_horizon=True
        )



        # 6. Train with tqdm Progress Bar
        total_timesteps = 50000
        intervals = 10
        steps_per_interval = 2048

        print("\nTraining AIRL (Generator vs. Discriminator)...")
        for _ in tqdm.tqdm(range(intervals), desc="AIRL Training", unit="chunk"):
            airl_trainer.train(total_timesteps=steps_per_interval)



        # 7. Extract the exact linear weights from the BasicRewardNet
        weights = [0.0, 0.0, 0.0]
        for param in reward_net.parameters():
            # Find the 1x3 weight matrix connected to our state inputs
            if param.shape == (1, 3) or param.shape == (3,):
                weights = param.detach().cpu().numpy().flatten()
                break

        score = calculate_impulsivity_score(weights)



        # Calculate Mean Assessed Reward on the expert data
        with torch.no_grad():
            states_tensor = torch.tensor(expert_obs[:-1], dtype=torch.float32)
            actions_tensor = torch.tensor(acts, dtype=torch.float32)
            next_states_tensor = torch.tensor(expert_obs[1:], dtype=torch.float32)
            dones_tensor = torch.zeros(len(acts), dtype=torch.bool)
            
            # Use reward_net to predict rewards for the player's states
            mean_reward = reward_net.predict_processed(
                states_tensor, actions_tensor, next_states_tensor, dones_tensor
            ).mean().item()



        # 8. Track and Print DataFrame
        self.training_history.append({
            "Episode": self.episode_count,
            "Frames": ep_length,
            "Mean Reward": round(mean_reward, 4),
            "w_speed": round(weights[0], 4),
            "w_dist": round(weights[1], 4),
            "Impulsivity": round(score, 2)
        })

        df = pd.DataFrame(self.training_history)
        print("\n=== AIRL Training Results ===")
        print(df.to_string(index=False))
        print("=============================\n")
        
        print("\n\nExpert obs mean:", expert_obs.mean(axis=0))
        print("Gen obs mean:", venv.reset()[0])
        print()

        return score, weights

if __name__ == "__main__":
    server = UnrealPostGameBridge()
    server.start_server()