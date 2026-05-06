import socket
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# AIRL Model
# The discriminator gets the reward

def calculate_impulsivity_score(learned_weights):
    """
    Interpret the weights to get a score.
    Weights order assumed: [Speed, DistanceToLead, LaneDeviation]
    """
    w_speed = learned_weights[0]
    w_dist = learned_weights[1] 
    
    gen = learned_weights[2]  

    if abs(w_dist) < 0.001: w_dist = -0.001

    raw_score = w_speed / abs(w_dist)

    score_hud = min(max(raw_score * 50, 0), 100)
    return float(score_hud)


# Bridge to Unreal Engine
class UnrealPostGameBridge:
    def __init__(self, hostname="127.0.0.1", port=3000):
        self.host = hostname
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.trajectory_buffer = []  
        self.episode_count = 0       # Tracks the number of full games played
        self.training_history = []   # Stores data for the pandas DataFrame

    def start_server(self):
        try:
            self.sock.bind((self.host, self.port))
            self.sock.listen(1)
            print(f"Ready for Game Session on {self.host}:{self.port}...")

            while True:
                conn, addr = self.sock.accept()
                print(f"Game Client Connected: {addr}")
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
                data = conn.recv(4096)
                if not data: break

                buffer += data.decode('utf-8') + "\n"

                while "\n" in buffer:
                    message, buffer = buffer.split("\n", 1)
                    if not message.strip(): continue

                    packet = json.loads(message)

                    # Commented out verbose printing to keep the console clean for the DataFrame
                    # print(f"DATA Received: {packet}")

                    msg_type = packet.get("type", "state")

                    if msg_type == "state":
                        if isinstance(packet.get("data"), list):
                            self.trajectory_buffer.append(packet["data"])

                    elif msg_type == "level_complete":
                        print(f"Level Complete! Analyzing {len(self.trajectory_buffer)} frames...")

                        score, weights = self.run_analysis()

                        result = {
                            "type": "result",
                            "impulsivity_score": float(score),
                            "details": f"Spd:{weights[0]:.2f} Safe:{weights[1]:.2f}"
                        }
                        conn.sendall((json.dumps(result) + "\n").encode('utf-8'))
                        print(f"Sent Score: {score}\n")
                        return  
                    else:
                        print(f"Unknown Message...\n")

            except Exception as e:
                print(f"Session Error: {e}")
                break

    def run_analysis(self):
        ep_length = len(self.trajectory_buffer)
        if ep_length < 10:
            return 0, [0, 0, 0] 
            
        self.episode_count += 1

        expert = torch.tensor(self.trajectory_buffer, dtype=torch.float32)  

        expert[:, 0] /= 150.0  
        expert[:, 1] /= 100.0  

        states = expert[:-1]  
        next_states = expert[1:]  

        expert_actions = torch.zeros((len(states), 2))  

        class Discriminator(nn.Module):
            def __init__(self, state_dim):
                super().__init__()
                self.reward = nn.Linear(state_dim, 1)  
                self.value = nn.Linear(state_dim, 1)   

            def forward(self, s, s_next, log_pi):
                r = self.reward(s)  
                v = self.value(s)   
                v_next = self.value(s_next)  

                f = r + 0.99 * v_next - v  

                return torch.sigmoid(f - log_pi)  

        disc = Discriminator(state_dim=3)  
        optimizer = optim.Adam(disc.parameters(), lr=1e-4)  

        policy = torch.distributions.Normal(
            torch.zeros(2), torch.ones(2)
        )

        def compute_log_pi(actions):
            return policy.log_prob(actions).sum(dim=1, keepdim=True)  

        log_exp = compute_log_pi(expert_actions)  

        def rollout(n=32):
            s = states[torch.randint(0, len(states), (n,))]  
            a = torch.randn((n, 2))  
            s_next = s + 0.01 * torch.randn_like(s)  
            log_pi = compute_log_pi(a)  
            return s, a, s_next, log_pi  

        # --- AIRL training loop ---
        for _ in range(5):  
            gs, ga, gs2, log_gen = rollout()  

            d_exp = disc(states, next_states, log_exp)  
            d_gen = disc(gs, gs2, log_gen)
            
            d_exp = torch.clamp(d_exp, 1e-6, 1-1e-6)
            d_gen = torch.clamp(d_gen, 1e-6, 1-1e-6)

            loss = -(torch.log(d_exp + 1e-8).mean() + torch.log(1 - d_gen + 1e-8).mean())

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  

        weights = disc.reward.weight.detach().cpu().numpy().flatten()
        score = calculate_impulsivity_score(weights)

        # --- Evaluate the Learned Reward ---
        # We pass the player's states back through the newly trained reward layer
        # to see what average reward the model assigns to their gameplay.
        with torch.no_grad():
            mean_reward = disc.reward(states).mean().item()

        # --- Track and Print DataFrame ---
        self.training_history.append({
            "Episode": self.episode_count,
            "Episode Length": ep_length,
            "Mean Assessed Reward": round(mean_reward, 4),
            "Impulsivity Score": round(score, 2)
        })

        df = pd.DataFrame(self.training_history)
        print("\n=== AIRL Training Results ===")
        print(df.to_string(index=False))
        print("=============================\n")

        return score, weights

if __name__ == "__main__":
    server = UnrealPostGameBridge()
    server.start_server()