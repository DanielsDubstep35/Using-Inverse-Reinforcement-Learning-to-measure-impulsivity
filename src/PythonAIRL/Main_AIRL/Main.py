import socket
import json
from symtable import Class

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch import Generator

# AIRL Model
# The discriminator gets the reward

def calculate_impulsivity_score(learned_weights):
    """
    Interpret the weights to get a score.
    Weights order assumed: [Speed, DistanceToLead, LaneDeviation]

    Theory:
    - Speed weight > 0: Likes going fast
    - Distance weight < 0: Dislikes being close (Safety)

    Impulsivity = (Weight_Speed) / abs(Weight_Distance)
    """
    w_speed = learned_weights[0]
    w_dist = learned_weights[1]  # Usually negative (penalty for being close)
    
    gen = learned_weights[2]  # Lane deviation (not used in score)

    # Avoid division by zero
    if abs(w_dist) < 0.001: w_dist = -0.001

    # Higher number = More Impulsive
    raw_score = w_speed / abs(w_dist)

    # Normalize to 0-100 for Game HUD
    score_hud = min(max(raw_score * 50, 0), 100)
    return float(score_hud)


# Bridge to Unreal Engine
class UnrealPostGameBridge:
    def __init__(self, hostname="127.0.0.1", port=3000):
        self.host = hostname
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # We need:
        # Mean Speed
        # Mean distance from the closest ahead car
        # Number of overtakes
        # Number of crashes
        # Mean Task Score
        self.trajectory_buffer = []  # Stores [speed, dist, dev] for every tick

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
                self.trajectory_buffer = []  # Reset for new game

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

                    # Parse JSON
                    packet = json.loads(message)

                    print(f"DATA Received: {packet}")

                    # Check Packet Type
                    msg_type = packet.get("type", "state")
                    print(f"Received Message: {msg_type}")

                    if msg_type == "state":
                        # RECORD DATA
                        # Expecting: {"type": "state", "data": [80.5, 12.0, 0.1]}
                        if isinstance(packet.get("data"), list):
                            self.trajectory_buffer.append(packet["data"])

                    elif msg_type == "level_complete":
                        # ANALYZE DATA
                        print(f"Level Complete! Analyzing {len(self.trajectory_buffer)} frames...")

                        score, weights = self.run_analysis()

                        # Send Result back to Unreal
                        result = {
                            "type": "result",
                            "impulsivity_score": float(score),
                            "details": f"Spd:{weights[0]:.2f} Safe:{weights[1]:.2f}"
                        }
                        conn.sendall((json.dumps(result) + "\n").encode('utf-8'))
                        print(f"Sent Score: {score}\n")
                        return  # End session
                    else:
                        print(f"Unknown Message...\n")

            except Exception as e:
                print(f"Session Error: {e}")
                break

    def run_analysis(self):

        if len(self.trajectory_buffer) < 10:
            return 0, [0, 0, 0]  # not enough frames

        expert = torch.tensor(self.trajectory_buffer, dtype=torch.float32)  # load expert data

        expert[:, 0] /= 150.0  # normalize speed feature
        expert[:, 1] /= 100.0  # normalize distance feature

        states = expert[:-1]  # current states batch
        next_states = expert[1:]  # next states batch

        # dummy expert actions
        expert_actions = torch.zeros((len(states), 2))  

        # --- Define discriminator ---
        class Discriminator(nn.Module):
            def __init__(self, state_dim):
                super().__init__()
                self.reward = nn.Linear(state_dim, 1)  # reward network layer
                self.value = nn.Linear(state_dim, 1)   # value network layer

            def forward(self, s, s_next, log_pi):
                r = self.reward(s)  # compute state reward
                v = self.value(s)   # compute value estimate
                v_next = self.value(s_next)  # next state value

                f = r + 0.99 * v_next - v  # AIRL shaping term

                return torch.sigmoid(f - log_pi)  # discriminator output

        disc = Discriminator(state_dim=3)  # init discriminator model
        optimizer = optim.Adam(disc.parameters(), lr=1e-4)  # stable learning rate

        # --- Minimal policy ---
        policy = torch.distributions.Normal(
            torch.zeros(2), torch.ones(2)
        )

        def compute_log_pi(actions):
            return policy.log_prob(actions).sum(dim=1, keepdim=True)  # log probability actions

        log_exp = compute_log_pi(expert_actions)  # expert log probabilities

        # --- Fake rollout generator ---
        def rollout(n=32):
            s = states[torch.randint(0, len(states), (n,))]  # sample random states
            a = torch.randn((n, 2))  # random actions sample
            s_next = s + 0.01 * torch.randn_like(s)  # noisy transition step
            log_pi = compute_log_pi(a)  # compute action log prob
            return s, a, s_next, log_pi  # return rollout batch

        # --- AIRL training loop ---
        for _ in range(5):  # outer AIRL iterations

            gs, ga, gs2, log_gen = rollout()  # generate policy samples

            # discriminator forward passes
            d_exp = disc(states, next_states, log_exp)  
            d_gen = disc(gs, gs2, log_gen)
            
            d_exp = torch.clamp(d_exp, 1e-6, 1-1e-6)
            d_gen = torch.clamp(d_gen, 1e-6, 1-1e-6)

            # adversarial classification loss
            loss = -(torch.log(d_exp + 1e-8).mean() + torch.log(1 - d_gen + 1e-8).mean())

            optimizer.zero_grad()  # reset gradients step
            loss.backward()  # backpropagate gradients
            optimizer.step()  # update discriminator weights

        # extract learned reward weights
        weights = disc.reward.weight.detach().cpu().numpy().flatten()

        # compute impulsivity score
        score = calculate_impulsivity_score(weights)

        return score, weights

if __name__ == "__main__":
    server = UnrealPostGameBridge()
    server.start_server()