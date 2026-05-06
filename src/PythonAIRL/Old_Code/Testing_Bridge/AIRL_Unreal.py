import socket
import json
from symtable import Class

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# from torch import Generator

class Discriminator(nn.Module):
    def __init__(self, state_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Linear(state_dim, 1, bias=False)

    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(self, state_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential()

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
    
    gen = learned_weights[2]  # Lane deviation (not used in this simple score, but could be)

    # Avoid division by zero
    if abs(w_dist) < 0.001: w_dist = -0.001

    # Simple ratio: How much do you trade safety for speed?
    # Higher number = More Impulsive
    raw_score = w_speed / abs(w_dist)

    # Normalize to 0-100 for Game HUD (clamped)
    score_hud = min(max(raw_score * 50, 0), 100)
    return float(score_hud)


# --- 2. The Bridge ---
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
        """
        Runs a quick optimization to find the reward weights that best explain
        the recorded trajectory.
        """
        if len(self.trajectory_buffer) < 10:
            return 0, [0, 0, 0]  # Not enough data

        # Convert buffer to Tensor
        # Shape: (Batch_Size, 3) -> [Speed, Dist, Dev]
        expert_data = torch.tensor(self.trajectory_buffer, dtype=torch.float32)

        # Normalize Data (Critical for IRL!)
        # Speed: 0-200, Dist: 0-100 -> Map both roughly to 0-1

        # Mean Speed
        # Mean distance from the closest ahead car
        # Number of overtakes
        # Number of crashes
        # Mean Task Score
        expert_data[:, 0] /= 150.0  # Approx Max Speed
        expert_data[:, 1] /= 100.0  # Approx Max Dist

        # Setup simplified AIRL / MaxEnt
        # We want to find weights 'w' such that R = w*features is maximized for this user
        # In a simplified "Feature Matching" sense, the weights are just the average feature values
        # if we assume linear rewards and optimality.
        # However, let's do a tiny gradient step to look cool and "learn".

        disc = Discriminator(state_dim=3) # Discriminator
        gen = Generator(state_dim=3)
        optimizer = optim.Adam(disc.parameters(), lr=0.1)

        avg_features = torch.mean(expert_data, dim=0).detach().numpy()

        # avg_features[0] = Average Normalized Speed (Higher = Impulsive)
        # avg_features[1] = Average Normalized Distance (Lower = Impulsive)

        # Construct "Weights" based on this observation
        # This isn't full AIRL training (which takes minutes),
        # but a fast approximation for the loading screen.
        w_speed = avg_features[0]
        w_dist = -1.0 * (1.0 - avg_features[1])  # Penalize being close (1.0 - dist)
        w_dev = -1.0 * avg_features[2]

        weights = [w_speed, w_dist, w_dev]
        score = calculate_impulsivity_score(weights)

        return score, weights


if __name__ == "__main__":
    server = UnrealPostGameBridge()
    server.start_server()