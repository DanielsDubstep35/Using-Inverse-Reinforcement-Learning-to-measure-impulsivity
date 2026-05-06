import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# --- 1. Environment (Same as before) ---
class GridWorld:
    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4
        self.state_to_idx = lambda x, y: x * size + y
        self.idx_to_state = lambda i: (i // size, i % size)
        self.P = np.zeros((self.n_states, self.n_actions, self.n_states))
        actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        for s in range(self.n_states):
            x, y = self.idx_to_state(s)
            for a, (dx, dy) in enumerate(actions):
                nx, ny = max(0, min(size - 1, x + dx)), max(0, min(size - 1, y + dy))
                self.P[s, a, self.state_to_idx(nx, ny)] = 1.0


# --- 2. The Policy Network (Generator) ---
# This replaces "Value Iteration". It learns to act directly.
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def get_action(self, state_tensor):
        probs = self.forward(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


# --- 3. The AIRL Discriminator (Reward Learner) ---
# D(s,a) = exp(f(s)) / (exp(f(s)) + pi(a|s))
class AIRLDiscriminator(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super(AIRLDiscriminator, self).__init__()
        # This network learns the REWARD function g(s)
        self.g_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)  # Outputs raw reward value
        )

    def forward(self, state, log_prob_policy):
        # Predict Reward g(s)
        g_s = self.g_net(state)

        # AIRL Formulation:
        # Discriminator output D = exp(g) / (exp(g) + pi(a|s))
        # Logit D = g(s) - log(pi(a|s))
        # We return the logit because PyTorch's BCEWithLogitsLoss is more stable
        return g_s - log_prob_policy, g_s


# --- 4. Helper: Generate Trajectories ---
def collect_trajectories(env, policy_net, n_trajs=10, max_len=10):
    trajs = []
    # GridWorld coordinates normalization
    normalize = lambda s: [env.idx_to_state(s)[0] / (env.size - 1), env.idx_to_state(s)[1] / (env.size - 1)]

    for _ in range(n_trajs):
        traj = []
        curr = np.random.randint(0, env.n_states)  # Random start
        for _ in range(max_len):
            state_in = torch.tensor(normalize(curr), dtype=torch.float32)
            action, log_prob = policy_net.get_action(state_in)

            # Step environment
            next_s = np.argmax(env.P[curr, action])
            traj.append((curr, action, log_prob))
            curr = next_s
        trajs.append(traj)
    return trajs


# --- 5. Main AIRL Training Loop ---
def train_airl(env, expert_trajs, epochs=500):
    state_dim = 2  # (x, y)
    action_dim = env.n_actions

    # Init Networks
    policy = PolicyNet(state_dim, action_dim)
    discriminator = AIRLDiscriminator(state_dim)

    # Optimizers
    opt_gen = optim.Adam(policy.parameters(), lr=0.005)
    opt_disc = optim.Adam(discriminator.parameters(), lr=0.01)

    # Prepare Expert Data (Flat list of states)
    normalize = lambda s: [env.idx_to_state(s)[0] / (env.size - 1), env.idx_to_state(s)[1] / (env.size - 1)]
    expert_states = []
    for traj in expert_trajs:
        for s, _, _ in traj:
            expert_states.append(normalize(s))
    expert_states = torch.tensor(expert_states, dtype=torch.float32)

    print("Training AIRL...")

    for epoch in range(epochs):
        # --- A. Collect Generator Trajectories ---
        gen_trajs = collect_trajectories(env, policy, n_trajs=10)

        # Flatten Generator Data
        gen_states = []
        gen_log_probs = []
        for traj in gen_trajs:
            for s, a, log_p in traj:
                gen_states.append(normalize(s))
                gen_log_probs.append(log_p)

        gen_states = torch.tensor(gen_states, dtype=torch.float32)
        gen_log_probs = torch.stack(gen_log_probs).detach()  # Detach for Discriminator training

        # --- B. Train Discriminator (The Reward Learner) ---
        opt_disc.zero_grad()

        # 1. Expert Pass: Label = 1
        # Expert policy prob is unknown, approximated as uniform or ignored in simplified AIRL
        # Here we assume a small constant log_prob for stability if unknown
        expert_logits, _ = discriminator(expert_states, torch.zeros(len(expert_states), 1))
        loss_expert = nn.BCEWithLogitsLoss()(expert_logits, torch.ones_like(expert_logits))

        # 2. Generator Pass: Label = 0
        gen_logits, predicted_rewards = discriminator(gen_states, gen_log_probs.unsqueeze(1))
        loss_gen = nn.BCEWithLogitsLoss()(gen_logits, torch.zeros_like(gen_logits))

        loss_d = loss_expert + loss_gen
        loss_d.backward()
        opt_disc.step()

        # --- C. Train Generator (Policy) with REINFORCE ---
        # The reward for the generator is the Discriminator's estimate g(s)
        opt_gen.zero_grad()

        # We need fresh log_probs with gradients attached
        policy_loss = 0
        flattened_rewards = predicted_rewards.detach().squeeze()  # Use current reward estimate

        # Standardize rewards for stability
        flattened_rewards = (flattened_rewards - flattened_rewards.mean()) / (flattened_rewards.std() + 1e-8)

        idx = 0
        for traj in gen_trajs:
            R = 0
            for i, (s, a, _) in enumerate(reversed(traj)):
                # Get fresh log_prob
                state_in = torch.tensor(normalize(s), dtype=torch.float32)
                probs = policy(state_in)
                log_prob = torch.log(probs[a])

                # Reward comes from the Discriminator (Reverse accumulated return)
                # We use the immediate reward learned by D
                r = flattened_rewards[idx + len(traj) - 1 - i].item()
                R = r + 0.99 * R  # Discounted return

                # Policy Gradient Loss: - log(pi) * Return
                policy_loss -= log_prob * R
            idx += len(traj)

        policy_loss /= len(gen_trajs)
        policy_loss.backward()
        opt_gen.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Disc Loss {loss_d.item():.4f} | Gen Reward Mean {flattened_rewards.mean():.4f}")

    return discriminator.g_net


# --- 6. Execution and Visualization ---
if __name__ == "__main__":
    # Setup
    size = 5
    env = GridWorld(size)

    # --- Generate Expert Data (Ground Truth) ---
    # Top-right corner is best
    true_rewards = np.zeros(env.n_states)
    true_rewards[env.state_to_idx(0, size - 1)] = 1.0

    # --- FIX START: Value Iteration with Correct Broadcasting ---
    V = np.zeros(env.n_states)
    for _ in range(100):
        # Calculate (Reward + Discount * Value)
        # Shape: (25, 25) -> (Start State, Next State)
        values = true_rewards.reshape(-1, 1) + 0.99 * V.reshape(1, -1)

        # Broadcast against P: (25, 4, 25) * (25, 1, 25)
        Q = np.sum(env.P * values[:, None, :], axis=2)
        V = np.max(Q, axis=1)

    # Final Q calculation for policy
    values = true_rewards.reshape(-1, 1) + 0.99 * V.reshape(1, -1)
    Q = np.sum(env.P * values[:, None, :], axis=2)
    # --- FIX END ---

    expert_policy_probs = np.zeros((env.n_states, 4))
    expert_policy_probs[np.arange(env.n_states), np.argmax(Q, axis=1)] = 1.0


    def generate_expert_trajs(n=20):
        trajs = []
        for _ in range(n):
            t = []
            curr = env.state_to_idx(4, 0)  # Start bottom left
            for _ in range(10):
                a = np.random.choice(4, p=expert_policy_probs[curr])
                ns = np.argmax(env.P[curr, a])
                t.append((curr, a, 0))  # Log prob dummy
                curr = ns
            trajs.append(t)
        return trajs


    expert_trajs = generate_expert_trajs()

    # --- Run AIRL ---
    reward_net = train_airl(env, expert_trajs)

    # --- Visualize Learned Reward ---
    learned_rewards = np.zeros((size, size))
    with torch.no_grad():
        for x in range(size):
            for y in range(size):
                state_in = torch.tensor([x / (size - 1), y / (size - 1)], dtype=torch.float32)
                r = reward_net(state_in).item()
                learned_rewards[x, y] = r

    plt.imshow(learned_rewards, cmap='hot', interpolation='nearest')
    plt.title("Learned AIRL Reward")
    plt.colorbar()
    plt.savefig("airl_result.png")
    print("Saved plot to airl_result.png")