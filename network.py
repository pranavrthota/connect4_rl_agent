"""Dual-head policy/value network: a small residual CNN trunk over the
6x7 board, with a policy head (move distribution) and value head
(expected outcome from the current player's perspective).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import COLS, N_ACTIONS, ROWS


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels=2, channels=64, n_res_blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(n_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ROWS * COLS, N_ACTIONS),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * ROWS * COLS, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, encoded_state, device):
        """encoded_state: (2, ROWS, COLS) numpy array for a single position.
        Returns (policy_probs: np.array of length N_ACTIONS, value: float).
        """
        x = torch.tensor(encoded_state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, float(value.item())

    @torch.no_grad()
    def predict_batch(self, encoded_states, device):
        """encoded_states: (N, 2, ROWS, COLS) numpy array. One GPU forward
        pass for all N positions at once — this is the key lever for GPU
        utilization, since batch-1 inference is latency- not compute-bound.
        Returns (probs: (N, N_ACTIONS) np.array, values: (N,) np.array).
        """
        x = torch.tensor(encoded_states, dtype=torch.float32, device=device)
        logits, values = self.forward(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs, values.cpu().numpy()


class NetworkEvaluator:
    """Adapts a PolicyValueNet to MCTS's evaluator interface: .evaluate(game)."""

    def __init__(self, network, device):
        self.network = network
        self.device = device

    def evaluate(self, game):
        return self.network.predict(game.encode(), self.device)
