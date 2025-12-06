import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """
    간단한 MLP 기반 Q-network
    입력: state (4차원, CartPole)
    출력: 각 action(2개)에 대한 Q값
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        """
        x: (batch_size, state_dim) 텐서
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        return q_values
