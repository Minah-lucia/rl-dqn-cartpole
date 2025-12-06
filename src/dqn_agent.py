from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import QNetwork
from replay_buffer import ReplayBuffer


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 50_000
    batch_size: int = 32
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    target_update_interval: int = 1_000
    min_buffer_size: int = 1_000
    device: str = "cpu"


class DQNAgent:
    def __init__(self, config: DQNConfig):
        self.cfg = config

        self.device = torch.device(config.device)

        self.q_net = QNetwork(config.state_dim, config.action_dim).to(self.device)
        self.target_q_net = QNetwork(config.state_dim, config.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=config.lr)

        self.replay_buffer = ReplayBuffer(config.buffer_size)

        self.gamma = config.gamma
        self.batch_size = config.batch_size

        # epsilon 탐험 파라미터
        self.epsilon_start = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay_steps = config.epsilon_decay_steps

        self.total_steps = 0  # epsilon 및 target 업데이트에 사용

    def compute_epsilon(self) -> float:
        """
        선형 감소 epsilon 스케줄
        """
        if self.total_steps >= self.epsilon_decay_steps:
            return self.epsilon_end
        else:
            frac = self.total_steps / self.epsilon_decay_steps
            return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """
        epsilon-greedy 정책으로 action 선택
        eval_mode=True이면 greedy만 사용
        """
        if not eval_mode:
            epsilon = self.compute_epsilon()
        else:
            epsilon = 0.0

        self.total_steps += 1

        if np.random.rand() < epsilon:
            # exploration
            return np.random.randint(self.cfg.action_dim)
        else:
            # exploitation
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)  # (1, action_dim)
            action = int(torch.argmax(q_values, dim=1).item())
            return action

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        """
        Replay buffer에서 mini-batch를 샘플링해 Q-network를 한 번 업데이트.
        """
        if len(self.replay_buffer) < self.cfg.min_buffer_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)  # (B, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # (B, 1)

        # 현재 Q(s, a)
        q_values = self.q_net(states)  # (B, action_dim)
        q_value = q_values.gather(1, actions)  # (B, 1)

        # target Q 계산
        with torch.no_grad():
            target_q_values = self.target_q_net(next_states)  # (B, action_dim)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]  # (B, 1)
            target = rewards + (1.0 - dones) * self.gamma * max_target_q_values

        loss = nn.MSELoss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network 하드 업데이트 (일정 step마다)
        if self.total_steps % self.cfg.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
