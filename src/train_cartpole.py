import argparse
import os
import random
from typing import List

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from dqn_agent import DQNAgent, DQNConfig


def set_seed(env, seed: int):
    """
    재현성을 위한 seed 설정
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env.reset(seed=seed)
    env.action_space.seed(seed)


def train_dqn(
    episodes: int = 500,
    max_steps_per_episode: int = 500,
    seed: int = 0,
    render: bool = False,
    save_dir: str = "./results",
):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)

    set_seed(env, seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50_000,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=50_000,
        target_update_interval=1_000,
        min_buffer_size=1_000,
        device=device,
    )

    agent = DQNAgent(cfg)

    episode_returns: List[float] = []

    os.makedirs(save_dir, exist_ok=True)

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for t in range(max_steps_per_episode):
            action = agent.select_action(state, eval_mode=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        episode_returns.append(total_reward)

        # 콘솔 로그
        if episode % 10 == 0:
            avg_return = np.mean(episode_returns[-10:])
            print(
                f"[Episode {episode:4d}] "
                f"Return: {total_reward:6.1f} | "
                f"Last 10 avg: {avg_return:6.2f}"
            )

    env.close()

    # 결과 저장
    returns_path = os.path.join(save_dir, f"returns_seed{seed}.npy")
    np.save(returns_path, np.array(episode_returns))
    print(f"Saved returns to {returns_path}")

    # 학습 곡선 플롯
    plt.figure(figsize=(8, 5))
    plt.plot(episode_returns, label=f"seed={seed}")
    window = min(50, len(episode_returns))
    if window > 1:
        moving_avg = np.convolve(
            episode_returns,
            np.ones(window) / window,
            mode="valid",
        )
        plt.plot(
            range(window - 1, window - 1 + len(moving_avg)),
            moving_avg,
            label=f"moving avg (window={window})",
        )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("DQN on CartPole-v1")
    plt.legend()
    plt.grid(True)

    plot_path = os.path.join(save_dir, f"dqn_cartpole_seed{seed}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DQN CartPole-v1 Reproduction")
    parser.add_argument("--episodes", type=int, default=500, help="number of training episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--render", action="store_true", help="enable rendering")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./results",
        help="directory to save logs and plots",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_dqn(
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        seed=args.seed,
        render=args.render,
        save_dir=args.save_dir,
    )
