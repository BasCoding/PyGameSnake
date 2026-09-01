import argparse
import os

import torch

from pygamesnake.reinforcement_learning.agent import Agent
from pygamesnake.reinforcement_learning.environment import SnakeEnv


def train(n_games=200, speed=40, render=True, save_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(device)
    env = SnakeEnv(render=render, speed=speed)

    total_score = 0
    record = 0

    while agent.n_games < n_games:
        env.hud_lines = [f"Game {agent.n_games + 1}/{n_games}", f"Record {record}"]

        # 1) observe -> 2) act -> 3) observe result
        state_old = env.get_state()
        action = agent.get_action(state_old)
        reward, done, score = env.step(action)
        state_new = env.get_state()

        # 4) learn from this single step, and store it for later replay
        agent.train_short_memory(state_old, action, reward, state_new, done)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()  # replay past experience after each game

            record = max(record, score)
            total_score += score
            mean_score = total_score / agent.n_games
            print(f"Game {agent.n_games}  Score {score}  Record {record}  Mean {mean_score:.2f}")

    if save_path:
        agent.model.save(save_path)
        print(f"Training finished. Best score: {record}. Model saved to {save_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a snake agent with reinforcement learning (Deep Q-Learning)')
    parser.add_argument('--games', type=int, default=200, help='Number of games to train for')
    parser.add_argument('--speed', type=int, default=40,
                        help='Rendering speed in frames per second; higher is faster')
    parser.add_argument('--no-render', action='store_true',
                        help='Train without the game window (much faster)')
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    save_path = os.path.join('models', 'rl_model')
    train(n_games=args.games, speed=args.speed, render=not args.no_render, save_path=save_path)
