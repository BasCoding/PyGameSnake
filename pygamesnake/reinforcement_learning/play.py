import argparse
import os

import numpy as np
import torch

from pygamesnake.reinforcement_learning.agent import Linear_QNet
from pygamesnake.reinforcement_learning.environment import SnakeEnv


def play(model_path, games=10, speed=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Linear_QNet().to(device)
    model.load(model_path, device)
    model.eval()

    env = SnakeEnv(render=True, speed=speed)
    played = 0
    while played < games:
        env.hud_lines = [f"Game {played + 1}/{games}", f"Score {env.snake.SnakeLength}"]

        state = env.get_state()
        with torch.no_grad():
            state0 = torch.tensor(np.array(state), dtype=torch.float32, device=device)
            move = torch.argmax(model(state0)).item()
        action = [0, 0, 0]
        action[move] = 1

        _, done, score = env.step(action)
        if done:
            print(f"Game {played + 1}  Score {score}")
            env.reset()
            played += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Watch a trained reinforcement-learning snake agent play')
    parser.add_argument('--model', default=os.path.join('models', 'rl_model'),
                        help='Path to the trained RL model')
    parser.add_argument('--games', type=int, default=10, help='Number of games to watch')
    parser.add_argument('--speed', type=int, default=20,
                        help='Rendering speed in frames per second')
    args = parser.parse_args()

    play(args.model, games=args.games, speed=args.speed)
