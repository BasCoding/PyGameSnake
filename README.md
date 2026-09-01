# PyGameSnake

A classic Snake game built in Python with [pygame](https://www.pygame.org/), plus an optional machine learning setup (PyTorch) that can learn to play the game for you.

## Requirements

- Python ~3.13
- [Poetry](https://python-poetry.org/) for dependency management

## Installation

```bash
poetry install
```

## Game

To start the game simply run:

```bash
python pygamesnake/game/snake.py
```

Move the snake with:

| Key | Direction |
| --- | --------- |
| `W` | Up        |
| `A` | Left      |
| `S` | Down      |
| `D` | Right     |

Eat the food to grow. The game ends if the snake hits a wall or itself.

## Machine Learning

You can train a machine learning model to play the game for you! The workflow has three steps: collect training data, train a model, then watch it play.

### 1. Collect training data

```bash
python pygamesnake/machine_learning/create_training_data.py
```

You'll be prompted to choose a mode:

- **Algorithm** (`a`, default): a simple built-in algorithm plays many rounds automatically and records the successful ones. Data is saved to `data/training_data_algorithm.npy`.
- **Manual** (`m`): you play with `W`/`A`/`S`/`D` and every move is recorded. Data is saved to `data/training_data_manual.npy`.

New samples are appended to any existing data file.

### 2. Train a model

```bash
python pygamesnake/machine_learning/train_model.py
```

This loads the collected data, trains a neural network, and saves the result under `models/` (e.g. `trained_model_manual_5`).

### 3. Test a model

```bash
python pygamesnake/machine_learning/test_model.py <model_version>
```

For example:

```bash
python pygamesnake/machine_learning/test_model.py trained_model_algorithm_5
```

Add `--normal-speed` to watch the game at normal playable speed instead of running as fast as possible.

## Reinforcement Learning

Instead of learning from human or algorithmic examples, the snake can also figure out a strategy entirely on its own through **reinforcement learning**. This uses Deep Q-Learning (DQN): the agent tries moves, receives rewards (+10 for eating food, −10 for dying), and gradually learns which moves pay off.

No training data is needed — just run the training and watch it improve.

### How it works (in short)

- **State** (what the agent sees): 11 simple values — is there danger straight/right/left, which way it's heading, and where the food is relative to the head.
- **Actions**: 3 relative moves — go straight, turn right, or turn left.
- **Reward**: +10 for food, −10 for dying, 0 otherwise.
- **Learning**: a small neural network is updated after every move and replays past games from memory.

The code is split into three focused pieces:

- [pygamesnake/reinforcement_learning/environment.py](pygamesnake/reinforcement_learning/environment.py) — wraps the game and defines the state and rewards.
- [pygamesnake/reinforcement_learning/agent.py](pygamesnake/reinforcement_learning/agent.py) — the neural network, the Q-learning update, and the explore/exploit logic.
- [pygamesnake/reinforcement_learning/train.py](pygamesnake/reinforcement_learning/train.py) — the training loop that ties it together.

### Train the agent

```bash
python pygamesnake/reinforcement_learning/train.py
```

You'll see a game window and per-game scores in the console. The snake starts by moving randomly and steadily gets better; it typically begins scoring within the first ~30–40 games and reaches a record in the 20–30+ range after ~150 games. The best model is saved to `models/rl_model`.

Useful options:

- `--games 300` — train for more games (higher scores).
- `--speed 80` — render faster (higher frames per second).
- `--no-render` — train without the window for maximum speed, then watch the result afterwards.

### Watch the trained agent

```bash
python pygamesnake/reinforcement_learning/play.py
```

This loads `models/rl_model` and plays greedily (no random exploration). Use `--speed` to slow it down for a demo, or `--games` to set how many rounds to watch.

## Project Structure

```
data/                             saved training data (.npy)
models/                           trained models
pygamesnake/
  game/snake.py                   the playable Snake game
  machine_learning/
    create_training_data.py       collect training data (algorithm or manual)
    train_model.py                train a model on collected data
    test_model.py                 watch a trained model play
    model.py                      neural network definition
    grid.py                       grid representation used as model input
    directkeys.py                 keyboard input helpers
  reinforcement_learning/
    environment.py                reinforcement-learning game wrapper (state + rewards)
    agent.py                      DQN network, Q-learning update, explore/exploit logic
    train.py                      train the agent with reinforcement learning
    play.py                       watch a trained reinforcement-learning agent play
```

