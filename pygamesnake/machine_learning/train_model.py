import os
import numpy as np
from pygamesnake.machine_learning.model import neural_network_model


def train_model(
    training_data, 
    epochs,
    LR=1e-3,
    model=False
    ):
    X = np.array([i[0] for i in training_data])
    y = np.array([i[1] for i in training_data])
    # importance weights (longer snakes count more) applied via oversampling
    weights = np.array([i[2] for i in training_data])
    repeats = np.maximum(1, np.round(weights)).astype(int)
    X = np.repeat(X, repeats, axis=0)
    y = np.repeat(y, repeats, axis=0)
    print(f"{len(training_data)} collected samples oversampled to {len(X)}")

    if not model:
        model = neural_network_model(input_size=X[0].size, LR=LR)

    print(f"training for {epochs} epochs (LR={LR})")
    model.fit(X, y, n_epoch=epochs)
    return model


if __name__ == "__main__":

    LR = 1e-3
    epochs = 5
    data_path = os.path.join('data', 'training_data_manual.npy')
    print(f"loading training data from {data_path}")
    training_data = np.load(data_path, allow_pickle=True)

    model = train_model(training_data, epochs, LR)
    os.makedirs('models', exist_ok=True)
    save_path = os.path.join('models', f'trained_model_manual_{epochs}')
    model.save(save_path)
    print(f"saved model to {save_path}")
