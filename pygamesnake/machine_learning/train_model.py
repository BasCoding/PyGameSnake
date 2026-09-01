import os
import tensorflow as tf
tf.compat.v1.reset_default_graph()
import numpy as np
from pygamesnake.machine_learning.model import neural_network_model


def train_model(
    training_data, 
    epochs,
    LR=1e-3,
    model=False
    ):
    X = np.array([i[0] for i in training_data])#.reshape(-1,len(training_data[0][0]),1)#.reshape(1,len(training_data),len(training_data[0]),len(training_data[0][0]))
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]), LR=LR)#[0])*len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=epochs, snapshot_step=250, show_metric=True, run_id='openai_learning')
    return model


if __name__ == "__main__":

    LR = 1e-3
    epochs = 5
    training_data = np.load('training_data_weighted_head.npy', allow_pickle=True)

    model = train_model(training_data, epochs, LR)
    model.save(os.path.join('model', f'trained_model_head_{epochs}'))
