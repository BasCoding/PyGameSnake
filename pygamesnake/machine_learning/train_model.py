import os
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.compat.v1.reset_default_graph()
import numpy as np


def neural_network_model(input_size):

    network = input_data(shape=[None, 15,20], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 5, activation='softmax')      #size of y!!!
    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model


def train_model(
    training_data, 
    epochs,
    model=False
    ):
    X = np.array([i[0] for i in training_data])#.reshape(-1,len(training_data[0][0]),1)#.reshape(1,len(training_data),len(training_data[0]),len(training_data[0][0]))
    y = [i[1] for i in training_data]

    if not model:
        model = neural_network_model(input_size=len(X[0]))#[0])*len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=epochs, snapshot_step=250, show_metric=True, run_id='openai_learning')
    return model


if __name__ == "__main__":

    LR = 1e-3
    epochs = 5
    training_data = np.load('training_data_weighted_head.npy', allow_pickle=True)

    model = train_model(training_data)
    model.save(os.path.join('model', f'trained_model_head_{epochs}'))
