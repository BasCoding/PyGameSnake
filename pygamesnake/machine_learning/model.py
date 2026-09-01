import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


def neural_network_model(input_size, LR=1e-3):
    """
    Purpose
        Build the neural network used to play the snake game
    Input
        input_size, int: size of the input layer
        LR, float: learning rate for the optimizer
    Output
        model, tflearn.DNN: the compiled neural network model
    """
    network = input_data(shape=[None, 15, 20], name='input')

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
