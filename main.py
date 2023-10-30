import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from network import Network

# Utility functions 

def plot_mnist(train_X):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap("Blues"))
    plt.savefig("mnist.png")

def to_output_vector(y):
    vec = np.zeros((10, 1))
    vec[y-1] = 1.0
    return vec


# Training data for networks

XOR = [
    (np.array([np.array([0]), np.array([0])]), 0),
    (np.array([np.array([1]), np.array([1])]), 0),
    (np.array([np.array([0]), np.array([1])]), 1),
    (np.array([np.array([1]), np.array([0])]), 1)
]

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()


# Clean up data
tdata = [(np.reshape(inp, (784, 1)), to_output_vector(y)) for inp, y in zip(train_X, train_Y)]


# train xor net
xor_net = Network([2, 3, 1])
xor_net.train(XOR, 1, 50000, 0.1, "xor_error", 100)

digit_net = Network([784, 30, 10])
digit_net.train(tdata, 10, 30, 3.0, "mnist_error", 1)
