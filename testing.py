import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


class Net:
    def __init__(self, layout) -> None:
        self.layout = layout
        self.setup_layout(layout)
    
    def setup_layout(self, layout):
        self.biases = [np.random.randn(y, 1) for y in layout[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layout[:-1], layout[1:])]
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def train(self, tdata, bsize, alpha, cycles):
        errors = list()
        
        for c in range(cycles):
            if not c % 100: errors.append(self.cost_function(tdata))
            
            np.random.shuffle(tdata)
            mbatches = [tdata[n:n+bsize] for n in range(0, len(tdata), bsize)]
            for mbatch in mbatches:
                self.learn(mbatch, alpha)

        plt.plot(errors)
        plt.xlabel("cycles")
        plt.ylabel("error")
        plt.savefig("error.png")
    
    def learn(self, mbatch, alpha):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in mbatch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # update the weights & biases
        
        self.weights = [w - (alpha / len(mbatch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (alpha / len(mbatch)*nb) for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # feedforward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
            
        # go back
        # delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        delta = np.multiply(self.cost_derivative(activations[-1], y), sigmoid_prime(zs[-1])) # BP1
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, len(self.layout)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    def cost_function(self, tdata):
        total = 0
        for x, y in tdata:
            a = self.feedforward(x)
            total += sum(np.square(y - a))
        return total / len(tdata)
    
    def evaluate(self, test_data):
        test_rs = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for x, y in test_rs)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network:
    def __init__(self, layout) -> None:
        self.layout = layout

        self.weights, self.biases = np.array([]), np.array([])
        self.setup_layout(layout)

    def setup_layout(self, layout):
        self.weights = [np.random.randn(layout[k], layout[k-1]) for k in range(1, len(layout))]
        self.biases = [np.random.randn(k, 1) for k in layout[1:]]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
     
    def feedforward(self, inputs):
        for l in range(len(self.layout)-1):
            a = self.sigmoid(np.dot(self.weights[l], inputs) + self.biases[l])
            inputs = a
        return a
    
    def forward_propogate(self, inputs):
        w, a = [], []
        for l in range(len(self.layout)-1):
            w_l = np.dot(self.weights[l], inputs) + self.biases[l]
            a_l = self.sigmoid(w_l)
            w.append(w_l)
            a.append(a_l)
            inputs = a_l
        return w, a
    
    def cost(self, output, true):
        return np.square(true - output)
    
    def train(self, data, learning_rate, cycles):
        cost = list()
        for i in range(cycles):
            # print(self.weights)
            # print(self.biases)
            # print()
            w, a = self.forward_propogate(data[0])

            cost.append(self.cost(a[-1], data[1]))

            error_L = (data[1] - a[-1]) * self.sigmoid_prime(w[-1])
            errors = [error_L]

            for l in range(len(self.weights)-2, -1, -1):
                print((self.weights[l+1].T * errors[::-1][0]).shape, self.sigmoid_prime(w[l]).shape)
                print(len(self.sigmoid_prime(w[l])))
                print()
                print(self.weights[l+1].T, errors[::-1][0], self.sigmoid_prime(w[l]))
                error_l = (self.weights[l+1].T * errors[:-1]) * self.sigmoid_prime(w[l])
                errors.append(error_l)

            # for l in range(len(self.layout)-2, -1, -1):
            #     error_l = (self.weights[l].T * errors[:-1]) * self.sigmoid_prime(w[l])
            #     print(error_l)
            #     # error_l = np.multiply((self.weights[l].T * errors[:-1]), self.sigmoid_prime(w[l]))
            #     errors.append(error_l)

            # print(errors)
            
            for l in range(len(self.weights)):
                self.weights[l] = self.weights[l] - (learning_rate * errors[::-1][l] * a[l]).T
            
            for l in range(len(self.biases)):
                self.biases[l] = self.biases[l] - (learning_rate * errors[::-1][l])
        
        return cost

xor = [
    (np.array([np.array([0]), np.array([0])]), 0),
    (np.array([np.array([1]), np.array([1])]), 0),
    (np.array([np.array([0]), np.array([1])]), 1),
    (np.array([np.array([1]), np.array([0])]), 1)
]

net = Net([2, 3, 1])
net.train(xor, 1, 0.1, 50000)

for e in xor:
    print(net.feedforward(e[0]), e[1])

# net = Net([784, 30, 10])

# (train_x, train_y), (test_x, test_y) = mnist.load_data()

# tdata = [(x, y) for x, y in zip(train_x, train_y)]
# print(len(tdata))
# net.train(tdata, 30, 0.1, 10)

