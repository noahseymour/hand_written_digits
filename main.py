import numpy as np

class Network:
    def __init__(self, layout) -> None:
        self.layout = layout

        self.weights, self.biases = np.array([]), np.array([])
        self.setup_layout(layout)

    def setup_layout(self, layout):
        self.weights = [np.random.randn(layout[k], layout[k-1]) for k in range(1, len(layout))]
        self.biases = [np.random.randn(k) for k in layout[1:]]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
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

net = Network(np.array([2, 3, 1]))

data = [(1, 1), 0]

cost = net.train(data, 0.1, 1)

