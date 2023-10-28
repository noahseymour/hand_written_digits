import numpy as np

class Network:
    def __init__(self, layout) -> None:
        self.layout = layout 
        
        self.weights, self.biases = np.array([]), np.array([])
        self.setup_layout()
    
    def setup_layout(self):
        self.weights = [np.random.randn(self.layout[k], self.layout[k-1]) for k in range(1, len(self.layout))]
        self.biases = [np.random.randn(k) for k in self.layout[1:]]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def forward_propogate(self, inputs):
        w, a = [], [np.array(self.sigmoid(inputs))]
        for l in range(len(self.weights)):
            w_l = np.dot(self.weights[l], inputs) + self.biases[l]
            a_l = self.sigmoid(w_l)
            w.append(w_l)
            a.append(a_l)
            inputs = a_l
        return w, a 
    
    def train(self, data, learning_rate, cycles):
        for i in range(cycles):
            w, a = self.forward_propogate(data[0])

            error_L = np.multiply((a[-1] * data[1]), self.sigmoid_prime(w[-1]))
            s = error_L.shape
            errors = [error_L]

            for l in range(len(self.weights)-2, -1, -1):
                b = (self.weights[l+1].T * errors[::-1][0]).shape
                c = self.sigmoid_prime(w[l]).shape
                error_l = (self.weights[l+1].T * errors[::-1][0]) * self.sigmoid_prime(w[l])
                errors.append(error_l)

            for i in errors:
                print(i.shape)
            for i in self.weights:
                print(i.shape)
            
            for l in range(len(self.weights)):
                print(self.weights[l].shape, errors[::-1][l].shape, a[l+1].shape, a[l])
                self.weights[l] -= learning_rate * errors[::-1][l] * a[l+1]
            
            for l in range(len(self.biases)):
                self.biases[l] -= learning_rate * errors[::-1][l]

# 2-1 = 2-n * n-1
# m-n * n-p = m-p
        
# 2-2 * 2-1
# 2-1, 1-2

net = Network([1, 2, 1])

data = np.array([[1], [1]])
net.train(data, 0.1, 1)

print(net.weights)
