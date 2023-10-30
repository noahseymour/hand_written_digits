import numpy as np
import matplotlib.pyplot as plt


class Network:
    def __init__(self, layout) -> None:
        self.layout = layout

        self.weights, self.biases = self.setup_network(layout)
    
    def setup_network(self, layout):
        # weights = [np.random.rand(layout[k], layout[k-1]) for k in range(1, len(layout))]
        weights = np.array([np.random.randn(y, x) for x, y in zip(layout[:-1], layout[1:])])
        biases = [np.random.randn(k, 1) for k in layout[1:]]
        
        return weights, biases
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
    
    def forward_propogate(self, inputs):
        for l in range(len(self.layout)-1):
            a = self.sigmoid(np.dot(self.weights[l], inputs) + self.biases[l])
            inputs = a
        return a

    def train(self, tdata, bsize, cycles, alpha):
        for c in range(cycles):
            np.random.shuffle(tdata)
            mbatches = [tdata[n:n+bsize] for n in range(0, len(tdata), bsize)]
            for mbatch in mbatches:
                self.learn(mbatch, alpha)
    
    def learn(self, mbatch, alpha):
        weight_deriv = [np.zeros(weight.shape) for weight in self.weights]
        bias_deriv = [np.zeros(bias.shape) for bias in self.biases]
        
        for x, y in mbatch:
            delta_nabla_b, delta_nabla_w = self.backwards_propogate(x, y)
            bias_deriv = [nb+dnb for nb, dnb in zip(bias_deriv, delta_nabla_b)]
            weight_deriv = [nw+dnw for nw, dnw in zip(weight_deriv, delta_nabla_w)]
        # for input, exp_output in mbatch:            # HERE MAYBE
        #     dw, db = self.backwards_propogate(input, exp_output)
        #     weight_deriv = np.sum(weight_deriv, dw)
        #     bias_deriv = np.sum(bias_deriv, db)
        
        self.weights = [w - (alpha / len(mbatch))*wd for w, wd in zip(self.weights, weight_deriv)]
        self.biases = [b - (alpha / len(mbatch))*bd for b, bd in zip(self.biases, bias_deriv)]
    
    def backwards_propogate(self, input, output):
        weight_deriv = [np.zeros(weight.shape) for weight in self.weights]
        bias_deriv = [np.zeros(bias.shape) for bias in self.biases]
                
        a = input
        actv = list(input)
        ws = list()
        for weight, bias in zip(self.weights, self.biases):
            w = np.dot(weight, a) + bias
            a = self.sigmoid(w)
            ws.append(w)
            actv.append(a)
        
        # propogate error backwards
        error_l = np.multiply(self.cost_derivative(actv[-1], output), self.sigmoid_prime(ws[-1]))
        bias_deriv[-1] = error_l
        weight_deriv[-1] = np.dot(error_l, actv[-2].transpose())
        
        for l in range(2, len(self.layout)):
            error_l = np.dot(self.weights[-l+1].transpose(), error_l) * self.sigmoid_prime(ws[-1])
            weight_deriv[-l] = np.dot(error_l, actv[-l-1].transpose())
            bias_deriv[-l] = error_l
        
        return weight_deriv, bias_deriv
        
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
        plt.savefig("network_performance.png")
    
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