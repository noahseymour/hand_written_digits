import numpy as np
import matplotlib.pyplot as plt
        

class Network:
    def __init__(self, layout) -> None:
        self.layout = layout
        
        self.weights, self.biases = self.setup_layout(layout)
    
    def setup_layout(self, layout):
        weights = [np.random.rand(m, n) for n, m in zip(layout[:-1], layout[1:])]
        biases = [np.random.rand(k, 1) for k in layout[1:]]
        
        return weights, biases
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_prime(z):
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))
    
    @staticmethod
    def cost_derivative(a, y):
        return (a - y)
    
    def cost(self, tdata):
        total = 0
        for inp, y in tdata:
            a = self.forward_propogate(inp)
            total += sum(np.square(y - a))
        return total / len(tdata)
    
    def forward_propogate(self, inputs):
        for l in range(len(self.layout)-1):
            a = Network.sigmoid(np.dot(self.weights[l], inputs) + self.biases[l])
            inputs = a
        return a 

    def train(self, tdata, bsize, cycles, alpha, fname, sample_interval):
        errors = list()
        
        for c in range(cycles):
            if not c % sample_interval: errors.append(self.cost(tdata))
            
            np.random.shuffle(tdata)
            mbatches = [tdata[n:n+bsize] for n in range(0, len(tdata), bsize)]
            for mbatch in mbatches:
                self.learn(mbatch, alpha)

        plt.plot(errors)
        plt.xlabel("cycles")
        plt.ylabel("error")
        plt.savefig(f"{fname}.png")
    
    def learn(self, mbatch, alpha):
        dw = [np.zeros(weight.shape) for weight in self.weights]
        db = [np.zeros(bias.shape) for bias in self.biases]
        
        for inp, y in mbatch:
            cdw, cdb = self.backwards_propogate(inp, y)
            dw = [w + cw for w, cw in zip(dw, cdw)]
            db = [b + cb for b, cb in zip(db, cdb)]
            
        self.weights = [w - (alpha / len(mbatch)) * wd for w, wd in zip(self.weights, dw)]
        self.biases = [b - (alpha / len(mbatch)) * bd for b, bd in zip(self.biases, db)]

    def backwards_propogate(self, inp, y):
        dw = [np.zeros(weight.shape) for weight in self.weights]
        db = [np.zeros(bias.shape) for bias in self.biases]
        
        # forward prop
        a = inp
        actv = [a]
        ws = []
        for weight, bias in zip(self.weights, self.biases):
            w_l = np.dot(weight, a) + bias
            a = Network.sigmoid(w_l)
            ws.append(w_l)
            actv.append(a)
            
        # go back
        error_l = np.multiply(Network.cost_derivative(actv[-1], y), Network.sigmoid_prime(ws[-1])) # BP1
        dw[-1] = np.dot(error_l, actv[-2].transpose())
        db[-1] = error_l
        
        for l in range(2, len(self.layout)):
            w_l = ws[-l]
            error_l = np.dot(self.weights[-l+1].transpose(), error_l) * Network.sigmoid_prime(w_l)
            dw[-l] = np.dot(error_l, actv[-l-1].transpose())
            db[-l] = error_l
        
        return (dw, db)
