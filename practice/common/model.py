from abc import ABC, abstractmethod
class Model(ABC):
    def __init__(self):
        self.__params = []
        self.__grads = []
        self.x = None   # input
        self.out = None # output
    
    def get_params(self):
        return self.__params
    
    def get_grads(self):
        return self.__grads
    
    def add_params(self, params):
        self.__params += params
    
    def add_grads(self, grads):
        self.__grads += grads
    
    def subtr_params(self, index, subs):
        self.__params[index] -= subs

    def set_grads(self, index, subs):
        self.__grads[index][...] = subs
    
class Layer(Model):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

class LossLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        pass

    @abstractmethod
    def forward(self, x, t):
        pass

class Net(Model):
    def __init__(self):
        super().__init__()
        self.layers = []
        self.loss_layers = []
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = 0
        for loss_layer in self.loss_layers:
            loss += loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        d = 0
        for loss_layer in self.loss_layers:
            d += loss_layer.backward(dout)
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return None
    
    def add_layers(self, layers): # List
        self.layers += layers
        for layer in layers:
            self.add_params(layer.get_params())
            self.add_grads(layer.get_grads())
    
    def add_lossLayer(self, lossLayer):
        self.loss_layers += lossLayer

class Optimizer(ABC):
    def __init__(self, lr):
        self.lr = lr
    
    @abstractmethod
    def update(self):
        pass

    def set_lr(self, lr):
        self.lr = lr