from common.model import Optimizer
class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__(lr)
    
    def update(self, model):
        params = model.get_params()
        grads = model.get_grads()
        for i in range(len(params)):
            grad = grads[i]
            params[i] -= self.lr * grads[i]
            model.subtr_params(i, self.lr * grad)


# class SGD:
#     '''
#     확률적 경사하강법(Stochastic Gradient Descent)
#     '''
#     def __init__(self, lr=0.01):
#         self.lr = lr
        
#     def update(self, params, grads):
#         for i in range(len(params)):
#             params[i] -= self.lr * grads[i]