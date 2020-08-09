# coding: utf-8
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.optimizer import SGD
from common.util import preprocess, create_context_target, convert_one_hot
from common.trainer import Trainer
import torch
from data import ptb
import sys
sys.path.append('..')
import numpy as np
from common.layer import SoftmaxWithLoss
from common.model import Layer, Net

class MatMul(Layer):
    def __init__(self, input_size, output_size, shared=None):
        super().__init__()
        if shared is None:
            W = 0.01 * torch.randn(input_size, output_size).float()
        else:
            W = shared
        dW = torch.zeros_like(W)
        self.add_params([W])
        self.add_grads([dW])
        self.x = None

    def forward(self, x):
        W, = self.get_params()
        out = torch.mm(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.get_params()
        dx = torch.mm(dout, W.T)
        dW = torch.mm(self.x.T, dout)
        self.set_grads(0, dW)
        return dx

class SimpleSkipGram(Net):
    def __init__(self, input_size, hidden_size, output_size, num):
        super().__init__()
        I, H, O = input_size, hidden_size, output_size

        self.in_layer = MatMul(I, H)
        self.out_layer = MatMul(H, O)
        for _ in range(num):
            self.add_lossLayer([SoftmaxWithLoss()])

        self.num = num

        self.add_layers([
            self.in_layer,
            self.out_layer
        ])
        self.word_vecs = self.in_layer.get_params()[0]

    def forward(self, contexts, target):
        score = self.predict(target)
        loss = 0
        for i, loss_layer in enumerate(self.loss_layers):
            loss += loss_layer.forward(score, contexts[:, i])
        return loss


    def backward(self, dout=1):
        d = 0
        for loss_layer in self.loss_layers:
            d += loss_layer.backward(dout)
        for layer in reversed(self.layers):
            d = layer.backward(d)
        return None

        # dl1 = self.loss_layers[0].backward(dout)
        # dl2 = self.loss_layers[1].backward(dout)
        # ds = dl1 + dl2
        # dh = self.out_layer.backward(ds)
        # self.in_layer.backward(dh)
        return None

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 5000

# with open('text8', 'r') as f:
#     text = f.read()
text = "you say goodbye and i say hello"
corpus, word_to_id, id_to_word = preprocess(text, subset=1.0)
print("complete")
vocab_size = len(word_to_id)
contexts, target = create_context_target(corpus, window_size)
target = convert_one_hot(target, vocab_size).float()
contexts = convert_one_hot(contexts, vocab_size).float()
parallel_num = contexts.shape[1]
model = SimpleSkipGram(vocab_size, hidden_size, vocab_size, parallel_num)
optimizer = SGD()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])