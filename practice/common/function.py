import torch
def softmax(x):
    if x.ndim == 2:
        m, _ = x.max(axis=1, keepdims=True)
        x = x - m
        x = torch.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - torch.max(x)
        x = torch.exp(x) / torch.sum(torch.exp(x))
    return x


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size() == y.size():
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -torch.sum(torch.log(y[torch.arange(batch_size), t] + 1e-7)) / batch_size