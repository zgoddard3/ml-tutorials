import numpy as np
import torch as th
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import SGD


class LineModel(nn.Module):
    def __init__(self, seed=None):
        super().__init__()
        self.rng = th.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        self.slope = nn.Parameter(th.rand(1,generator=self.rng,dtype=th.float32)*2-1)
        self.bias = nn.Parameter(th.rand(1,generator=self.rng,dtype=th.float32)*2-1)
    
    def forward(self, x : th.Tensor, noise=False):
        y = self.slope*x + self.bias
        if noise:
            y += 0.3*th.randn(x.shape,generator=self.rng)
        return y

class Trainer:
    def __init__(self, model : nn.Module, lr=1e-2):
        self.model = model
        self.opt = SGD(model.parameters(), lr=lr)
    
    def train(self, x, y):
        self.opt.zero_grad()
        ypred = self.model(x)
        error = ypred - y
        loss = th.sum(error**2)
        loss.backward()
        self.opt.step()
        
    def fit(self, x, y):
        xmat = th.vstack((x,th.ones(x.shape))).T
        beta = xmat.T.matmul(xmat).inverse().matmul(xmat.T.matmul(y))
        with th.no_grad():
            self.model.slope.copy_(beta[0])
            self.model.bias.copy_(beta[1])


def plot(x,y,model,target):
    plt.figure()
    plt.scatter(x,y,c='red')

    with th.no_grad():
        y = model(x)
    plt.plot(x,y)

    with th.no_grad():
        y = target(x)
    plt.plot(x,y, '--')
    
    plt.show()


if __name__ == "__main__":
    target = LineModel(3)

    line = LineModel(2)

    x = th.linspace(-1, 1, 30)
    target_y = target(x, noise=True).detach()

    trainer = Trainer(line)
    plot(x,target_y,line,target)

    for _ in range(10):
        trainer.train(x,target_y)
        plot(x,target_y,line,target)

    line = LineModel(2)
    trainer = Trainer(line)
    trainer.fit(x,target_y)
    plot(x, target_y, line, target)