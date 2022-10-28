import numpy as np
import pandas as pd
import tqdm
from tqdm import trange
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import plotly.express as px

from fetch_data import fetch


class RomNet(nn.Module):
    def __init__(self):
        super(RomNet, self).__init__()
        self.l1 = nn.Linear(784, 128, bias=False)
        self.l2 = nn.Linear(128, 10, bias=False)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.sm(x)
        return x


def test():
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    model = RomNet()

    loss_f = nn.NLLLoss(reduction='none')
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    BS = 32
    losses, accuracies = [], []

    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=BS)
        X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[samp]).long()
        model.zero_grad()
        out = model(X)
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss_f(out, Y)
        loss = loss.mean()
        loss.backward()
        optim.step()
        loss, accuracy = loss.item(), accuracy.item()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description(f'loss: {loss:.2f} accuracy: {accuracy:.4f}')

    df = pd.DataFrame()
    df['accuracies'] = accuracies
    df['losses'] = losses
    fig = px.line(df, range_y=(0, 1.5))
    fig.show()

    # evaluation
    Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
    print(f'test accuracy: {(Y_test == Y_test_preds).mean():.4f}')


if __name__ == '__main__':
    test()