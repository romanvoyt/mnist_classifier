import numpy as np
import pandas as pd
import torch
import plotly.express as px
from tqdm import trange

from fetch_data import fetch


def softmax(x):
    c = x.max(axis=1)
    return c + np.log(np.exp(x-c.reshape((-1, 1))).sum(axis=1))


def forward_backward(x, y, l1, l2):
    out = np.zeros((len(y), 10), np.float32)
    out[range(out.shape[0]), y] = 1

    x_l1 = x.dot(l1)
    x_relu = np.maximum(x_l1, 0)
    x_l2 = x_relu.dot(l2)
    x_softmax = x_l2 - softmax(x_l2).reshape((-1, 1))
    x_loss = (-out * x_softmax).mean(axis=1)

    d_out = -out/len(y)
    dx_softmax = d_out - np.exp(x_softmax)*d_out.sum(axis=1).reshape((-1, 1))

    d_l2 = x_relu.T.dot(dx_softmax)
    dx_relu = dx_softmax.dot(l2.T)

    dx_l1 = (x_relu > 0).astype(np.float32) * dx_relu

    d_l1 = x.T.dot(dx_l1)

    return x_loss, x_l2, d_l1, d_l2


def layer_init(m, h):
    # gaussian is strong
    # ret = np.random.randn(m,h)/np.sqrt(m*h)
    # uniform is stronger
    ret = np.random.uniform(-1., 1., size=(m,h))/np.sqrt(m*h)

    return ret.astype(np.float32)


def forward(x, l1, l2):
    x = x.dot(l1)
    x = np.maximum(x, 0)
    x = x.dot(l2)
    return x


def numpy_eval(x_test, y_test, l1, l2):
    Y_test_preds_out = forward(x_test.reshape((-1, 28*28)), l1, l2)
    Y_test_preds = np.argmax(Y_test_preds_out, axis=1)
    return (y_test == Y_test_preds).mean()

def test():
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    l1 = layer_init(784, 128)
    l2 = layer_init(128, 10)

    lr = 0.001
    BS = 64
    losses, accuracies = [], []

    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=BS)
        X = X_train[samp].reshape((-1, 28*28))
        Y = Y_train[samp]
        x_loss, x_l2, d_l1, d_l2 = forward_backward(X, Y, l1, l2)

        cat = np.argmax(x_l2, axis=1)
        accuracy = (cat == Y).mean()

        # stochastic gradient decent
        l1 = l1 - lr*d_l1
        l2 = l2 - lr*d_l2

        loss = x_loss.mean()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description((f'loss {loss:.4f} accuracy {accuracy:.4f}'))

    df = pd.DataFrame()
    df['accuracies'] = accuracies
    df['losses'] = losses
    fig = px.line(df, range_y=(-0.1, 1.1))
    fig.show()

    test_acc = numpy_eval(X_test, Y_test, l1, l2)
    print(f'test accuracy = {test_acc:.4f}')


if __name__ == '__main__':
    test()
