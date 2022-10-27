import gzip
import os, hashlib

import numpy as np
import requests


def fetch(url):
    file_path = os.path.join('/tmp', hashlib.md5(url.encode(encoding='utf-8')))
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            dat = f.read()
    else:
        with open(file_path, 'wb') as f:
            dat = requests.get(url).content
            f.write(dat)
    return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()


if __name__ == '__main__':
    X_train = fetch('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')[0x10:].reshape((-1, 28,28))