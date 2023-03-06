import numpy as np 


def softmax(x):
    d = []
    s = []
    for xj in x:
        d.append(np.exp(xj))
    for xi in x:
        s.append(np.exp(xi)/sum(d))
    return s

def sigmoid(x):

    sig = (1/(1 + np.exp(-x)))

    return sig