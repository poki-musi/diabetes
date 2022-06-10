import numpy as np
from numpy import linalg as alg
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd

def main():
    data = pd.read_csv("diabetes.csv").to_numpy()
    X = data[:,:4]
    Y = np.equal(data[:,-1:], np.array([[0, 1]]))

    X = minmaxnorm(X)

    # cv10()

    model = make_model()
    model.fit(X, Y, verbose=0, batch_size=2, epochs=300)
    model.evaluate(X, Y)

def minmaxnorm(data):
    m,M = data.min(axis=0), data.max(axis=0)
    return (data - m) / (M - m)

def make_model():
    model = Sequential([
        Dense(8, activation = 'linear'),
        Dense(2, activation = 'softmax'),
    ])
    adam = tf.keras.optimizers.Adam(learning_rate=0.005)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model

def cv10():
    res = 0
    K = 10
    for i in range(K):
        print(f"i={i}")
        idx = np.arange(Y.shape[0]) % K != i
        Xtrn, Xtst = X[idx,:], X[~idx,:]
        Ytrn, Ytst = Y[idx,:], Y[~idx,:]

        model = make_model()
        model.fit(Xtrn, Ytrn, verbose=0, batch_size=2, epochs=300)
        err = model.evaluate(Xtst, Ytst, verbose=0)
        print(err)
        res += err

    return res / 10

if __name__ == "__main__":
    main()
