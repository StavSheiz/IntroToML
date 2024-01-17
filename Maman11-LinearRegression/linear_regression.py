import numpy as np
import pandas as pd


def train(x_train, y_train):
    y_train_int = y_train.astype(int)
    y_train_onehot = pd.get_dummies(y_train_int).astype(int)

    # Convert DataFrame and Series to NumPy arrays
    X = x_train.to_numpy()
    y = y_train_onehot.to_numpy()

    w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return w


def test(x, w):
    result_matrix = x.dot(w)
    x["preds"] = np.clip(np.round(result_matrix.apply(np.argmax, axis=1)).astype(int), 0, 9)
