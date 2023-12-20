import numpy as np


def train(x_train, y_train):
    # Convert DataFrame and Series to NumPy arrays
    X = x_train.to_numpy()
    y = y_train.to_numpy()

    w = np.dot(np.dot(np.linalg.pinv(np.dot(X.T, X)), X.T), y)
    return w


def test(x, w):
    result_matrix = x.dot(w)
    x["preds"] = np.clip(np.round(result_matrix).astype(int), 0, 9)
