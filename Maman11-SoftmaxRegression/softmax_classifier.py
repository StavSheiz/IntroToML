import numpy as np
import pandas as pd


def test(x, w_multi):
    result_matrix = x.dot(w_multi.T)
    x["preds"] = result_matrix.apply(np.argmax, axis=1)


def train(x_train, y_train, x_test, y_test, learning_rate, max_error, max_iterations):
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    y_train_onehot = pd.get_dummies(y_train_int).astype(int)

    # init weights vector for all classes
    w = np.random.uniform(0, 0, (10, 785))
    loss_per_epoch = []
    loss_test_over_epoch = []

    for i in range(max_iterations):
        gradient = calc_gradient(x_train, y_train_onehot, w)
        error = get_errors(x_train, y_train_int, w)
        test_error = get_errors(x_test, y_test_int, w)
        loss_per_epoch.append(error)
        loss_test_over_epoch.append(test_error)

        if np.linalg.norm(gradient) < max_error:
            return w

        w = w - learning_rate * gradient

    return w, loss_per_epoch, loss_test_over_epoch


def calc_gradient(x_train, y_train, w):
    # h(x)
    wx = np.dot(x_train, w.T)
    softmax_probs = softmax(wx)

    N = x_train.shape[0]
    gradient = np.dot((y_train - softmax_probs).T, x_train) / -N

    return gradient


def softmax(s):
    # subtract by max to protect from overflow
    exp_x = np.exp(s - np.max(s, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def get_errors(x_train, y_train, w):
    all_preds = x_train.dot(w.T).apply(np.argmax, axis=1)
    errors = (all_preds != y_train)
    return np.sum(errors)
