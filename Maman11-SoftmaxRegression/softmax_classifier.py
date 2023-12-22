import numpy as np


def test(x, w_multi):
    result_matrix = x.dot(w_multi.T)
    x["preds"] = result_matrix.apply(np.argmax, axis=1)


def train(x_train, y_train, learning_rate, max_error, max_iterations):
    # init weights vector for all classes
    w = np.random.uniform(0, 0, (10, 785))

    for i in range(max_iterations):
        gradient = calc_gradient(x_train, y_train, w)

        if np.linalg.norm(gradient) < max_error:
            return w

        w = w - learning_rate * gradient

    return w


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
