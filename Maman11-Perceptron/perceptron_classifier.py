import numpy as np
import pandas as pd


def test(x, w_multi):
    result_matrix = x.dot(w_multi.T)
    x["preds"] = result_matrix.apply(np.argmax, axis=1)


def train(x_train, y_train, max_iterations):
    y_train_int = y_train.astype(int)
    y_train_onehot = pd.get_dummies(y_train_int).astype(int)

    # init weights vector for all classes
    best_w_multi = np.random.uniform(-1, 1, (10, 785))

    # for each class, add label '1' if the example belongs to the class, else '-1'
    y_train_onehot_labels = y_train_onehot.apply(lambda col: col * 2 - 1)

    w = best_w_multi
    min_error = x_train.shape[0]
    error_count, idx_error = get_errors(x_train, y_train_int, w)

    errors_for_epoc = list()

    for epoch in range(max_iterations):

        if error_count > 0:
            # update weights for all classes according to their result on the misclassified example
            for i in range(10):
                xt, yt = get_xt_yt(x_train, y_train_onehot_labels.iloc[:, i], idx_error)

                # predict for single class
                p = np.inner(xt, w[i])
                pred = np.sign(p).astype(int)
                if yt != pred:
                    w[i] = w[i] + yt * np.array(xt)

            error_count, idx_error = get_errors(x_train, y_train_int, w)
            errors_for_epoc.append(error_count)

            # save pocket
            if error_count < min_error:
                best_w_multi = w
                min_error = error_count
        else:
            best_w_multi = w
            break

    return best_w_multi, errors_for_epoc


def get_errors(x_train, y_train, w):
    all_preds = x_train.dot(w.T).apply(np.argmax, axis=1)
    errors = (all_preds != y_train)
    error_rows = errors.idxmax()
    total_errors = np.sum(errors)

    if total_errors != 0:
        return total_errors, error_rows
    else:
        return total_errors, -1


def get_xt_yt(x, y, idx):
    # extract x(t)
    xrow = x.iloc[idx]
    y_idx = xrow.name
    xt = xrow.tolist()

    # extract y(t)
    yt = y.get(y_idx)

    return xt, yt
