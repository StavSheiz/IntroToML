import numpy as np
import pandas as pd


def test(x, w_multi):
    result_matrix = x.dot(w_multi.T)
    x["preds"] = result_matrix.apply(np.argmax, axis=1)


def train(x_train, y_train, x_test, y_test, max_iterations):
    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)
    y_train_onehot = pd.get_dummies(y_train_int).astype(int)

    # init weights vector for all classes
    best_w_multi = np.random.uniform(0, 0, (10, 785))

    # for each class, add label '1' if the example belongs to the class, else '-1'
    y_train_onehot_labels = y_train_onehot.apply(lambda col: col * 2 - 1)

    w = best_w_multi
    min_error = x_train.shape[0]
    error_count, idx_error = get_errors(x_train, y_train_int, w)

    errors_for_epoc = list()
    test_errors_for_epoc = list()

    for epoch in range(max_iterations):

        if error_count > 0:
            # update weights for all classes according to their result on the misclassified example
            xt = x_train.loc[idx_error]
            yt = y_train_onehot_labels.loc[idx_error]

            ytshape = yt.values.reshape((10,1))
            xtshape = xt.values.reshape((1, 785))

            w = w + np.dot(ytshape, xtshape)

            error_count, idx_error = get_errors(x_train, y_train_int, w)

            # save pocket
            if error_count < min_error:
                best_w_multi = w
                min_error = error_count

            test_errors, test_error_idx = get_errors(x_test, y_test_int, best_w_multi)
            test_errors_for_epoc.append(test_errors / x_test.shape[0])
            errors_for_epoc.append(min_error / x_train.shape[0])

        else:
            best_w_multi = w
            break

    return best_w_multi, errors_for_epoc, test_errors_for_epoc


def get_errors(x_train, y_train, w):
    all_preds = x_train.dot(w.T).apply(np.argmax, axis=1)
    errors = (all_preds != y_train)
    error_rows = errors.idxmax()
    total_errors = np.sum(errors)

    if total_errors != 0:
        return total_errors, error_rows
    else:
        return total_errors, -1
