import numpy


def test(x, w_multi):
    result_matrix = x.dot(w_multi.T)
    x["preds"] = result_matrix.apply(numpy.argmax, axis=1)


def train(x_train, y_train_onehot, max_iterations):
    # init weights vector for all classes
    best_w_multi = numpy.random.uniform(0, 0, (10, 785))

    # for each class, add label '1' if the example belongs to the class, else '-1'
    y_train_onehot_labels = y_train_onehot.apply(lambda col: col * 2 - 1)

    # train each class separately
    for i in range(10):
        best_w = w = best_w_multi[i]
        min_error = get_errors(x_train, y_train_onehot_labels.iloc[:, i], w)

        for epoch in range(max_iterations):
            is_misclassified, w = run_pla(x_train, y_train_onehot_labels.iloc[:, i], w)

            if is_misclassified:
                errors_w = get_errors(x_train, y_train_onehot_labels.iloc[:, i], w)

                # save pocket
                if errors_w < min_error:
                    best_w = w
                    min_error = errors_w
            else:
                best_w_multi[i] = w
                break

        best_w_multi[i] = best_w

    return best_w_multi


def get_errors(x_train, y_train_onehot_labels, w):
    all_preds = numpy.sign(numpy.dot(x_train.values, w))
    return numpy.sum(all_preds != y_train_onehot_labels)


def run_pla(x, y, w):
    is_misclassified, idx = get_misclassified_idx(x, y, w)

    if is_misclassified:
        xt, yt = get_xt_yt(x, y, idx)

        # predict for single class
        pred = numpy.sign(numpy.inner(xt, w))
        if yt != pred:
            w = w + yt * numpy.array(xt)

    return is_misclassified, w


def get_misclassified_idx(x_train, y_train_onehot_labels, w):
    all_preds = numpy.sign(numpy.dot(x_train.values, w))
    row_indices = numpy.where(all_preds != y_train_onehot_labels)

    if len(row_indices) != 0:
        return True, row_indices[0][0]
    else:
        return False, -1


def get_xt_yt(x, y, idx):
    # extract x(t)
    xrow = x.iloc[idx]
    y_idx = xrow.name
    xt = xrow.tolist()

    # extract y(t)
    yt = int(y.get(y_idx))

    return xt, yt
