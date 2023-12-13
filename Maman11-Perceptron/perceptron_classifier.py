from math import copysign
from sklearn.metrics import accuracy_score
import numpy


def test(x, y, w_multi):
    preds = []
    actual = []

    for idx in range(0, x.shape[0]):
        # extract x(t), y(t)
        xt, yt = get_xt_yt(x, y, idx)


        # init confidence and predictions per class
        conf_multi = []
        class_preds = []

        # predict for each class
        for p_class in range(0,9):
            conf = numpy.inner(xt, w_multi[p_class])

            if conf < 0:
                prediction = 0
            else:
                prediction = 1

            conf_multi.append(conf)
            class_preds.append(prediction)

        # get the class with the highest confidence
        preds.append(numpy.argmax(numpy.array(conf_multi)))
        actual.append(yt)

    return preds, actual


def train(x_train, y_train, max_iterations):
    best_w = w = numpy.random.uniform(-1.0, 1.0, (10, 785))

    curr_preds, curr_actual = test(x_train, y_train, w)
    min_error = get_errors(curr_preds, curr_actual)

    for i in range(max_iterations):
        misclassified_idx = get_misclassified(curr_preds, curr_actual)

        if misclassified_idx.size > 0:
            xt, yt = get_xt_yt(x_train, y_train, misclassified_idx[0])

            update_w(xt, yt, w)
            curr_preds, curr_actual = test(x_train, y_train, w)
            errors_w = get_errors(curr_preds, curr_actual)

            # save pocket
            if errors_w < min_error:
                best_w = w
                min_error = errors_w
        else:
            return w

    return best_w


def get_errors(preds, actual):
    accuracy = accuracy_score(actual, preds)
    error_rate = 1 - accuracy

    return error_rate


def get_misclassified(preds, actual):
    preds_arr = numpy.asarray(preds)
    actual_arr = numpy.asarray(actual)
    misclassified = numpy.where(actual_arr != preds_arr)

    return misclassified[0]


def update_w(xt, yt, w):

    for i in range(10):
        w[i] = w[i] + -1 * numpy.array(xt)


def get_xt_yt(x, y, idx):
    # extract x(t)
    xrow = x.iloc[idx]
    y_idx = xrow.name
    xt = xrow.tolist()
    xt.insert(0, 1)

    # prepare binary of current class and extract y(t)
    yt = int(y.get(y_idx))

    return xt, yt
