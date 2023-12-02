from math import copysign


def fit(example_set):
    w = [None] * 10
    for idx in range(0, 9):
        label_class_set(example_set)
        w[idx] = get_weight_pocket([0, 0], example_set)

    return w


def test_data(test_set, w):
    confusion_matrix = init_confusion_matrix()

    for item in test_set:
        prediction = predict(item, w)
        update_confusion_matrix(item, prediction, confusion_matrix)

    calc_confusion_matrix_results(confusion_matrix)
    return confusion_matrix


def predict(item, w):
    confidence_scores = list(map(lambda conf: conf*item.label, w))
    max_confidence = max(confidence_scores)

    return confidence_scores.index(max_confidence)


def update_confusion_matrix(item, prediction, confusion_matrix):
    for idx in range (0, 9):
        if item.label == idx and idx == prediction:
            confusion_matrix[idx][item.label]["TP"] = confusion_matrix[idx][item.label]["TP"] + 1
        if item.label != idx and idx != prediction:
            confusion_matrix[idx][item.label]["TN"] = confusion_matrix[idx][item.label]["TN"] + 1
        if item.label != idx and idx == prediction:
            confusion_matrix[idx][item.label]["FP"] = confusion_matrix[idx][item.label]["FP"] + 1
        if item.label == idx and idx != prediction:
            confusion_matrix[idx][item.label]["FN"] = confusion_matrix[idx][item.label]["FN"] + 1


def calc_confusion_matrix_results(confusion_matrix):
    for row_idx in range(0, 9):
        for col_idx in range(0, 9):
            cell = confusion_matrix[row_idx][col_idx]
            cell["ACC"] = (cell["TP"]+cell["TN"])/(cell["TP"]+cell["TN"]+cell["FP"]+cell["FN"])
            cell["TPR"] = cell["TP"]/(cell["TP"]+cell["FN"])
            cell["TNR"] = cell["TN"]/(cell["TN"]+cell["FP"])


def init_confusion_matrix():
    confusion_matrix = [[None] * 10] * 10 # using index 10 as total of all classes

    for row_idx in range(0, 9):
        for col_idx in range(0, 9):
            confusion_matrix[row_idx][col_idx] = {
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0
            }

    return confusion_matrix


def label_class_set(class_idx, example_set):
    for example in example_set:
        example.class_label = 1 if example[class_idx] == 1 else -1


# all of this may need refactor because w, example.label should be vectors
def get_weight_pocket(w_init, example_set):
    w = w_init
    pocket = {'w': w, 'value': 0}

    for example in example_set: # might need to change this to some decided limit for iterations?
        new_w = get_single_weight_pla(w, example)
        new_w_eval = evaluate_w(w, example_set)
        if pocket['value'] < new_w_eval:
            pocket['value'] = new_w_eval
            pocket['w'] = new_w

    return pocket['w']


# example.label - original label vector, for example for label 0 -> [1,0,0,0,0,0,0,0,0,0]
# example.class_label - label added for the classifier, for example for class 0 -> 1 if example label is 0, -1 if not.
def get_single_weight_pla(w, example):
    new_w = w
    if example.class_label == copysign(1, w):
        new_w = w + example.class_label * example.label

    return new_w


def evaluate_w(w, example_set):
    count = 0

    for example in example_set:
        if example.class_label == copysign(1, w * example.label):
            count = count + 1

    return count
