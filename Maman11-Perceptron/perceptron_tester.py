

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
