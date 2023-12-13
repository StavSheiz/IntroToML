# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from perceptron_classifier import train, test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np


def run_perceptron():
    print(f'Starting perceptron')

    # Fetch MNIST dataset
    mnist = fetch_openml('mnist_784')

    # Extract features (x) and labels (y)
    x, y = mnist.data, mnist.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000)
    y_train_onehot = pd.get_dummies(y_train.astype(int)).astype(int)

    # Add bias
    x_train.insert(0, 'Bias', 1)
    x_test.insert(0, 'Bias', 1)

    w = train(x_train, y_train_onehot, 50)
    test(x_test, w)
    actual = y_test.astype(int)
    preds = x_test["preds"]

    # generate confusion matrix
    test_cmatrix = confusion_matrix(actual, preds)
    test_display = ConfusionMatrixDisplay(test_cmatrix)
    test_display.plot()
    plt.savefig('confusion_matrix.png')

    plt.show()

    print(f'accuracy: {accuracy_score(actual, preds)}')

    sensitivity_per_class = []

    for i in range(10):
        true_positives = test_cmatrix[i, i]
        false_negatives = sum(test_cmatrix[i, :]) - true_positives
        sensitivity = true_positives / (true_positives + false_negatives)
        sensitivity_per_class.append(sensitivity)

    print("Sensitivity for each class:")
    for i, sensitivity in enumerate(sensitivity_per_class):
        print(f"Class {i}: {sensitivity:.2f}")


if __name__ == '__main__':
    run_perceptron()

