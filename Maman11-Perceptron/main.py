# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from perceptron_classifier import train, test
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

    # Fetch MNIST dataset
    mnist = fetch_openml('mnist_784')

    # Extract features (x) and labels (y)
    x, y = mnist.data, mnist.target

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=10000)

    w = train(x_train, y_train, 5)
    preds, actual = test(x_test, y_test, w)
    test_cmatrix = confusion_matrix.from_predictions(actual, preds)
    test_display = ConfusionMatrixDisplay(test_cmatrix)
    test_display.plot()
    plt.show()

    print("done")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
