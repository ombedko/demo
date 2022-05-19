
import keras as keras
from keras import losses, activations, optimizers,metrics

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron

def plot(x, y, test_data):

    i0 = np.where(y == 0)
    i1 = np.where(y == 1)

    plt.scatter(x[i0, 0], x[i0, 1], c='r')
    plt.scatter(x[i1, 0], x[i1, 1], c='b')
    plt.scatter(test_data[:, 0], test_data[:, 1], c='g')
    plt.show()


def iris():
    # load data and slice
    iris = load_iris()
    x = iris.data[:, (2, 3)]  # petal length and petal width
    y = (iris.target == 0).astype(float)

    test_data = np.array([[1.8, 0.5], [1.5, 0.4], [5, 1.5], [6, 2]])

    # plot(x, y, test_data)

    # Perceptron
    model = Perceptron()
    model.fit(x, y)

    y_pred = model.predict(test_data)
    print(y_pred)


    # Sequential API
    model2 = keras.models.Sequential()
    model2.add(keras.layers.Dense(3, activation=keras.activations.tanh))
    model2.add(keras.layers.Dense(1, activation=keras.activations.tanh))
    model2.compile(loss=keras.losses.mean_squared_error,
                   optimizer='adam',
                   metrics=['accuracy'])

    model2.fit(x, 1.8*(y-0.5), verbose=0, epochs=1000)

    y_pred2 = model2.predict(test_data)
    print(y_pred2)




if __name__ == '__main__':
    iris()