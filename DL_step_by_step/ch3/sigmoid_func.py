import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1/(1 + np.exp(-x))


if __name__ == '__main__':
    print("")
    x = np.arange(-10.0, 10.0, 0.1)
    y = sigmoid(x)
    y2 = x * 0 + 0.5

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.ylim(-0.1, 1.1)
    plt.show()