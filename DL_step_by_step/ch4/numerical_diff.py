import numpy as np
import sys
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.dataset.mnist import load_mnist


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
f2 = tangent_line(function_1, 5)
y2 = f2(x)


# plt.xlabel("x")
# plt.ylabel("f(x)")
plt.plot(x, y)
plt.plot(x, y2)
plt.show()

print(numerical_diff(function_1, 5))

print(numerical_diff(function_1, 10))