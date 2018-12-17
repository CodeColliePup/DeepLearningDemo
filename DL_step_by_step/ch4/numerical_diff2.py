#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/5 上午11:38
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: numerical_diff2.py
# @Software: PyCharm

import numpy as np
import sys
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.dataset.mnist import load_mnist
import matplotlib.pylab as plt
from DL_step_by_step.ch4.numerical_diff import numerical_diff



def function_2(x):
    return x[0] ** 2 + x[1] ** 2

def function_3(X,Y):
    return X ** 2 + Y ** 2

x0 = np.arange(-20.0, 20.0, 0.25)
x1 = np.arange(-20.0, 20.0, 0.25)

X,Y = np.meshgrid(x0,x1)

R= np.sqrt(X**2, Y**2)

Z = function_3(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

plt.show()


def function_tmp1(x0):
    return x0*x0 + 4.0 ** 2.0


print(numerical_diff(function_tmp1, 3.0))

