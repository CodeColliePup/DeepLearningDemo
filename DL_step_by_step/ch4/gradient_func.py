#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/5 下午6:01
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: gradient_func.py
# @Software: PyCharm

import numpy as np
import sys
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.common.common_func import numerical_gradient
import matplotlib.pylab as plt
from DL_step_by_step.common.common_func import function_2



# def numerical_gradient_on_batch(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)
#
#     for index in range(x.size):
#         tmp_val = x[index]
#         x[index] = tmp_val + h
#         fxh1 = f(x)
#
#         x[index] = tmp_val - h
#         fxh2 = f(x)
#
#         grad[index] = (fxh1 - fxh2) / (2 * h)
#         x[index] = tmp_val
#
#     return grad
#
#
# print(numerical_gradient_on_batch(function_2, np.array([3.0, 4.0])))


# gradient test
# x = np.array([3.0, 4.0])
# f= function_2
# h = 1e-4
# grad = np.zeros_like(x)
#
# for index in range(x.size):
#
#     tmp_val = x[index]
#
#     x[index] = tmp_val + h
#     fxh1 = f(x)
#
#     x[index] = tmp_val - h
#     fxh2 = f(x)
#
#     grad[index] = (fxh1 - fxh2) / (2 * h)
#     x[index] = tmp_val


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)




    X, Y = np.meshgrid(x0, x1)

    print(x0.shape)
    print(x1.shape)


    X = X.flatten()
    Y = Y.flatten()
    # plt.scatter(X,Y)
    # plt.show



    z = np.array([X, Y])

    print(z.shape)

    grad = numerical_gradient(function_2, z)

    print(grad)
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")  # ,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()