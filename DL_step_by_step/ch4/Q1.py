#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/7 上午11:15
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: Q1.py
# @Software: PyCharm

import sys
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.common.common_func import numerical_gradient
import matplotlib.pylab as plt

import numpy as np



def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x-= lr*grad

    return x, np.array(x_history)





def function_2(x):
    return x[0] ** 2 + x[1] ** 2

init_x = np.array([-3.0, 4.0])

# print(init_x.ndim)
# grad,history = gradient_descent(function_2, init_x = init_x, lr = 0.1,  step_num=100)
#
# print(grad)
# print(history)

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 100
x, x_history = gradient_descent(function_2, init_x,lr = lr, step_num=step_num)

print(x)


plt.plot( [-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5,5], '--b')
plt.plot(x_history[:, 0], x_history[:, 1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()