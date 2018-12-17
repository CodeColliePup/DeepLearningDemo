#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/13 下午4:00
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: sigmoid_layer.py
# @Software: PyCharm

import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

if __name__ == '__main__':
    X_dot_W = np.array([[0, 0, 0], [10, 10, 10]])

    B = np.array([1, 2, 3])

    print(X_dot_W)

    print(X_dot_W + B)

    dY = np.array([[1, 2, 3], [4, 5, 6]])

    print(dY)

    dB = np.sum(dY, axis=0)

    print(dB)