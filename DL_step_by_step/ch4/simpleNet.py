#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/7 下午5:15
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: simpleNet.py
# @Software: PyCharm

import sys
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.common.common_func import *
import matplotlib.pylab as plt



import numpy as np

class simpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entroy_error(y, t)

        return loss



if __name__ == '__main__':
    net = simpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])
    print(x.shape)
    p =net.predict(x)
    print(p)

    print(np.argmax(p))

    t = np.array([0, 0, 1])

    print(net.loss(x,t))

    f = lambda w: net.loss(x, t)

    dW = numerical_gradient(f, net.W)

    print(dW)