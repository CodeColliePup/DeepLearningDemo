#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/7 下午5:37
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: twoLayerNet.py
# @Software: PyCharm

import sys
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.common.common_func import *
import matplotlib.pylab as plt


class TwoLayerNet:
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 weight_init_std = 0.01
                 ):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        """

        :param x: 输入数据
        :param t: 监督数据
        :return: loss损失值
        """
        y = self.predict(x)
        return cross_entropy_error(y, t)


    def accuracy(self, x, t):
        """

        :param x:
        :param t:
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuray = np.sum(y == t) / float(x.shape[0])

        return accuray

    def numberical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def print_params(self):
        print("W1 : ")
        print(self.params['W1'].shape)
        print(self.params['W1'])
        print("b1 : ")
        print(self.params['b1'].shape)
        print(self.params['b1'])
        print("W2 : ")
        print(self.params['W2'].shape)
        print(self.params['W2'])
        print("b2 : ")
        print(self.params['b2'].shape)
        print(self.params['b2'])




if __name__ == '__main__':

    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    net.print_params()

    x = np.random.rand(100, 784)
    t = np.random.rand(100, 10)

    grads = net.numberical_gradient(x, t)
    print(grads['W1'])