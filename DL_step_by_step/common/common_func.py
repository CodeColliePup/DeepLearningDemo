#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/5 下午6:11
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: common_func.py
# @Software: PyCharm

import numpy as np

def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

# def _numerical_gradient_no_batch(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)
#     # print("x.size")
#     # print(x.shape)
#     for index in range(x.size):
#         # print(index)
#         tmp_val = x[index]
#         x[index] = float(tmp_val) + h
#         fxh1 = f(x)
#
#         x[index] = float(tmp_val) - h
#         fxh2 = f(x)
#
#         grad[index] = (fxh1 - fxh2) / (2 * h)
#         x[index] = tmp_val
#
#     return grad


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def gradient_descent(f, init_x, lr = 0.01, step_num =100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# def cross_entropy_error(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)
#
#     batch_size = y.shape[0]
#
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        test_c = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    test_a = y[np.arange(batch_size), t]
    test_b = np.log(y[np.arange(batch_size), t] + 1e-7)
    z = -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    return z