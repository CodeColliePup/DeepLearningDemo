#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/13 上午10:36
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: simpleLayer3.py
# @Software: PyCharm

import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


if __name__ == '__main__':
    x = np.array([[1.0, -0.5], [-2.0, 3.0]])
    print(x)

    mask = (x<=0)
    print(mask)

    relu_layer =  Relu()
    signal_out = relu_layer.forward(x)
    print(signal_out)





