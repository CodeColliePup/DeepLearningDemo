#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/20 下午4:29
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : DeepLearningDemo
# @FileName: softmaxWithLoss.py
# @Software: PyCharm

import numpy as np
from DL_step_by_step.common.common_func import *


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backword(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx