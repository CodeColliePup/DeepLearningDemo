#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/21 上午11:34
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : DeepLearningDemo
# @FileName: TwoLayerNet.py
# @Software: PyCharm

import sys, os
sys.path.append(os.pardir)
import numpy as np
from DL_step_by_step.common import *
from collections import OrderedDict

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine()
