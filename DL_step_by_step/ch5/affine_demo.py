#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/17 下午6:12
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: affine_demo.py
# @Software: PyCharm

import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
