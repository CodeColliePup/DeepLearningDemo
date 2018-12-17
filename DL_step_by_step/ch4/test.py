#
#!/usr/bin/python3.6.2
# -*- coding: utf-8 -*-
# @Time    : 2018/12/11 下午5:53
# @Author  : Wenson
# @Desc : ==============================================
# Life is Short I Use Python!!!                      ===
# If this runs wrong,don't ask me,I don't know why;  ===
# If this runs right,thank god,and I don't know why. ===
# Maybe the answer,my friend,is blowing in the wind. ===
# ======================================================
# @Project : Keras_start
# @FileName: test.py
# @Software: PyCharm

import numpy as np
z = np.arange(10)


t = [2,7,0,9,4]
y = np.arange(50)
y = y.reshape(5, 10)

a = y[np.arange(5), t]