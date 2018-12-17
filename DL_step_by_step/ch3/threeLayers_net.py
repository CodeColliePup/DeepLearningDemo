import numpy as np
import matplotlib.pyplot as plt

from DL_step_by_step.ch3.sigmoid_func import sigmoid

X = np.array([1.0, 0.5])

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2 ,0.3])

print("X1 : ")
print(X)
print(X.shape)

print("W1 : ")
print(W1)
print(W1.shape)

print("B1 : ")
print(B1)
print(B1.shape)

A1 = np.dot(X, W1) + B1
print("A1 : ")
print(A1)
print(A1)

Z1 = sigmoid(A1)
print("Z1 : ")
print(Z1)
print(Z1.shape)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print("W2 : ")
print(W2)
print(W2.shape)

print("B2 : ")
print(B2)
print(B2.shape)

A2 = np.dot(Z1, W2) + B2
print("A2 : ")
print(A2)
print(A2.shape)

Z2 = sigmoid(A2)
print("Z2 : ")
print(Z2)
print(Z2.shape)


def indentity_function(x):
    return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])

B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = indentity_function(A3)
