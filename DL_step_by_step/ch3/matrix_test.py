import numpy as np
import matplotlib.pyplot as plt

# A = np.array([1, 2, 3, 4])
# print(A)
# print(np.ndim(A))
# print(A.shape)
# print(A.shape[0])
#
# B = np.array([[1, 2], [3, 4], [5, 6]])
# print(B)
# print(np.ndim(B))
# print(B.shape)

A = np.array([[1, 2],[3, 4]])
print(A)
print(A.shape)

B = np.array([[5, 6],[7, 8]])
print(B)
print(B.shape)
z = np.dot(A, B)
print(z)