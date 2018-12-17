import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])

A = np.array([[1, 2], [3, 4]])
print(A)

A.shape


B = np.array([10, 20])

A * B

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)

X[0]
X[0][1]

for row in X:
    print(row)

X = X.flatten()
X[np.array([0, 2, 4])]

X>15

X[X>15]