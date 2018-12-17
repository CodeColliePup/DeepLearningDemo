import numpy as np

a = np.array([0.3, 2.9, 4.0])

print(a)
exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) #溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


print("a")
a = np.array([1010, 1000, 990])
print(a)
# z1 = np.exp(a) / np.sum(np.exp(a))
# print(z1)

c = np.max(a)
print(c)
d = a - c
print(d)
z2 =np.exp(d) / np.sum(np.exp(d))
print(z2)

print("stage 2")

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))