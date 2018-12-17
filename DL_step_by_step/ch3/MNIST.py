import sys, os
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.dataset.mnist import load_mnist
from DL_step_by_step.ch3.sigmoid_func import sigmoid
from DL_step_by_step.ch3.softmax_func import softmax

import numpy as np
from PIL import Image
import pickle




def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()



# img = x_train[0]
# label = t_train[0]
# print(label)
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
# img_show(img)


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    print(x_train[0])
    print(x_train.shape)
    print(t_train[0])
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
print(type(network))

batch_size = 100

for key in network:
    print(key)
    print(network[key].shape)

accuracy_cnt = 0
# for i in range(len(x)):
#     y = predict(network, x[i])
#     # print(y)
#     p = np.argmax(y)
#     # print(p)
#     if p == t[i]:
#         accuracy_cnt += 1


for i in range(0, len(x), batch_size):
    x_batch = x [i: i + batch_size]
    # print(x_batch.shape)
    y_batch = predict(network, x_batch)
    # print(y_batch.shape)
    p = np.argmax(y_batch, axis=1)
    # print(p.shape)
    # input()
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy : " +str(float(accuracy_cnt) /len(x)))