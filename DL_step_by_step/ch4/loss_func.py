import numpy as np
import sys
sys.path.extend(['/Users/songwenxuan/PycharmProjects/Keras_start'])
from DL_step_by_step.dataset.mnist import load_mnist

def mean_squared_error(y, t):
    """
    func: 计算均方误差


    因为带了平方，后面要用梯度下降法，要求导，这样求导多出的乘2就和二分之一抵消了，一个简化后面计算的技巧

    :param y:
    :param t:
    :return:
    """
    return 0.5 * np.sum((y-t)**2)


# def cross_entropy_error(y, t):
#     """
#     func: 计算交叉熵误差
#
#     :param y:
#     :param t:
#     :return:
#     """
#     delta = 1e-7
#     return -np.sum(t*np.log(y+delta))


def cross_entroy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size





if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    print(x_train.shape)
    print(t_train.shape)

    train_size = x_train.shape[0]

    batch_size = 10

    batch_mask = np.random.choice(train_size, batch_size)

    x_batch = x_train[batch_mask]

    t_batch = t_train[batch_mask]

