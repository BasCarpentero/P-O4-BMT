import math


def get_y(x, W):
    y = W*x
    return y


def get_feature_vector(y):
    f = []
    for i in range(len(y)):
        s2 = sum(y[i] ^ 2)
        f.append(math.log(s2, 10))
    return f
