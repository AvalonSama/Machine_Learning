import numpy as np
import jieba
from numpy import *


def LoadData(s):
    text = []
    with open(s, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\ufeff')
            text.append(line)
    res = jieba.cut(str(text[0]))
    return list(res)


def LoadStop(s):
    stoplis = []
    with open(s, "r", encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip('\n')
            stoplis.append(word)
    return stoplis


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))


def output(x, W, U, b, d):
    h = tanh(d+np.dot(H, x).transpose())
    y = b+np.dot(W, x).transpose()+np.dot(U, h)
    return y, h


def softmax(y, one_hot):
    temp_up = np.exp(np.dot(y.transpose(), one_hot))
    temp_y = y.tolist()
    temp_down = sum(np.exp(temp_y))
    return temp_up/temp_down


def Vectorlize(text, embedding_size):
    word_bag = list(set(text))
    print(word_bag)
    word_bag_size = len(word_bag)
    one_hot = np.eye(word_bag_size)
    word_vect = np.random.random((embedding_size, word_bag_size))
    return word_bag, word_bag_size, one_hot, word_vect


def ParameterInit(word_bag_size, window_size, hidden_size, embedding_size):
    W = np.matrix(np.random.random(
        (word_bag_size, (window_size-1)*embedding_size))*0.01)
    U = np.matrix(np.random.random((word_bag_size, hidden_size))*0.01)
    H = np.matrix(np.random.random(
        (hidden_size, (window_size-1)*embedding_size))*0.01)
    b = np.matrix(np.random.random((word_bag_size, 1))*0.01)
    d = np.matrix(np.random.random((hidden_size, 1))*0.01)
    return W, U, H, b, d


def Train(W, U, H, b, d, word_vect, one_hot, word_bag, word_bag_size, window_size):
    alpha = 0.001
    T = 100
    while(T > 0):
        T -= 1
        for i in range(0, word_bag_size-window_size):
            x = word_vect[:, i]
            temp_one_hot = np.matrix(one_hot[:, i+window_size])
            for j in range(1, window_size-1):
                x = np.append(x, word_vect[:, j])
            y, h = output(x, W, U, b, d)
            dy = temp_one_hot.transpose()-y
            db = dy
            dU = np.dot(dy, h.transpose())
            dW = np.dot(dy, mat(x))

            dh = np.dot(dy.transpose(), U)
            do = np.multiply(dh.transpose(), (1-np.multiply(h, h)))
            temp_do = repeat(do, 200, 1)
            dH = np.multiply(temp_do, x)

            temp_h = repeat((1-np.multiply(h, h)), 200, 1)

            dhx = np.multiply(temp_h, H)

            dyx = W + np.dot(U, dhx)

            dx = np.dot(dy.transpose(), dyx)

            b += alpha * db
            U += alpha * dU
            W += alpha * dW
            d += alpha * do
            H += alpha * dH
            dx = dx.tolist()
            x += np.array(dx[0])
            for k in range(1, window_size):
                temp = x[(k-1)*50:k*50]
                word_vect[:, i+(k-1)] = temp
    return word_vect


if __name__ == '__main__':
    window_size = 5
    embedding_size = 50
    hidden_size = 20
    text = LoadData('data2NNLM.txt')
    stoplis = LoadStop('stoplis.txt')
    word_bag, word_bag_size, one_hot, word_vect = Vectorlize(
        text, embedding_size)
    W, U, H, b, d = ParameterInit(
        word_bag_size, window_size, hidden_size, embedding_size)
    result = Train(W, U, H, b, d, word_vect, one_hot,
                   word_bag, word_bag_size, window_size)
    mindis = 1000000000000000
    ans1 = 0
    ans2 = 0
    for i in range(0, word_bag_size):
        for j in range(i+1, word_bag_size):
            if i != j and mindis > np.dot(result[:, i]-result[:, j], result[:, i]-result[:, j]):
                ans1 = i
                ans2 = j
                mindis = np.dot(result[:, i]-result[:, j],
                                result[:, i]-result[:, j])
    print(word_bag[ans1])
    print(word_bag[ans2])

    # print(result)
