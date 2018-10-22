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
    y = b+np.dot(W, x)+np.dot(U, tanh(d+np.dot(H, x)))
    return y


def Vectorlize(text,embedding_size):
    word_bag = list(set(text))
    word_bag_size = len(word_bag)
    one_hot = np.eye(word_bag_size)
    word_vect = np.random.random((embedding_size, word_bag_size))
    return word_bag, one_hot, word_vect

def ParameterInit(word_bag_size, window_size, hidden_size, embedding_size):
    W = np.random.random((word_bag_size, (window_size-1)*embedding_size))


if __name__ == '__main__':
    window_size = 5
    embedding_size = 50
    text = LoadData('data2NNLM.txt')
    stoplis = LoadStop('stoplis.txt')
    word_bag, word_bag_size, one_hot, word_vect = Vectorlize(text)
    W,U,b,d = ParameterInit(text)
    #x,W,U,b,d = Init(text)
    #result = train(x,W,U,b,d)
    # print(result)
