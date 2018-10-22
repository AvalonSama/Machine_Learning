import numpy as np
import jieba
from numpy import *

def LoadData(s):
    text = []
    with open(s,"r",encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line.strip('\ufeff')
            text.append(line)
    res = jieba.cut(str(text[0]))
    #res = ",".join(res)
    return list(res)


def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

def output(x,W,U,b,d):
    y = b+np.dot(W,x)+np.dot(U,tanh(d+np.dot(H,x)))
    





if __name__=='__main__':
    text = LoadData('data2NNLM.txt')
    print(text);
    #x,W,U,b,d = Init(text)
    #result = train(x,W,U,b,d)
    #print(result)