import matplotlib.pyplot as plt
import numpy.random as npr
from numpy import *

def LoadDataSet():
    dataMat = []
    labelMat = []
    fr = open('C:/Users/92469/Desktop/data.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def H(theta, x):
    flag = mat(x) * mat(theta)
    if flag > 0:
        return 1;
    else:
        return 0;


def hw(x):
    x = array(x);
    cnt = x.size;
    for i in range(cnt):
        if x[i] >= 0:
            x[i] = 1;
        else:
            x[i] = 0;
    return mat(x);


def SGD(dataMat, lableMat):
    dataMat = mat(dataMat);
    m, n = shape(dataMat);
    alpha = 0.01;
    times = 10000;
    theta = zeros((n, 1));
    for i in range(times):
        cnt = 0
        while True:
            cnt += 1;
            id = npr.randint(0, m);
            temp = array(dataMat)[id];
            h = H(theta, temp);
            if h != lableMat[id]:
                break
            if cnt > 500:
                break
        theta = theta + array(mat(alpha*(lableMat[id]-h)*mat(temp)).transpose());
       # plot(theta)
        cost = Cost(theta, dataMat, lableMat);
       # print(theta)

        print(cost);
    return theta;


def Cost(theta, data, label):
    data = mat(data);
    label = mat(label);
    theta = mat(theta);
    temp = hw(data * theta);
    temp = mat(temp) - label.transpose();
    res = mat(theta).transpose() * data.transpose();
    res = res * temp;
    return res;

def plot(theta):
    dataMat, lableMat = LoadDataSet();
    dataArr = array(dataMat);
    n = shape(dataArr)[0];
    xcord1 = []; ycord1 = [];
    xcord2 = []; ycord2 = [];
    for i in range(n):
        if(int(lableMat[i])) == 0:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure();
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, marker='+')
    ax.scatter(xcord2, ycord2, s=10)

    x = arange(-1, 100, 0.1);
    y = (-theta[0]-theta[1]*x)/theta[2];

    ax.plot(x, y);
    plt.xlabel('Exam1');
    plt.ylabel('Exam2');
    plt.show();

def Solve():
    dataMat, labelMat = LoadDataSet();
    theta = SGD(dataMat, labelMat);
    print(theta);
    plot(theta);

if __name__ == '__main__':
    Solve();
