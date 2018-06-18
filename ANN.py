from numpy import *


def LoadDataSet():
    dataMat = [];
    labelMat = [];
    fr = open('C:/Users/92469/Desktop/data2.txt')
    for line in fr.readlines():
        lineArr = line.strip().split();
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])]);
        labelMat.append(int(lineArr[2]));
    return dataMat, labelMat;


def Sigmoid(x):
    temp = exp(-x);
    cnt=0;
    for i in temp:
        i = i+1;
        i = 1.0/i;
        temp[cnt]=i;
        cnt+=1;
    return temp;


def BP(data, label, q):
    n, m = shape(data);
    l = 1;
    v = random.random(size=(m, q));
    w = random.random(size=(q, l));
    theta = random.random();
    gamma = random.random(size=(q, 1));
    times = 500;
    rate = 0.001;
    for t in range(times):
        for i in range(n):
            alpha = (mat(data[i]) * mat(v)).transpose();
            temp = array(alpha - gamma);
            b = Sigmoid(temp);
            beta = mat(w).transpose() * b;
            y = Sigmoid(beta - theta);
            g = y * (1-y) * (label[i]-y);
            e = [];
            for j in range(q):
                e.append(double(b[j] * (1-b[j]) * w[j] * g));
            e = array(e);

            w = w-rate * double(g[0]) * b;
            theta = theta + rate * g;
            v = v - rate * e * mat(data[i]).transpose();
            gamma = gamma + rate * e;
    return v, w, theta, gamma;


def Solve():
    data, label = LoadDataSet();
    v, w, theta, gamma = BP(data, label, 5);
    print("v:\n")
    print(v);
    print("w:\n");
    print(w);
    print("theta:\n");
    print(theta);
    print("gamma:\n");
    print(gamma);

if __name__ == '__main__':
    Solve();