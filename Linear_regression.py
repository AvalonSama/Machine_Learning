import matplotlib.pyplot as mt
from numpy import *
import numpy.linalg as nla

def LoadDataSet():
    datamat = [];
    attrmat = [];
    labelmat = [];
    fr = open('C:/Users/92469/Desktop/data.txt');
    for line in fr.readlines():
        lineArr = line.strip().split();
        datamat.append([float(lineArr[0]),float(lineArr[1])]);
        attrmat.append([1.0,float(lineArr[0])]);
        labelmat.append([float(lineArr[1])]);
    return datamat,attrmat,labelmat;


def CloseForm():
    datamat,attrmat,labelmat = LoadDataSet();
    tempMat = mat(attrmat).transpose();
    tempMat = tempMat*attrmat;
    tempMat = nla.inv(tempMat);
    theta = tempMat * (mat(attrmat).transpose()) * mat(labelmat);
    return theta;


def plot(theta):
    datamat,attrmat,labelmat= LoadDataSet();
    dataarr = array(datamat);
    n=shape(dataarr)[0];
    xcord = [];ycord=[];
    for i in range(n):
        xcord.append(dataarr[i,0]);
        ycord.append(dataarr[i,1]);
    fig = mt.figure();
    ax = fig.add_subplot(111);
    ax.scatter(xcord, ycord, c='red', marker='s');
    x = arange(1999, 2014, 1);
    theta = array(theta);
    y = theta[0] + theta[1] * x;
    ax.plot(x, y);
    mt.show()

def Solve():
    theta = CloseForm();
    print(theta);
    plot(theta);

if __name__=='__main__':
    Solve();