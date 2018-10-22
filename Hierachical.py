from numpy import *;
import numpy as np;

def LoadDataSet():
    word_bag = [];
    text=[];
    fr = open('C:/Users/92469/Desktop/word_bag.txt');
    for line in fr.readlines():
        lineArr = line.strip().split();
        word_bag = lineArr;
    fr = open('C:/Users/92469/Desktop/data.txt')
    for line in fr.readlines():
        lineArr = line.strip().split();
        text.append(lineArr);
    return word_bag, text;

def Vectorlize(word_bag, texts):
    texts_v=[];
    for text in texts:
        vector=[];
        vector.append(text[0]);
        for word in word_bag:
            num = text.count(word);
            if num>0:
                num=1;
            vector.append(num);
        texts_v.append(vector);
    return texts_v;

def tDis(t1,t2,texts_v):
    v1 = texts_v[t1][1:];
    v2 = texts_v[t2][1:];
    up = np.dot(v1,v2);
    down = linalg.norm(v1) * linalg.norm(v2);
    return up / down;

def Dis(i, j, Cl,texts_v):
    sum = 0;
    for t1 in Cl[i]:
        for t2 in Cl[j]:
            sum += tDis(t1,t2,texts_v);
    sum/=(len(Cl[i])*len(Cl[j]));
    return sum;

def Hierachical(texts_v,k):
    Cl = [];
    for i in range(0,len(texts_v)):
        tempc = [];
        tempc.append(i);
        Cl.append(tempc);
    cnt = 1;
    Cnum = len(Cl);
    while(Cnum>k):
        mmax=-1;
        m = -1;
        n = -1;
        d = -1.0;
        for i in range(0,len(Cl)):
            for j in range(0,len(Cl)):
                if i<j and len(Cl[i])!=0 and len(Cl[j])!=0:
                    d = Dis(i,j,Cl,texts_v);
                    if d>mmax:
                        mmax = d;
                        n = i;
                        m = j;
        print("--"*20);
        print("第 %d 轮"%(cnt));
        cnt +=1;
        print("第 %d 号簇与第 %d 号簇合并，相似度为 %f"%((n+1),(m+1),mmax));
        for i in Cl[m]:
            Cl[n].append(i);
        Cl[m].clear();
        Cnum -= 1;
    return Cl;

if __name__=="__main__":
   word_bag,texts = LoadDataSet();
   texts_v = Vectorlize(word_bag, texts);
   Cl = Hierachical(texts_v,3);
   print("最终结果")
   print(Cl);
