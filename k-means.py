from numpy import *

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
        vector.append(int(text[0]));
        for word in word_bag:
            num = text.count(word);
            if num>0:
                num=1;
            vector.append(num);
        texts_v.append(vector);
    return texts_v;

def Kmeans(texts_v):
    m , n = shape(texts_v);
    C1 = [];
    C2 = [];
    C3 = [];
    C1_center = texts_v[1][1:];
    C2_center = texts_v[4][1:];
    C3_center = texts_v[6][1:];

    flag = False;
    while flag == False:
        print("-"*20);
        C1.clear();
        C2.clear();
        C3.clear();
        for text in texts_v:
            d1 = EU(C1_center,text);
            d2 = EU(C2_center,text);
            d3 = EU(C3_center,text);
            if d1<=d2 and d1<=d3:
                C1.append(text);
            elif d2<=d1 and d2<=d3:
                C2.append(text);
            elif d3<=d1 and d3<=d2:
                C3.append(text);
        C1_center_before = C1_center;
        C2_center_before = C2_center;
        C3_center_before = C3_center;
        C1_center = GetCenter(C1);
        C2_center = GetCenter(C2);
        C3_center = GetCenter(C3);
        if C1_center_before==C1_center and C2_center_before==C2_center and C3_center==C3_center_before:
            flag = True;

    temp1 = [];
    temp2 = [];
    temp3 = [];

    for t in C1:
        temp1.append(t[0]);
    for t in C2:
        temp2.append(t[0]);
    for t in C3:
        temp3.append(t[0]);
    return temp1,temp2,temp3;


if __name__=='__main__':
    word_bag,texts = LoadDataSet();
    Kmeans(Vectorlize(word_bag,texts));
    