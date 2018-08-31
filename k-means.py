from numpy import *

def LoadDataSet():
    word_bag = [];
    text=[];
    fr = open('C:/Users/92469/Desktop/word_bag.txt');
    for line in fr.readlines():
        lineArr = line.strip().split();
        word_bag = lineArr;
    fr = open('C:/Users/92469/Desktop/text.txt')
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


def EU(a, b):
    temp = mat(a)-mat(b[1:]);
    temp = mat(temp) * mat(temp).transpose();
    temp = sqrt(float(temp));
    return float(temp);


def GetCenter(c):
    sum = zeros((1,23));
    for text in c:
        sum = sum + mat(text[1:]);
    sum/=len(c);
    return sum.tolist()[0];


def Kmeans(texts_v):
    m , n = shape(texts_v);
    C1 = [];
    C2 = [];
    C3 = [];
    C1_center = texts_v[0][1:];
    C2_center = texts_v[4][1:];
    C3_center = texts_v[7][1:];

    flag = False;
    while flag == False:
        C1.clear();
        C2.clear();
        C3.clear();
        for text in texts_v:
            d1 = EU(C1_center,text);
            d2 = EU(C2_center,text);
            d3 = EU(C3_center,text);
            if d1<d2 and d1<d3:
                C1.append(text);
            elif d2<d1 and d2<d3:
                C2.append(text);
            elif d3<d1 and d3<d2:
                C3.append(text);
        C1_center_before = C1_center;
        C2_center_before = C2_center;
        C3_center_before = C3_center;
        C1_center = GetCenter(C1);
        C2_center = GetCenter(C2);
        C3_center = GetCenter(C3);
        if C1_center_before==C1_center and C2_center_before==C2_center and C3_center==C3_center_before:
            flag = True;
    return C1,C2,C3;

def D(a,b):
    return 0;

def Single_pass(texts_v,T):
    CList = [];
    C=[];
    C.append(texts_v[0]);
    CList.append(C);

    for text in texts_v[1:]:
        maxd = -1;
        Cid = -1;
        for i in range(0,len(CList)):
            d = D(text,CList[i]);
            if d>maxd:
                maxd = d;
                Cid = i;
        if maxd>T:
            CList[Cid].append(text);
        else:
            tempC=[];
            tempC.append(text);
            CList.append(tempC);
    return CList;


def Combine(a, b):
    C = [];
    for temp in a:
        C.append(temp);
    for temp in b:
        C.append(temp);
    return C;

def Hierarchical(texts_v, k):
    Clist = [];
    for text in texts_v:
        C = [];
        C.append(text);
        Clist.append(C);

    while len(Clist)>k:
        d = [[]];
        for i in range(0,len(Clist)):
            for j in range(0,len(Clist)):
                d[i][j] = DC(Clist[i],Clist[j]);
        maxd = -1;
        id1=-1;
        id2=-1;
        for i in range(0,len(Clist)):
            for j in range(0,len(Clist)):
                if maxd<d[i][j]:
                    maxd = d[i][j];
                    id1 = i;
                    id2 = j;
        C = Combine(Clist[i],Clist[j]);
        Clist.remove(Clist[i]);
        Clist.remove(Clist[j]);
        Clist.append(C);
    return Clist;

def Dis(a,b):
    temp = mat(a[1:])-mat(b[1:]);
    temp = mat(temp) * mat(temp).transpose();
    temp = sqrt(float(temp));
    return float(temp);

def DBSCAN(texts_v, R, n):
    Clist = [];
    Cnoise = [];
    tag = [];
    for i in range(0,len(texts_v)):
        tag[i] = False;
    for i in range(0,len(texts_v)):
        if tag[i] == False:
            temp = [];
            for j in range(0,len(texts_v)):
                if tag[j]==False:
                    d = Dis(texts_v[i],texts_v[j]);
                    if d<R:
                        temp.append(j);
            if(len(temp)<n):
                Cnoise.append(i);
            else:
                C = [];
                C.append(i);
                for j in temp:
                    count = [];
                    for k in len(texts_v):
                        if Dis(texts_v[j],texts_v[k])<R and k!=i:
                            count.append(k);
                    if len(count)<n:
                        Cnoise.append(j);
                    else:
                        for k in count:
                            C.append(k);
                            tag[k]=True;
        tag[i] = True;
        Clist.append(Cnoise);
    return Clist;
                    






if __name__=='__main__':
    word_bag,texts = LoadDataSet();
    texts_v = Vectorlize(word_bag, texts); 
    Clist = DBSCAN(texts_v,2.1,3);
    print(Clist);
    