from numpy import *
import queue;

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
    C1_center = texts_v[1][1:];
    C2_center = texts_v[4][1:];
    C3_center = texts_v[6][1:];

    flag = False;
    cnt = 1;
    while flag == False:
        print("----"*20);
        print("第 %d 轮"%(cnt));
        cnt+=1;
        C1.clear();
        C2.clear();
        C3.clear();
        for text in texts_v:
            num = int(text[0]);
            d1 = EU(C1_center,text);
            d2 = EU(C2_center,text);
            d3 = EU(C3_center,text);

            print("text %d 距离三个簇中心的距离分别为 %f %f %f"%(num,d1,d2,d3));

            if d1<=d2 and d1<=d3:
                C1.append(text);
                print("加入C1");
            elif d2<=d1 and d2<=d3:
                C2.append(text);
                print("加入C2");
            elif d3<=d1 and d3<=d2:
                C3.append(text);
                print("加入C3");
        
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


def Dis(a,b):
    temp = mat(a[1:])-mat(b[1:]);
    temp = mat(temp) * mat(temp).transpose();
    temp = sqrt(float(temp));
    return float(temp);

def Count(n, texts_v, R):
    res = [];
    for i in range(0,len(texts_v)):
        if R > Dis(texts_v[i],texts_v[n]):
            res.append(i);
    return res;

def DBSCAN(texts_v, R, n):
    Clist = [];
    Cnoise = [];
    tag = [];
    for i in range(0,len(texts_v)):
        tag.append(False);
    for i in range(0,len(texts_v)):
        print("--"*20);
        print("样本 %d"%(i+1));
        if tag[i] == False:
            temp = [];
            temp = Count(i,texts_v,R);
            print("领域中样本个数为 %d"%(len(temp)));
            if len(temp)<n:
                Cnoise.append(i);
                print("噪声点");
            else:
                print("创建新簇,簇中内容为");
                C=[];
                for j in temp:
                    k = Count(j,texts_v,R);
                    if len(k) > n:
                        for t in k:
                            if temp.count(t) == 0:
                                temp.append(t);
                    if tag[j] == False:
                        print("%d"%(j+1));
                        C.append(j);
                        tag[j] = True;
                    if tag[j] and Cnoise.count(j)!=0:
                        Cnoise.remove(j);
                        C.append(j);
                Clist.append(C);

        tag[i] = True;
    Clist.append(Cnoise);
    return Clist;
                
if __name__=='__main__':
    word_bag,texts = LoadDataSet();
    texts_v = Vectorlize(word_bag, texts);
    type = "2";
    if type == "1":
        C1,C2,C3 = Kmeans(texts_v);
        print(C1);
        print(C2);
        print(C3);
    else:
        Clist = DBSCAN(texts_v,2.1,3);
        print("最终结果（文本编码从0开始）")
        print(Clist);
    