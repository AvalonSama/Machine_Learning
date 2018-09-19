from numpy import *;

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


def Single_pass(texts_v,T):
    Cl = [];
    c = [];
    c.append(texts_v[0]);
    Cl.append(c);
    for i in range(1,len(texts_v)):
        print("--"*40);
        print("第%d个文本"%(i+1));
        flag = -1;
        mmax = -1000000;
        for j in range(0,len(Cl)):
            center = GetCenter(Cl[j]);
            d = -EU(center,texts_v[i]);
            print("与簇 %d 的距离"%(j+1),d);
            if d>T:
                if(mmax<d):
                    mmax = d;
                    flag = j;
        if flag == -1:
            tempc = [];
            tempc.append(texts_v[i]);
            Cl.append(tempc);
            print("构建新簇");
        else:
            Cl[flag].append(texts_v[i]);
            print("加入到簇 %d "%(flag+1));
    return Cl;

if __name__=="__main__":
    word_bag,text = LoadDataSet();
    texts_v = Vectorlize(word_bag, text);
    Cl = Single_pass(texts_v,-2.35);
    print(Cl);