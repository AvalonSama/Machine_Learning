

def LoadDic(s):
    f = open(s);
    dic = [];
    for line in f.readlines():
        dic.append(line);
    return;


def LoadText(s):









if __name__=="__main__":
    print(dic = LoadDic("‪C:\Users\Kai Chen\Desktop\文本信息处理\作业1布置材料\作业1布置材料\中文分词词典（作业一用)\中文分词词典（作业一用).TXT");)
    stoplis = LoadDic("‪C:\Users\Kai Chen\Desktop\文本信息处理\作业1布置材料\作业1布置材料\中文停用词表（作业一用）\stoplis.txt");
    text = LoadText("‪C:\Users\Kai Chen\Desktop\文本信息处理\作业1布置材料\作业1布置材料\test.txt");
    result1 = ForwardSegment(text,dic,text);
    result2 = BackwardSegment(text,dic,text);
