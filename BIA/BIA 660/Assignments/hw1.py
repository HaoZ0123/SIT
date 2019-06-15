# BIA-660-C Haodong Zhao
# Q1
import csv
import string
from collections import Counter

def tokenize(x):
    # remove newline characters('\n')
    # 去除换行符，由于连词符被换行符打断，进行重连
    x = x.replace('\n', ' ')
    x = x.replace('- ', '-')
    x = x.replace(' -', '-')

    # split string into a list by space
    # 按照空格分割字符串成列表
    tokens = x.split()
    
    # remove punctuation
    # 去除列表中每个字符串首尾符号，由于有可能不止一个，进行遍历
    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if tokens[i][j] in string.punctuation:
                tokens[i] = tokens[i][1:]
            else:
                break

    for i in range(len(tokens)):
        for j in range(len(tokens[i])):
            if tokens[i][len(tokens[i]) - 1] in string.punctuation:
                tokens[i] = tokens[i][:len(tokens[i]) - 1]
            else:
                break

    # remove empty tokens
    # 去掉空项
    tokens = [i for i in tokens if i != '']

    # return all tokens as a list output with lower case
    # 函数返回一个转化成小写的列表
    return [i.lower() for i in tokens]


# Q2
# Bubble sort
# 冒泡排序，后面使用
def bubble_sort(list):
    n = len(list)
    for j in range(n - 1):
        count = 0
        for i in range(n - 1 - j):
            if list[i][0] > list[i + 1][0]:
                list[i], list[i + 1] = list[i + 1], list[i]
                count += 1
        if count == 0:
            break
    return list


class Text_Analyzer(object):
    def __init__(self, text):
        self.text = text

    # sort with word freq
    # 输出词频按照text顺序
    def analyze(self, text):
        print("\nThis is Q2 analyze function:")
        s = tokenize(text)
        result = {}
        for i in s:
            if s.count(i) > 1:
                result[i] = s.count(i)
            else:
                result[i] = 1
        print(result)

    # 输出词频按大小顺序
    '''def analyze(self, text):
        s = tokenize(text)
        result = Counter(s)
        print(result)'''

    def topN(self, N):
        print("\nThis is Q2 topN function:")
        s = tokenize(text)
        
        # build word freq table
        # 词频表
        result = {}
        for i in s:
            if s.count(i) > 1:
                result[i] = s.count(i)
            else:
                result[i] = 1
    
        # change dict to list
        # 将字典转换成列表
        opt = []
        for key, value in result.items():
            opt.append([value, key])
        
        # use bubble sort to sort list with the word freq
        # 调用冒泡排序，将列表按照词频从小到大排序
        opt = bubble_sort(opt)
        
        # output topN
        # 调换键与值的位置，并切片输出topN
        ls = []
        for i in reversed(range(len(opt))):
            ls.append((opt[i][1],opt[i][0]))
        print(ls[:N])

# Q3
def bigram(doc, N):
    print("\nThis is Q3:")
    s = tokenize(doc)
    
    # build a list and get any 2 consecutive tokens
    # 建立列表将每两个相邻的词作为一个元素
    ls = []
    for i in range(len(s)-1):
        ls.append([s[i],s[i+1]])
    # 合并词组，去除嵌套列表
    for i in range(len(ls)):
        ls[i] = ls[i][0] + ' ' + ls[i][1]
    # 统计词频
    result = Counter(ls)

    # output topN
    # 将词频表转换成列表，并排序，切片输出topN
    lis = []
    for key, value in result.items():
        lis.append([value,key])

    lis = bubble_sort(lis)
    l = []
    for i in reversed(range(len(lis))):
        l.append((lis[i][1],lis[i][0]))
    print(l[:N])


if __name__ == "__main__":
    # Test Question 1
    text = ''' There was nothing so VERY remarkable in that; nor did Alice
think it so VERY much out of the way to hear the Rabbit say to
itself, `Oh dear!  Oh dear!  I shall be late!'  (when she thought
it over afterwards, it occurred to her that she ought to have
wondered at this, but at the time it all seemed quite natural);
but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT-
POCKET, and looked at it, and then hurried on, Alice started to
her feet, for it flashed across her mind that she had never
before seen a rabbit with either a waistcoat-pocket, or a watch to
take out of it, and burning with curiosity, she ran across the
field after it, and fortunately was just in time to see it pop
down a large rabbit-hole under the hedge.
'''

    print(tokenize(text))

    # Test Question 2
    analyzer = Text_Analyzer(text)
    analyzer.analyze(text)
    analyzer.topN(5)

    # 3 Test Question 3

    top_bigrams = bigram(text, 5)
