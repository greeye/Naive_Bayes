import numpy as np
import random
import re

# 英文单词切割器
def textParse(bigString):
    listOfTokens = re.split(r"\W*",bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

'''
获取词汇表，并将词汇表进行向量化

'''
# 英文单词过滤重复
def createVocablist(dataSet):
    vocabSet = set([])  # 创建一个空的，不重复的列表 set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)  #  | 表示 and   并集
    return list(vocabSet)  #  返回一个list列表


# 英文单词过滤重复
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个空的，不重复的列表 set()
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # | 表示 and   并集
    return list(vocabSet)  # 返回一个list列表

# 向量化  向量的每个元素为1 或者 0
# 问题：单词的频率怎么体现出来？
def setOfWord2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)  # 行向量都初始化为 0
    for word in inputSet():
        if word in vocabList:
            returnVec[vocabList.index(word)] =1
    return returnVec

'''
根据vocabList 词汇表，构建次袋模型
'''
def setOfWords2Vec(vocablist,inputSet):
    returnVec = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec

'''
# 朴素贝叶斯分类器训练函数
函数说明:朴素贝叶斯分类器训练函数
Parameters:
    trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
    trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
Returns:
    p0Vect - 非侮辱类的条件概率数组
    p1Vect - 侮辱类的条件概率数组
    pAbusive - 文档属于侮辱类的概率
'''
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)  #统计训练文档数目
    numWords =len(trainMatrix[0])    #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs) #文档属于侮辱类的概率
    p0Num = np.ones(numWords);p1Num = np.ones(numWords) #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑
    p0Denom = 2.0;p1Denom = 2.0  #分母初始化为2,拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]

            p1Denom += sum(trainMatrix[i])

        else:  # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = np.log(p1Num / p1Denom)  # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


'''
函数说明：朴素贝叶斯分类器分类函数
  Parameters:
      vec2Classify  - 待分类的词条数组
  Parameters:
    vec2Classify - 待分类的词条数组
    p0Vec - 非侮辱类的条件概率数组
    p1Vec -侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率

Returns:
    0 - 属于非侮辱类
    1 - 属于侮辱类
'''

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)        #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


'''
函数说明:测试朴素贝叶斯分类器
'''

def spamTest():
    docList = []; classList = []; fullText = []
    for i in range(1, 26):                                                  #遍历25个txt文件
        wordList = textParse(open('Y:/DataSet/Bayes/email/spam/%d.txt' % i, 'r').read())     #读取每个垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(1)                                                 #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('Y:/DataSet/Bayes/email/ham/%d.txt' % i, 'r').read())      #读取每个非垃圾邮件，并字符串转换成字符串列表
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)                                                 #标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList)                                    #创建词汇表，不重复

    trainingSet = list(range(50)); testSet = []                             #创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):                                                     #从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))                #随机选取索索引值
        testSet.append(trainingSet[randIndex])                              #添加测试集的索引值
        del(trainingSet[randIndex])                                         #在训练集列表中删除添加到测试集的索引值
    # 训练贝叶斯分类器， 得到 poV p1V pspam
    trainMat = []; trainClasses = []                                        #创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet:                                            #遍历训练集
        print("docIndex!!!!!",docIndex)
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))       #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])                            #将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  #训练朴素贝叶斯模型
    # 测试
    errorCount = 0                                                         #错误分类计数
    for docIndex in testSet:                                               #遍历测试集
        print("testSet!!!",testSet)
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])           #测试集的词集模型
        print('docList[docIndex]',docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:    #如果分类错误
            errorCount += 1                                                 #错误计数加1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ =='__main__':
    spamTest()





