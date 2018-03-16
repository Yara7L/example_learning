import jieba

seg_list=jieba.cut("一个人的命运",cut_all=True)
print("full mode:"+"/".join(seg_list))

seg_list=jieba.cut("一个人的命运",cut_all=False)
print("default mode:"+"/".join(seg_list))

seg_list=jieba.cut("一个人的命运")
print("default mode:"+"/".join(seg_list))

print("output is a generateor:")
print(seg_list)

seg_list=jieba.lcut("一个人的命运",cut_all=True)
print(seg_list)
print("full mode:"+"/".join(seg_list))


def sent2word(sentence):
    seg_list=jieba.cut(sentence)
    segResult=[]
    for w in seg_list:
        segResult.append(w)

#stopwords
    stopwords =readLine('stop_words.txt')
    newSeg=[]
    for word in segResult:
        if word in stopwords:
            continue
        else:
            newSeg.append(word)
    return newSeg

def classifyWords(wordDict):
    senList=readLines('sentiment_score.txt')
    senDict=defaultdict()
    for s in senList:
        senDict[s.split(' ')[0]]=s.split(' ')[1]
    
    notList=readLines('notDict.txt')
    
    degreeList=readLines('degreeDict.txt')
    degreeDict=defaultdict()

    for d in degreeDict():
        degreeDict[d.split(',')[0]]=d.split(',')[1]
    
    senWord=defaultdict()
    notWord=defaultdict()
    degreeWord=defaultdict()

    for word in wordDict.keys():
        if word in senDict.keys() and word not in notList and word not in degreeDict.keys():
            senWord[wordDict[word]]=senDict[word]
        elif  word in notList and word not in degreeDict.keys():
            notWord[wordDict[word]]=-1
        elif word in degreeDict.keys():
            degreeWord[wordDict[word]]=degreeDict[word]
    return senWord,notWord,degreeWord


def scoreSent(senWord, notWord, degreeWord, segResult):
    W = 1
    score = 0
    # 存所有情感词的位置的列表
    senLoc = senWord.keys()
    notLoc = notWord.keys()
    degreeLoc = degreeWord.keys()
    senloc = -1
    # notloc = -1
    # degreeloc = -1

    # 遍历句中所有单词segResult，i为单词绝对位置
    for i in range(0, len(segResult)):
        # 如果该词为情感词
        if i in senLoc:
            # loc为情感词位置列表的序号
            senloc += 1
            # 直接添加该情感词分数
            score += W * float(senWord[i])
            # print "score = %f" % score
            if senloc < len(senLoc) - 1:
                # 判断该情感词与下一情感词之间是否有否定词或程度副词
                # j为绝对位置
                for j in range(senLoc[senloc], senLoc[senloc + 1]):
                    # 如果有否定词
                    if j in notLoc:
                        W *= -1
                    # 如果有程度副词
                    elif j in degreeLoc:
                        W *= float(degreeWord[j])
        # i定位至下一个情感词
        if senloc < len(senLoc) - 1:
            i = senLoc[senloc + 1]
    return score


