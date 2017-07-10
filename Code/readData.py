import re
from collections import defaultdict
from string import ascii_lowercase

def rotate(lst,x):
    return lst[-x:]+lst[:-x]


def getData(trainingPoints=700,validPoints=300,filterKnight=False,removeKnight=False,crossValidate=False,foldId=0):
    data=[]
    data_words=[]

    for line in open("../Data/dataset.csv"):
        words=re.split("\W+",line)
        
        if filterKnight:
            if words[3]!='knight':
                continue

        if removeKnight:
            if words[3]=='knight':
                continue

        words=words[:3]
        char_words=[list(word.lower()) for word in words]

        data.append(char_words)
        data_words.append(words)
        #print words
        #print char_words
    

    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    #wids["EPSILON"]=2
    STOP=0
    SEPARATOR=1
    #EPSILON=2

    for c in ascii_lowercase:
        wids[c]=len(wids)

    data=[[[wids[character] for character in elem] for elem in x] for x in data]
    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]

    if crossValidate==False:
        trainInputs=dataInputs[:trainingPoints]
        trainOutputs=dataOutputs[:trainingPoints]
        validInputs=dataInputs[trainingPoints:trainingPoints+validPoints]
        validOutputs=dataOutputs[trainingPoints:trainingPoints+validPoints]
        testInputs=dataInputs[trainingPoints+validPoints:]
        testOutputs=dataOutputs[trainingPoints+validPoints:]

    else:
        foldSize=len(data)/10
        numOfFolds=10
        validFoldId=foldId
        testFoldId=(foldId+1)%numOfFolds
        trainInputs=[]
        validInputs=[]
        testInputs=[]
        trainOutputs=[]
        validOutputs=[]
        testOutputs=[]
        for i in range(numOfFolds):
            if i==validFoldId:
                validInputs=validInputs+dataInputs[i*foldSize:(i+1)*foldSize]
                validOutputs=validOutputs+dataOutputs[i*foldSize:(i+1)*foldSize]
            elif i==testFoldId:
                testInputs=testInputs+dataInputs[i*foldSize:(i+1)*foldSize]
                testOutputs=testOutputs+dataOutputs[i*foldSize:(i+1)*foldSize]
            else:
                trainInputs=trainInputs+dataInputs[i*foldSize:(i+1)*foldSize]
                trainOutputs=trainOutputs+dataOutputs[i*foldSize:(i+1)*foldSize]
        
        trainInputs=trainInputs+dataInputs[numOfFolds*foldSize:len(dataInputs)]
        trainOutputs=trainOutputs+dataOutputs[numOfFolds*foldSize:len(dataInputs)]

    #print wids["STOP"]
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids
    #return dataInputs,dataOutputs,wids

def getDataPhonoLexInput(trainingPoints=700,validPoints=300):
    data=[]
    data_words=[]

    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    wids_phonetic=defaultdict(lambda: 0)
    wids_phonetic["STOP"]=0
    wids_phonetic["SEPARATOR"]=1

    for c in ascii_lowercase:
        wids[c]=len(wids)

    phonemeSet=set()
    for line in open("../Data/datasetPhonetic.csv"):
        words=re.split(",",line)
        for word in words[1].split("|"):
            phonemeSet.add(word)
        for word in words[2].split("|"):
            phonemeSet.add(word)

    for phoneme in phonemeSet:
        wids_phonetic[phoneme]=len(wids_phonetic)


    for line in open("../Data/datasetPhonetic.csv"):
        words=re.split(",",line)
        #words=words[:3]
        #print words
        char_words_0=list(words[0].lower())
        char_words_1=words[1].split("|")
        char_words_2=words[2].split("|")
        char_words_4=list(words[4].lower())
        char_words_5=list(words[5].lower())
        char_words_0=[wids[character] for character in char_words_0]
        char_words_1=[wids_phonetic[character] for character in char_words_1]
        char_words_2=[wids_phonetic[character] for character in char_words_2]
        char_words_4=[wids[character] for character in char_words_4]
        char_words_5=[wids[character] for character in char_words_5]
        #print char_words_0
        #print char_words_4
        #print char_words_5
        #exit()
        data.append([char_words_0,char_words_1,char_words_2,char_words_4,char_words_5])

    STOP=0
    SEPARATOR=1

    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataAuxInputs=[x[3]+[SEPARATOR,]+x[4]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]
   
    trainInputs=dataInputs[:trainingPoints]
    trainAuxInputs=dataAuxInputs[:trainingPoints]
    trainOutputs=dataOutputs[:trainingPoints]
    validInputs=dataInputs[trainingPoints:trainingPoints+validPoints]
    validAuxInputs=dataAuxInputs[trainingPoints:trainingPoints+validPoints]
    validOutputs=dataOutputs[trainingPoints:trainingPoints+validPoints]
    testInputs=dataInputs[trainingPoints+validPoints:]
    testAuxInputs=dataAuxInputs[trainingPoints+validPoints:]
    testOutputs=dataOutputs[trainingPoints+validPoints:]

    #print wids["STOP"]
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids,wids_phonetic,trainAuxInputs,validAuxInputs,testAuxInputs


def getDataPhoneticInput(trainingPoints=700,validPoints=300):
    data=[]
    data_words=[]

    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    wids_phonetic=defaultdict(lambda: 0)
    wids_phonetic["STOP"]=0
    wids_phonetic["SEPARATOR"]=1

    for c in ascii_lowercase:
        wids[c]=len(wids)

    phonemeSet=set()
    for line in open("../Data/datasetPhonetic.csv"):
        words=re.split(",",line)
        for word in words[1].split("|"):
            phonemeSet.add(word)
        for word in words[2].split("|"):
            phonemeSet.add(word)

    for phoneme in phonemeSet:
        wids_phonetic[phoneme]=len(wids_phonetic)


    for line in open("../Data/datasetPhonetic.csv"):
        words=re.split(",",line)
        words=words[:3]
        #print words
        char_words_0=list(words[0].lower())
        char_words_1=words[1].split("|")
        char_words_2=words[2].split("|")
        char_words_0=[wids[character] for character in char_words_0]
        char_words_1=[wids_phonetic[character] for character in char_words_1]
        char_words_2=[wids_phonetic[character] for character in char_words_2]
        data.append([char_words_0,char_words_1,char_words_2])

    STOP=0
    SEPARATOR=1

    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]
   
    trainInputs=dataInputs[:trainingPoints]
    trainOutputs=dataOutputs[:trainingPoints]
    validInputs=dataInputs[trainingPoints:trainingPoints+validPoints]
    validOutputs=dataOutputs[trainingPoints:trainingPoints+validPoints]
    testInputs=dataInputs[trainingPoints+validPoints:]
    testOutputs=dataOutputs[trainingPoints+validPoints:]

    #print wids["STOP"]
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids,wids_phonetic
    #return dataInputs,dataOutputs,wids


def getDataDisjoint(trainingPoints=500,validPoints=200):
    data=[]
    data_words=[]

    for line in open("../Data/dataset.csv"):
        words=re.split("\W+",line)
        words=words[:3]
        char_words=[list(word.lower()) for word in words]

        data.append(char_words)
        data_words.append(words)
        print words
        print char_words


    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    STOP=0
    SEPARATOR=1

    for c in ascii_lowercase:
        wids[c]=len(wids)

    data=[[[wids[character] for character in elem] for elem in x] for x in data]
    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]
   
    print len(dataInputs)
    trainInputs=dataInputs[:trainingPoints]
    trainOutputs=dataOutputs[:trainingPoints]
    trainVocab=set()

    print len(trainInputs)
    for elem in data_words[:trainingPoints]:
        trainVocab.add(elem[1])
        trainVocab.add(elem[2])
    
    #print trainVocab

    validInputs=[]
    validOutputs=[]
    validVocab=set()

    for i,elem in enumerate(dataInputs):
        parentWord1=data_words[i][1]
        parentWord2=data_words[i][2]

        if (parentWord1 in trainVocab) or (parentWord2 in trainVocab):
            continue
        
        validInputs.append(dataInputs[i])
        validOutputs.append(dataOutputs[i])
        validVocab.add(parentWord1)
        validVocab.add(parentWord2)
        
        if len(validInputs)>=validPoints:
            break

    print len(validInputs)    
    #print validVocab

    testInputs=[]
    testOutputs=[]

    for i,elem in enumerate(dataInputs):
        parentWord1=data_words[i][1]
        parentWord2=data_words[i][2]

        if (parentWord1 in trainVocab) or (parentWord2 in trainVocab):
            continue
        
        if (parentWord1 in validVocab) or (parentWord2 in validVocab):
            continue
        
        testInputs.append(dataInputs[i])
        testOutputs.append(dataOutputs[i])
    
    print len(testInputs)
    #validInputs=dataInputs[trainingPoints:trainingPoints+validPoints]
    #validOutputs=dataOutputs[trainingPoints:trainingPoints+validPoints]
    #testInputs=dataInputs[trainingPoints+validPoints:]
    #testOutputs=dataOutputs[trainingPoints+validPoints:]

    #print wids["STOP"]
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids
    #return dataInputs,dataOutputs,wids

def getDataKnightHoldOut(trainingPoints=1000):
    data=[]
    data_words=[]

    knightData=[]
    knightData_words=[]

    for line in open("../Data/dataset.csv"):
        words=re.split("\W+",line)
        knightFlag=words[3]
        words=words[:3]
        char_words=[list(word.lower()) for word in words]
        
        if knightFlag=="knight":
            knightData.append(char_words)
            knightData_words=(words)
            print words
            print char_words
        else:
            data.append(char_words)
            data_words.append(words)
            print words
            print char_words
    
    print "Knight Words",len(knightData)
    print "Other Words",len(data)

    wids=defaultdict(lambda: 0)
    wids["STOP"]=0
    wids["SEPARATOR"]=1
    STOP=0
    SEPARATOR=1

    for c in ascii_lowercase:
        wids[c]=len(wids)

    data=[[[wids[character] for character in elem] for elem in x] for x in data]
    dataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in data]
    dataOutputs=[x[0]+[STOP,] for x in data]

    knightData=[[[wids[character] for character in elem] for elem in x] for x in knightData]
    knightDataInputs=[x[1]+[SEPARATOR,]+x[2]+[STOP,] for x in knightData]
    knightDataOutputs=[x[0]+[STOP,] for x in knightData]
 

    trainInputs=dataInputs[:trainingPoints]
    trainOutputs=dataOutputs[:trainingPoints]
    validInputs=dataInputs[trainingPoints:]
    validOutputs=dataOutputs[trainingPoints:]
    testInputs=knightDataInputs
    testOutputs=knightDataOutputs
    
    return trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids



def reverseDictionary(dic):
    reverseDic={}
    for x in dic:
        reverseDic[dic[x]]=x
    return reverseDic

if __name__=="__main__":
    #trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids,wids_phonetic,trainAuxInputs,validAuxInputs,testAuxInputs=getDataPhonoLexInput()
    #print wids_phonetic
    for i in range(10):
        trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids=getData(crossValidate=True,filterKnight=True,foldId=i)
        print len(trainInputs)
        print len(validInputs)
        print len(testInputs)


