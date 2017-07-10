import sys
import random

def getLengthList(alpha):
    return [len(x[0])+len(x[1]) for x in alpha]

def getLengthDistribution(inputs,lengthThresholds):
    outputs=[]
    for l in lengthThresholds:
        yAll=[]

        for x in inputs:
            if x>=l:
                yAll.append(x)
        outputs.append(len(yAll))
    return outputs


random.seed(20240094)

inFile=open(sys.argv[1])
outFile=open(sys.argv[2],"w")

lineList=inFile.readlines()
i=0

onlyKnightWrong=0
onlySystemWrong=0
bothWrong=0
bothRight=0
bothWrongAndDifferent=0

onlyKnightWrongList=[]
onlySystemWrongList=[]
bothWrongList=[]
bothRightList=[]
bothWrongAndDifferentList=[]
allList=[]

while i<len(lineList):
    words1=lineList[i].split()
    words2=lineList[i+1].split()
    words3=lineList[i+2].split()
    i+=4
    parentWord1=words1[0]
    parentWord2=words1[1]
    knightWord=words1[2]
    groundTruthWord=words2[1][words2[1].index(":")+1:]
    systemWord=words3[1][words3[1].index(":")+1:]
    tup=(parentWord1,parentWord2,groundTruthWord,knightWord,systemWord)
    if knightWord!=groundTruthWord and systemWord!=groundTruthWord:
        bothWrong+=1
        bothWrongList.append(tup)
        if knightWord!=systemWord:
            bothWrongAndDifferent+=1
            bothWrongAndDifferentList.append(tup)
            coinToss=random.randint(0,1)
            if coinToss==0:
                firstWord=knightWord
                secondWord=systemWord
            else:
                firstWord=systemWord
                secondWord=knightWord

            outFile.write(str(bothWrongAndDifferent-1)+" "+parentWord1+" "+parentWord2+" "+groundTruthWord+" "+str(coinToss)+" "+firstWord+" "+secondWord+"\n")    
    elif knightWord!=groundTruthWord and systemWord==groundTruthWord:
        onlyKnightWrong+=1
        onlyKnightWrongList.append(tup)
    elif knightWord==groundTruthWord and systemWord!=groundTruthWord:
        onlySystemWrong+=1
        onlySystemWrongList.append(tup)
    else:
        bothRight+=1
        bothRightList.append(tup)
    
    allList.append(tup)

print "Only Knight Wrong",onlyKnightWrong
print "Only System Wrong",onlySystemWrong
print "Both Wrong",bothWrong
print "Both Right",bothRight
print "Both Wrong and Different",bothWrongAndDifferent

outFile.close()

lengthThresholds=[0,10,15,20]

allListLengths=getLengthList(allList)
bothWrongListLengths=getLengthList(bothWrongList)
onlyKnightWrongListLengths=getLengthList(onlyKnightWrongList)
onlySystemWrongListLengths=getLengthList(onlySystemWrongList)
bothRightListLengths=getLengthList(bothRightList)

allListLengthThresholds=getLengthDistribution(allListLengths,lengthThresholds)
bothWrongListLengthThresholds=getLengthDistribution(bothWrongListLengths,lengthThresholds)
onlyKnightWrongListLengthThresholds=getLengthDistribution(onlyKnightWrongListLengths,lengthThresholds)
onlySystemWrongListLengthThresholds=getLengthDistribution(onlySystemWrongListLengths,lengthThresholds)
bothRightListLengthThresholds=getLengthDistribution(bothRightListLengths,lengthThresholds)

knightWrongs=zip(bothWrongListLengthThresholds,onlyKnightWrongListLengthThresholds)
knightWrongListLengthThresholds=[x[0]+x[1] for x in knightWrongs]

systemWrongs=zip(bothWrongListLengthThresholds,onlySystemWrongListLengthThresholds)
systemWrongListLengthThresholds=[x[0]+x[1] for x in systemWrongs]



print allListLengthThresholds
print [(x+0.0)/(y+0.0) for x,y in zip(knightWrongListLengthThresholds,allListLengthThresholds)]
print [(x+0.0)/(y+0.0) for x,y in zip(systemWrongListLengthThresholds,allListLengthThresholds)]
