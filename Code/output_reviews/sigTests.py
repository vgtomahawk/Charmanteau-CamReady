import random
import editdistance
from utilities import generateCandidates
import math

random.seed(9008465)

backwardLines=open("best_blind_backward.txt").readlines()

resultTable={}
entries=[]

i=0
while i<len(backwardLines):
    line1=backwardLines[i]
    words=line1.split()
    word1=words[0]
    word2=words[1]
    baseline=words[2]
    line2=backwardLines[i+1]
    line2=(line2.split())[1]
    words=line2.split(":")
    groundTruth=words[1]
    line3=backwardLines[i+2]
    line3=(line3.split())[1]
    words=line3.split(":")
    backward=words[1]
    entry={}
    entry["w1"]=word1
    entry["w2"]=word2
    entry["groundTruth"]=groundTruth
    entry["baseline"]=baseline
    entry["backward"]=backward

    entry["rightBaseline"]=0.0
    if entry["baseline"]==entry["groundTruth"]:
        entry["rightBaseline"]=1.0
    
    entry["rightBackward"]=0.0
    if entry["baseline"]==entry["backward"]:
        entry["rightBackward"]=1.0

    entry["baselineDistance"]=editdistance.eval(groundTruth,baseline)
    entry["backwardDistance"]=editdistance.eval(groundTruth,backward)

    resultTable[",".join([word1,word2,groundTruth])]=entry
    entries.append(entry)
    i+=4
    #break


print len(resultTable)
print len(entries)

import sys
mode=sys.argv[1]

if mode=="COV":
    forwardGreedyLines=open("best_blind_forward_greedy.txt").readlines()

    found=0
    j=0
    while j<len(forwardGreedyLines):
        line1=forwardGreedyLines[j]
        words=line1.split()
        word1=words[0]
        word2=words[1]
        baseline=words[2]
        line2=forwardGreedyLines[j+1]
        line2=(line2.split())[1]
        words=line2.split(":")
        groundTruth=words[1]
        line3=forwardGreedyLines[j+2]
        line3=(line3.split())[1]
        words=line3.split(":")
        forwardGreedy=words[1]
        key=",".join([word1,word2,groundTruth])
        if key in resultTable:
            found+=1
            resultTable[key]["forwardGreedy"]=forwardGreedy
            resultTable[key]["rightForwardGreedy"]=0.0
            resultTable[key]["forwardGreedyDistance"]=editdistance.eval(resultTable[key]["groundTruth"],forwardGreedy)
            if resultTable[key]["forwardGreedy"]==resultTable[key]["groundTruth"]:
                resultTable[key]["rightForwardGreedy"]=1.0
        j+=4



    #print found
    entries=resultTable.values()
    covered=0
    baselineCovered=0
    forwardGreedyCovered=0
    baselineNotCovered=0
    forwardGreedyNotCovered=0
    baselineEditNotCovered=0.0
    forwardGreedyEditNotCovered=0.0
    backwardEditNotCovered=0.0
    for entry in entries:
        candidates=generateCandidates(entry["w1"],entry["w2"])
        if entry["groundTruth"] in candidates:
            entry["covered"]=True
            baselineCovered+=entry["rightBaseline"]
            forwardGreedyCovered+=entry["rightForwardGreedy"]
            covered+=1
        else:
            baselineNotCovered+=entry["rightBaseline"]
            forwardGreedyNotCovered+=entry["rightForwardGreedy"]
            baselineEditNotCovered+=entry["baselineDistance"]
            forwardGreedyEditNotCovered+=entry["forwardGreedyDistance"]
            backwardEditNotCovered+=entry["backwardDistance"]
            print "Baseline Prediction:",entry["baseline"]
            print "Baseline Edit Distance:",entry["baselineDistance"]
            print baselineEditNotCovered
            print "Forward Greedy Prediction:",entry["forwardGreedy"]
            print "Forward Greedy Distance:",entry["forwardGreedyDistance"]
            print forwardGreedyEditNotCovered
            print "Ground Truth:",entry["groundTruth"]
            entry["covered"]=False
    
    notCovered=len(entries)-covered
    print "Not covered",notCovered
    print baselineCovered
    print forwardGreedyCovered
    print baselineNotCovered
    print forwardGreedyNotCovered
    print baselineEditNotCovered/notCovered
    print forwardGreedyEditNotCovered/notCovered
    print backwardEditNotCovered/notCovered

elif mode=="SIG":
    M=1000
    N=611

    averageRightBackward=0.0
    averageRightBaseline=0.0
    backwardBeatsBaseline=0.0
    backwardMarginBaseline=0.0
    averageAverageEditBackward=0.0
    averageAverageEditBaseline=0.0
    localAverageBaselineEdits=[]
    localAverageBackwardEdits=[]
    rightBackwards=[]
    rightBaselines=[]

    for bootId in range(M):
        random.shuffle(entries)
        sampledEntries=entries[:N]
        
        rightBackward=0.0
        rightBaseline=0.0
        localAverageBackwardEdit=0.0
        localAverageBaselineEdit=0.0

        for entry in sampledEntries:
            rightBackward+=entry["rightBackward"]
            rightBaseline+=entry["rightBaseline"]
            localAverageBackwardEdit+=entry["backwardDistance"]
            localAverageBaselineEdit+=entry["baselineDistance"]

        localAverageBackwardEdit/=N
        localAverageBaselineEdit/=N
        localAverageBaselineEdits.append(localAverageBaselineEdit)
        localAverageBackwardEdits.append(localAverageBackwardEdit)
        averageAverageEditBackward+=localAverageBackwardEdit
        averageAverageEditBaseline+=localAverageBaselineEdit

        averageRightBackward+=rightBackward
        averageRightBaseline+=rightBaseline
        rightBackwards.append(rightBackward)
        rightBaselines.append(rightBaseline)

        if rightBackward>rightBaseline:
            backwardBeatsBaseline+=1.0
        if localAverageBackwardEdit<localAverageBaselineEdit-0.2:
            backwardMarginBaseline+=1

    averageRightBackward/=M*N
    averageRightBaseline/=M*N
    averageAverageEditBackward/=M
    averageAverageEditBaseline/=M

    backwardBeatsBaseline/=M
    backwardMarginBaseline/=M

    lowerIndex=int(math.ceil(0.025*N))
    
    localAverageBaselineEdits.sort()
    localAverageBackwardEdits.sort()
    rightBackwards.sort()
    rightBaselines.sort()

    localAverageBaselineEdits=localAverageBaselineEdits[lowerIndex:-lowerIndex]
    localAverageBackwardEdits=localAverageBackwardEdits[lowerIndex:-lowerIndex]

    rightBackwards=rightBackwards[lowerIndex:-lowerIndex]
    rightBaselines=rightBaselines[lowerIndex:-lowerIndex]

    print averageRightBackward
    print averageRightBaseline
    print backwardBeatsBaseline

    print "Backward Margin Baseline:",backwardMarginBaseline
    print "Average average edit distance Backward:",averageAverageEditBackward
    print "Average average edit distance Baseline:",averageAverageEditBaseline
    
    print "Edit Distance Baseline Lower:",localAverageBaselineEdits[0]
    print "Edit Distance Baseline Upper:",localAverageBaselineEdits[-1]
    print "Edit Distance Backward Lower:",localAverageBackwardEdits[0]
    print "Edit Distance Backward Upper:",localAverageBackwardEdits[-1]

    print "Matches Baseline Lower:",rightBaselines[0]/(N+0.0)
    print "Matches Baseline Upper:",rightBaselines[-1]/(N+0.0)
    print "Matches Backward Lower:",rightBackwards[0]/(N+0.0)
    print "Matches Backward Upper:",rightBackwards[-1]/(N+0.0)


