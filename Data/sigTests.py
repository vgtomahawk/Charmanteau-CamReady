import re
groundTruth=open("dataset.csv")
datasetLines=groundTruth.readlines()

blindResults={}
otherInstances=0
for line in datasetLines:
    words=re.split("\W+",line)
    if words[3]=='other':
        blindResults[words[0]]={}
        blindResults[words[0]]["x1"]=words[1]
        blindResults[words[0]]["x2"]=words[2]

print len(blindResults)

baseline=open("baselineResults.txt").readlines()
print len(baseline)
