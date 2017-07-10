import sys
import random
import re

random.seed(500)

"""Format: <exampleId> <parent1> <parent2> <groundTruth> 0 <knight> <system>
       OR <exampleId> <parent1> <parent2> <groundTruth> 1 <system> <knight>"""
inFile=open("amtFile.txt")

lines=inFile.readlines()
lines=[line.split()+["actual",] for line in lines]
exampleCount=len(lines)

checkingFile=open("checkingFile2.txt")
checkingLines=checkingFile.readlines()

checkingLinesConverted=[]
for line in checkingLines:
    words=re.split("\W+",line)
    exampleId=exampleCount+int(words[0])
    parent1=words[1]
    parent2=words[2]
    groundTruth=words[3]
    coinToss=random.randint(0,1)
    if coinToss==0:
        option1=groundTruth
        option2=words[4]
    else:
        option1=words[4]
        option2=groundTruth
    convertedLine=[str(exampleId),parent1,parent2,groundTruth,str(coinToss),option1,option2,"baseline"]
    checkingLinesConverted.append(convertedLine)

lineId=0

repeatedLines=[]
repeats=1

for i in range(repeats):
    random.shuffle(lines)
    repeatedLines+=lines

print len(repeatedLines)

i=0
k=5

batches=[]

while i<len(repeatedLines):
    batch=repeatedLines[i:min(i+k,len(repeatedLines))]
    batches.append(batch)
    i+=k

print exampleCount
print len(batches)
print len(checkingLines)

print lines[0]
print checkingLinesConverted[0]
print len(batches[0])
print len(batches[1])

instanceId=0

amtOutputFile=open("amtOutputFile.csv","w")
amtLogFile=open("amtLogFile.txt","w")

for batch in batches:
    baselineLine=random.choice(checkingLinesConverted)
    batchToWrite=batch+[baselineLine,]
    random.shuffle(batchToWrite)
    reps=[]
    for line in batchToWrite:
        amtLogFile.write(" ".join([str(instanceId),]+line)+"\n")
        reps.append(",".join([line[1],line[2],line[5],line[6],str(instanceId)]))
        instanceId+=1
    reps=",".join(reps)
    amtOutputFile.write(reps+"\n")

print instanceId
amtLogFile.close()
amtOutputFile.close()
