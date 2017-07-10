import csv
import sys

results=[]
with open(sys.argv[1]) as csvfile:
    reader=csv.DictReader(csvfile)
    for row in reader:
        results.append(row)

wordAnswers=[]

answers={}
rowIds={}
for rowId,row in enumerate(results):
    for k in range(1,7):
        key1="Input.word"+str(k)+"1"
        key2="Input.word"+str(k)+"2"
        instanceId=int(row["Input.q"+str(k)])
        key3="Answer."+str(instanceId)
        answers[row[key1],row[key2]]=[int(row[key3]),rowId]
        if (row[key1],row[key2]) not in rowIds:
            rowIds[row[key1],row[key2]]=[]
        rowIds[row[key1],row[key2]].append([int(row[key3]),rowId])

print answers
print len(answers)


actualLogs={}
baselineLogs={}
for line in open(sys.argv[2]):
    words=line.split()
    #print words[2],words[3],int(words[5]),words[6],words[7],words[8]
    if words[8]=="actual":
        if (words[2],words[3]) in answers:
            coinToss=int(words[5])
            answer=answers[words[2],words[3]]
            actualLogs[words[2],words[3]]=[int(words[5]),answers[words[2],words[3]][0],answers[words[2],words[3]][1]]
    elif words[8]=="baseline":
        if (words[2],words[3]) in answers:
            answerList=rowIds[words[2],words[3]]
            for answer in answerList:
                baselineLogs[words[2],words[3],words[6],words[7],answer[1]]=[int(words[5]),answer[0],answer[1]]

print len(actualLogs)
print len(baselineLogs)

groundTruthBetter=0
reliableRowIds=set()
for key,value in baselineLogs.items():
    coinToss=value[0]
    answer=value[1]
    if coinToss==0:
        if answer==1 or answer==2:
            groundTruthBetter+=1
            reliableRowIds.add(value[2])
    else:
        if answer==3 or answer==4:
            groundTruthBetter+=1
            reliableRowIds.add(value[2])

rowFilter=True
knightMuchBetter=0
knightBetter=0
systemBetter=0
systemMuchBetter=0

knightMuchBetterList=[]
knightBetterList=[]
systemBetterList=[]
systemMuchBetterList=[]

for key,value in actualLogs.items():
    coinToss=value[0]
    answer=value[1]
    if rowFilter:
        if value[2] not in reliableRowIds:
            continue

    if coinToss==0:
        if answer==1:
            knightMuchBetter+=1
            knightMuchBetterList.append(key)
        elif answer==2:
            knightBetter+=1
            knightBetterList.append(key)
        elif answer==3:
            systemBetter+=1
            systemBetterList.append(key)
        else:
            systemMuchBetter+=1
            systemMuchBetterList.append(key)
    else:
        if answer==1:
            systemMuchBetter+=1
            systemMuchBetterList.append(key)
        elif answer==2:
            systemBetter+=1
            systemBetterList.append(key)
        elif answer==3:
            knightBetter+=1
            knightBetterList.append(key)
        else:
            knightMuchBetter+=1
            knightMuchBetterList.append(key)


print knightMuchBetter
print knightMuchBetterList
print knightBetter
print knightBetterList
print systemBetter
print systemBetterList
print systemMuchBetter
print systemMuchBetterList

print baselineLogs

print groundTruthBetter
print reliableRowIds
