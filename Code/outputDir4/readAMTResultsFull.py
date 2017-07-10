import csv
import sys

results=[]
fieldNames=None
with open(sys.argv[1]) as csvfile:
    reader=csv.DictReader(csvfile)
    fieldNames=reader.fieldnames
    for row in reader:
        results.append(row)

wordAnswers=[]

answers={}
rowIds={}
completeRows=0
for rowId,row in enumerate(results):
    skipRow=False
    for k in range(1,7):
        key1="Input.word"+str(k)+"1"
        key2="Input.word"+str(k)+"2"
        instanceId=int(row["Input.q"+str(k)])
        key3="Answer."+str(instanceId)
        print row[key1]
        print row[key2]
        if key3 not in row:
            skipRow=True
            break
        answers[row[key1],row[key2]]=[int(row[key3]),rowId]
        if (row[key1],row[key2]) not in rowIds:
            rowIds[row[key1],row[key2]]=[]
        rowIds[row[key1],row[key2]].append([int(row[key3]),rowId])

    if skipRow:
        continue
    completeRows+=1

print "Complete Rows:",completeRows

#print answers
print "Distinct Answers:",len(answers)


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


print "Knight Much Better:",knightMuchBetter
#print knightMuchBetterList
print "Knight Better:",knightBetter
#print knightBetterList
print "System Better:",systemBetter
#print systemBetterList
print "System Much Better",systemMuchBetter
#print systemMuchBetterList

#print baselineLogs

print "Ground Truth Better",groundTruthBetter
#print reliableRowIds

rowId=0
#fieldNames.append("Approve")

approvedRows=0
augmentedResults=[]
for row in results:
    if rowId in reliableRowIds:
        row["Approve"]="x"
        approvedRows+=1
    else:
        row["Approve"]=""
        row["Reject"]="UNRELIABLE_JUDGEMENT"
    augmentedResults.append(row)
    rowId+=1

print approvedRows

#print augmentedResults[1]["Approve"]

with open('approval.csv','w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=fieldNames)

    writer.writeheader()
    for row in augmentedResults:
        writer.writerow(row)

with open('approval.csv') as csvfile:
    reader=csv.DictReader(csvfile)
    print reader.fieldnames
    for row in reader:
        print row["Reject"]
        break
