import re
import copy

for line in open("dataset.csv"):
    words=re.split("\W+",line)

cmuDict={}


for line in open("0745.dict"):
    words=line.split()
    key=words[0].lower()
    value="|".join(words[1:])
    cmuDict[key]=value


unknowns=set()

fp=open("datasetPhonetic.csv","w")

id=0
for line in open("dataset.csv"):
    words=re.split("\W+",line)
    originalWords=copy.deepcopy(words)

    if words[1] in cmuDict:
        words[1]=cmuDict[words[1]]
    else:
        words[1]="NULL"

    if words[2] in cmuDict:
        words[2]=cmuDict[words[2]]
    else:
        words[2]="NULL"

    if words[1]=="NULL":
        unknowns.add(originalWords[1])
    if words[2]=="NULL":
        unknowns.add(originalWords[2])

    words=words[:-1]+[originalWords[1],originalWords[2],words[-1]]

    print words
    fp.write(",".join(words)+"\n")

fp.close()
#for unknown in unknowns:
#    print unknown


