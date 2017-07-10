import re
import random

finalOutFile=open("dataset.csv","w")

knightPorts=set()

knightFile=open("knightPortsFormatted.csv")

for line in knightFile:
    words=re.split("\W+",line)
    words=words[:3]
    knightPorts.add(",".join(words))

knightFile.close()

print len(knightPorts)

finalPortFile=open("finalPorts.csv")
finalPorts=set()

for line in finalPortFile:
    words=re.split("\W+",line)
    words=words[:3]
    finalPorts.add(",".join(words))

finalPortFile.close()

print len(finalPorts)

unionSet=knightPorts.union(finalPorts)

print len(unionSet)


finalList=[]

for line in unionSet:
    if line in knightPorts:
        finalList.append(line+",knight")
    else:
        finalList.append(line+",other")

print len(finalList)

random.shuffle(finalList)

for line in finalList:
    finalOutFile.write(line+"\n")

finalOutFile.close()
