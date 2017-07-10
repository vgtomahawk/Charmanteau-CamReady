import re

uniquePorts=set()

wikiFile=open("../Data/uniquePorts.csv")

for line in wikiFile:
    words=re.split("\W+",line)
    words=[word.lower() for word in words]
    words=words[:-1]
    portString=",".join(words)
    uniquePorts.add(portString)

wikiFile.close()

neoFile=open("../Data/uniquePorts2.csv")

for line in neoFile:
    words=re.split("\W+",line)
    words=[word.lower() for word in words]
    words=words[:-1]
    portString=",".join(words)
    uniquePorts.add(portString)

neoFile.close()


print len(uniquePorts)
uniqueFile=open("../Data/finalPorts.csv","w")

for element in uniquePorts:
    uniqueFile.write(element+"\n")

uniqueFile.close()
