import random
import re

fp=open("knightPortsFormatted.csv")

lines=fp.readlines()

fp.close()

for index,line in enumerate(lines):
    words=re.split("\W+",line)
    print len(words)
    if len(words)>=5:
        print index
        print len(words)

"""
random.shuffle(lines)

for line in lines:
    print line

fp=open("finalPorts.csv","w")

for line in lines:
    fp.write(line)

fp.close()
"""
