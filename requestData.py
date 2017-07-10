import requests
import json
from bs4 import BeautifulSoup
import re
import time

inFile=open("Data/dataset.csv")
outFile=open("Data/dataAliyaScraped.csv","w")

lineIndex=0
for line in inFile:
    words=re.split("\W+",line)
    parent1=words[1]
    parent2=words[2]
    print lineIndex
    print parent1
    print parent2
    parentString="24:"+parent1+":"+parent2
    #print parentString
    s="http://leps.isi.edu/fst/ajaxGenAll.php"
    data = json.dumps( {0: parentString } )
    response = requests.post( s, data )
    responseText=response.text
    index1=responseText.index(":</p>")
    index2=responseText.index("<p>Try")
    #print index1
    #print index2
    answer=responseText[index1+11:index2-13]
    print answer
    outFile.write(parent1+" "+parent2+" "+answer+"\n")
    time.sleep(1)
    lineIndex+=1
    if lineIndex%100==0:
        time.sleep(6)

outFile.close()
