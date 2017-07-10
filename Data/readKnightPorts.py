
outFile=open("knightPortsFormatted.csv","w")
fp=open("knightPorts.csv")

lines=fp.readlines()

k=0

while k<len(lines):
    word1=lines[k][:-1]
    word2=lines[k+2][:-1]
    word3=lines[k+4][:-1]
    print word1
    print word2
    print word3
    outFile.write(word3+","+word1+","+word2+"\n")
    k+=8

outFile.close()
fp.close()
