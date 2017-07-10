from collections import defaultdict
import numpy as np

def loadEmbeddings(wids,embeddingFile,EMB_SIZE):
    fp=open(embeddingFile)
    initParams=np.zeros((len(wids),EMB_SIZE))
    for line in fp:
        words=line.split()
        charKey=str(words[0])
        if charKey=="SENT_END":
            charKey="STOP"
        initParams[wids[charKey]]=[float(x) for x in words[1:]]
    
    return initParams


if __name__=="__main__":
    fileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"
    wids=defaultdict(int)
    wids["a"]=0
    wids["b"]=1
    wids["c"]=3
    wids["STOP"]=4
    initParams=loadEmbeddings(wids,fileName,50)
    print initParams
