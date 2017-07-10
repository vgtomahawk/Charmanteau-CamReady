import re
import numpy as np

def generateCandidates(w1, w2):
    n=len(w1)
    m=len(w2)
    all_candidates = []
    for i in range(1,n+1):
        for j in range(m):
            all_candidates.append(w1[0:i]+w2[j:])
    #print "len(all_candidates)= ",len(all_candidates)
    all_candidates=set(all_candidates)
    #print "len(all_candidates) after deduplication= ",len(all_candidates)
    return all_candidates

def getMaxSubsequence(w1,w2):
    ret=0
    for i in range(min(len(w1),len(w2))):
        if w1[i]!=w2[i]:
            return i
    return min( len(w1),len(w2) )

def getMaxSubsequenceRev(w1,w2):
    wtemp1=''.join(reversed(list(w1)))
    wtemp2=''.join(reversed(list(w2)))
    return getMaxSubsequence(wtemp1,wtemp2)

def scoresToRanks(scores, rev=False):
    sorted_scores = sorted(scores)
    ranks=[]
    prev=-1
    prev_rank=None
    m=len(scores)
    for i,score in enumerate(sorted_scores):
        if score==prev:
            ranks.append(prev_rank)
        else:
            ranks.append(i+1)
            prev_rank=i+1
            prev=score
    ranks = {score:rank for score,rank in zip(sorted_scores,ranks)}
    if rev:
        return [(m-ranks[s]+1) for s in scores]
    else:
        return [ranks[s] for s in scores]

def getEditDistance(w1,w2):
    ret=0
    n,m = len(w1),len(w2)
    dp=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            dp[i][j]=n*m
    for i,ch1 in enumerate(w1):
        for j,ch2 in enumerate(w2):
            if i==0 and j==0:
                if ch1==ch2:
                    dp[i][j]=0
                else:
                    dp[i][j]=2
            elif i==0:
                if ch1==ch2:
                    dp[i][j]=j
                else:
                    dp[i][j]=1+dp[i][j-1]
            elif j==0:
                if ch1==ch2:
                    dp[i][j]=i
                else:
                    dp[i][j]=1+dp[i-1][j]
            else:
                cost=1
                dp[i][j] = min( dp[i][j], cost + min(dp[i-1][j], dp[i][j-1])  )
                if ch1==ch2:
                    dp[i][j] = min(dp[i][j], dp[i-1][j-1])
    return dp[n-1][m-1]


def spitToBlendableOrNot(data):
    all_w1,all_w2,all_gold=data
    typ=[]
    for w1,w2,gold in zip(all_w1,all_w2,all_gold):
        a=getMaxSubsequence(w1,gold)
        b=getMaxSubsequenceRev(w2,gold)
        if (a+b)>=len(gold):
            typ.append(1)
        else:
            typ.append(0)
    return typ

def evaluate(gold, pred):
    edits = []
    em=0
    for g,p in zip(gold, pred):
        edits.append(getEditDistance(g,p))
        if g==p:
            em+=1
    avg_edit = np.mean(edits)
    print "avg_edit, em: ",avg_edit,em

if __name__=="__main__":

    data = open("baseline_results.txt","r").readlines()
    gold = [row.split('\t')[2].strip() for row in data]
    pred = [row.split('\t')[3].strip() for row in data]
    w1 = [row.split('\t')[0].strip() for row in data]
    w2 = [row.split('\t')[1].strip() for row in data]
    splt_types = spitToBlendableOrNot((w1,w2,gold))
    gold1=[x  for i,x in enumerate(gold) if splt_types[i]==1]
    pred1 = [x for i,x in enumerate(pred) if splt_types[i]==1 ]
    print len(gold1)
    print len(pred1)
    evaluate(gold1,pred1)
    gold2=[x  for i,x in enumerate(gold) if splt_types[i]==0]
    pred2 = [x for i,x in enumerate(pred) if splt_types[i]==0 ]
    print len(gold2)
    print len(pred2)
    evaluate(gold2,pred2)
    '''

    #print generateCandidates("abc","cdef")
    print scoresToRanks([3,5,1], True)
    print scoresToRanks([1,3,5,5,7], True)
    print scoresToRanks([7,7,7,1,1,1,9,8,8,8])
    print "-------"
    print getEditDistance("abc","abc")
    print getEditDistance("a","a")
    print getEditDistance("xyz","abc")
    print getEditDistance("xyzabxyz","abc")

    '''
