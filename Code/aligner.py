


def findAlignment(w,z,epsilon=0,SEPARATOR="SEPARATOR",mode="KEEPGEN",KEEP=None):
    x=w[:w.index(SEPARATOR)]
    y=w[w.index(SEPARATOR)+1:-1]
    alignment=[]
    i_z=0
    state=True
    for i,c_x in enumerate(x):
        if state and i_z<len(z[:-1]) and x[i]==z[:-1][i_z]:
            if mode=="KEEPDELETE":
                alignment.append(KEEP)
            else:
                alignment.append(c_x)
            i_z+=1
        else:
            state=False
            alignment.append(epsilon)
    
    alignment.append(epsilon)

    zSearch=z[i_z:-1]
    state=True
    i=0
    while i<len(y):
        if state and len(zSearch)==len(y[i:i+len(zSearch)]) and zSearch==y[i:i+len(zSearch)]:
            state=False
            if mode=="KEEPDELETE":
                alignment+=[KEEP,]*len(zSearch)
            else:
                alignment+=y[i:i+len(zSearch)]
            i+=len(zSearch)
        else:
            alignment.append(epsilon)
            i+=1

    alignment.append(z[-1])

    if mode!="KEEPDELETE":
        if not elementEqual(strip(alignment,epsilon=epsilon),z):
            alignment=[]
    else:
        if len(strip(alignment,epsilon=epsilon))!=len(z):
            alignment=[]

    return alignment

def strip(z,epsilon=0):
    zprime=[]
    for c_z in z:
        if c_z!=epsilon:
            zprime.append(c_z)
    return zprime

def elementEqual(x,y):
    if len(x)!=len(y):
        return False

    equal=True
    for i,c_x in enumerate(x):
        if c_x!=y[i]:
            equal=False
            break

    return equal

if __name__=="__main__":
    mode="KEEPDELETE"
    KEEP="A"
    x=["f","r","i","e","n","d","0","e","n","e","m","y","S"]
    z=["f","r","e","n","e","m","y","S"]
    print x
    a1=findAlignment(x,z,SEPARATOR="0",epsilon="0",mode=mode,KEEP=KEEP)
    print a1
    print strip(a1,epsilon="0")
    x=["c","o","d","i","n","g","0","m","o","z","i","l","l","a","n","S"]
    z=["c","o","d","z","i","l","l","a","S"]
    print x
    a2=findAlignment(x,z,SEPARATOR="0",epsilon="0",mode=mode,KEEP=KEEP)
    print a2
    print strip(a2,epsilon="0")
    print elementEqual([1,2,3],[1,2])
    print elementEqual([1,2,3],[4,5,6])
    print elementEqual([1,2,3],[1,2,3])
    x=["f","r","i","e","n","d","0","e","n","e","m","y","S"]
    z=["f","r","e","i","e","m","y","S"]
    print x
    a1=findAlignment(x,z,SEPARATOR="0",epsilon="0",mode=mode,KEEP=KEEP)
    print a1
    x=["d","a","r","k","0","a","r","c","a","n","e","S"]
    z=["d","a","r","k","a","n","e","S"]
    print x
    a1=findAlignment(x,z,SEPARATOR="0",epsilon="0",mode=mode,KEEP=KEEP)
    print a1
    print strip(a1,epsilon="0")
  
