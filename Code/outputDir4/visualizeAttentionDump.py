import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(font_scale=2.5)

inFile=open("attentionDump_Bennifer.txt")

lines=inFile.readlines()
lines=[[float(x) for x in line.split()] for line in lines]

lines=np.array(lines)

lines=(lines.T)**(0.500001)

print np.shape(lines)

sns.heatmap(lines,cmap="Greys",xticklabels=["b","e","n","n","i","f","e","r","."],yticklabels=["b","e","n",";","j","e","n","n","i","f","e","r","."],cbar=False)
#sns.heatmap(lines,cmap="Greys",xticklabels=["s","l","u","r","v","e","."],yticklabels=["s","l","i","d","e","r",";","c","u","r","v","e","."],cbar=False)
#sns.heatmap(lines,cmap="Greys",xticklabels=["S","L","U","R","V","E","."],yticklabels=["S","L","I","D","E","R",";","C","U","R","V","E","."],cbar=False)

plt.yticks(rotation=0)
plt.show()

#plt.imshow(lines, cmap='Greys')
#plt.show()
