import readData

for f in range(10):
    trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids=readData.getData(filterKnight=True,crossValidate=True,foldId=f)
    print len(trainInputs)
    print len(validInputs)
    print len(testInputs)
    print "Vocab Size:",len(wids)
    print "First Input:",trainInputs[0]
    print "First Output:",trainOutputs[0]
    reverse_wids=readData.reverseDictionary(wids)
    print "First Input Raw",[reverse_wids[c] for c in trainInputs[0]]
    print "First Output Raw",[reverse_wids[c] for c in trainOutputs[0]]
