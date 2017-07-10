from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import models
import random

def sampleFromDistribution(vals):
    p = random.random()
    s=0.0
    for i,v in enumerate(vals):
        s+=v
        if s>=p:
            return i
    return len(vals)-1

def generateSentence(model, word_to_index, start_token, end_token, unknown_token ):
    x = [ word_to_index[start_token] ]
    i=1
    while i<config.MAX_SEQUENCE_LENGTH:
        x_temp = pad_sequences([x], maxlen=config.MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        data = np.array( [ sequence[:-1] for sequence in x_temp] ) # only 1 sequence is actually there
        y = model.predict( data )
        #if i==1:
        idx = sampleFromDistribution(y[0][i])    
        #else:
        #    idx = np.argmax(y[0][i])
        if idx == word_to_index[end_token]:
            return x
        if idx == word_to_index[unknown_token]:
            i = 1
            x = [ word_to_index[start_token] ]
            print "Found unknown char. Retrying."
            continue
        x.append(idx)
        i += 1
    return x[1:] # removing sentence start


def getCMUDictData(fpath="./data/cmudict-0.7b"):
	print "--- Loading CMU data"
	data = open(fpath,"r").readlines()
	data = data[126:]
	data = [row.split(' ')[0] for row in data]
	data = [row.lower() for row in data]
	print "length of cmu data ",len(data)
	print "A couple of samples..."
	print data[0]
	print data[1]
	print data[2]
	print "------------"
	return data


if __name__ == "__main__":
	getCMUDictData()
