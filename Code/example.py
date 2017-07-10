import matplotlib.pyplot as plt
import numpy as np
import csv
import pickle
import sys

sys.path.insert(0, sys.path[0]+'/language_model/')
print sys.path
import configuration as config
import utilities
import main_lm
import readData as datasets



args = ("load","./language_model/checkpoints/weights.134-1.86.hdf5",None)
lm_model = main_lm.RNNLanguageModelHandler( args )

if __name__=="__main__":
    print lm_model.getSequenceScore("cat")
    print lm_model.getSequenceScore("zzz")
    print lm_model.getSequenceScore("ion")
