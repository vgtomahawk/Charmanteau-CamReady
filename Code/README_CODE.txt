Prerequisites:
The following libraries & requirements are prerequisites for running our code
- dynet version 2.0 (open source python NN framework with cpp backend) Link: http://dynet.readthedocs.io/en/latest/python.html
- scipy
- nltk
- editdistance (Cython-implemented library, to compute Levenshtein edit distance, can be installed on pip)
- tensorflow version 0.12 
- Atleast 7GB of RAM available (for dynet)

Instructions To Run the Code:
The main() function of our code is located in Code/barebones_enc_dec.py. The code can be run in various modes. We shall explain each of these modes here. The modes perform different experiments such as Cross-Validation, testing on D_BLIND etc. The code can be run as (starting from the root of the zip file):

cd Code
python barebones_enc_dec.py <mode> --dynet-mem 7000 --dynet-seed 786786

Modes:
- "KNIGHTTEST": Cross-validation on D_WIKI (401 examples). Replicates roughly the Experiment 1 Results
- "HOLDOUTTEST": Trains the model on D_WIKI and evaluates on D_BLIND (Held-out set, 1223 examples). 

   Output: Outputs the performance of both of our model and the BASELINE (Deri and Knight et al, 2015) on D_BLIND

Instructions To Query the Model for Arbitrary Words:
To query the trained model multiple times, follow the following steps.
- First train it in "HOLDOUTTEST". This will save the trained model(s) (ensemble) in Buffer/ directory.
- Refer to query.py for an idea of how to query the model (alternatively, view the following code snippet)
	import barebones_enc_dec as bed
	predictor=bed.getModel()
	answers=bed.query("<word1>","<word2>",predictor) #word1 and word2 are all-character, lowercase strings (no whitespaces, numbers, special characters)
  answers is the top-5 list of answers.
  Note that once the model is loaded, it can be queried multiple times.


Parameters:
Though the code works with the parameter values set in the code, in order to switch between Forward and Backward architectures etc, it is necessary to change some parameters in the code.

preTrain     - Pretrains the encoder of encoder-decoder models, set to False by default, suggested value for good performance
downstream   - Whether to use the downstream vector in the next character prediction softmax of the decoder, set to True by default, suggested value for good performance.
sharing      - Whether to share encoder and decoder character embeddings, set to False by default, suggested value for good performance.
GRU          - Whether to use LSTMs or GRU, set to False by default, suggested value for good performance
initFromFile - Whether to use the pretrained embeddings or not, set to True by default, suggested value for good performance
decodeMethod - "gen": Uses the SCORE decoding mechanism
             - "greedy": Uses the GREEDY decoding mechanism. Cannot be used when TRAIN_REVERSE==True
             - "beam": Uses the BEAM decoding mechanism. Cannot be used when TRAIN_REVERSE==True
TRAIN_REVERSE - Set to True for BACKWARD architecture, set to False for FORWARD architechture. Set to False by default. Generally, BACKWARD architechtures do better.
EMB_SIZE     - Character embedding size, 50 by default, suggested value for good performance. We have only included the vocabulary pre-trained embeddings for 50-dim embeddings, so if you set this to something else, ensure initFromFile is False
HIDDEN_SIZE  - Encoder and decoder hidden cell size. We found a value close to 100 works best, with optimal performance at 110 in most cases.
OPTIMIZER - "SGD", recommended. Can also use "ADAM", although we strongly advise against it. Given the small size of training data, ADAM may underperform.
UPDATE_ENC_EMBEDDINGS: Set to True if TRAIN_REVERSE=False, set to False if TRAIN_REVERSE=True
UPDATE_DEC_EMBEDDINGS: Set to False if TRAIN_REVERSE=True, set to True if TRAIN_REVERSE=False
ENSEMBLE_SIZE: 20-30 recommended. 1 is equivalent to having no ensembling. Note that decodeMethod=="greedy" and decodeMethod=="beam" will only use the first of the ensembled models, since we are ensembling at the score level.


Note that it is necessary to have the prerequisites installed and the directory structure maintained exactly as it is in the zip file.

