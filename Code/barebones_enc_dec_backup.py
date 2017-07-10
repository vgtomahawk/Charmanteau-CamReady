import sys
import math
import random
import dynet as dy
import readData
from collections import defaultdict
import argparse
import numpy as np
import datetime
import nltk
import copy

#Config Definition
EMB_SIZE=50
LAYER_DEPTH=1
HIDDEN_SIZE=100
NUM_EPOCHS=10
STOP=0
SEPARATOR=1

def attend(encoder_outputs,state_factor_matrix):
    miniBatchLength=state_factor_matrix.npvalue().shape[1]
    encoderOutputLength=state_factor_matrix.npvalue().shape[0]
    hiddenSize=encoder_outputs[0].npvalue().shape[0]

    factor_Products=[state_factor_matrix[l] for l in range(encoderOutputLength)]
    factor_Products=dy.esum([dy.cmult(encoder_outputs[l],dy.concatenate([state_factor_matrix[l]]*hiddenSize)) for l in range(encoderOutputLength)])
    
    return factor_Products

def attend_vector(encoder_outputs,state_factor_vector):
    encoderOutputLength=state_factor_vector.npvalue().shape[0]
    hiddenSize=encoder_outputs[0].npvalue().shape[0]
    
    factor_Products=[dy.cmult(dy.concatenate([state_factor_vector[l]]*hiddenSize),encoder_outputs[l]) for l in range(encoderOutputLength)]
   
    factor_Products=dy.esum(factor_Products)
    return factor_Products

def topk(vector,k):
    topklist=[]
    while len(topklist)<k:
        top=np.argmax(vector)
        topklist.append(top)
        vector[top]=-np.inf

    return topklist

def beamDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,downstream=False,k=10):
    dy.renew_cg()
    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])

    sentence_de_forward=sentence_de
    sentence_de_reverse=sentence_de[::-1]

    s=encoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    s_reverse=revcoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
    states_reverse=s_reverse.add_inputs(inputs)
    revcoder_outputs=[s.output() for s in states_reverse]

    final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
    final_state=states[-1].s()
    final_state_reverse=states_reverse[-1].s()
    final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
    final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

    s_init=decoder.initial_state().set_s(final_state_reverse)
    o_init=s_init.output() 
    alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
    c_init=attend_vector(final_combined_outputs,alpha_init)

    
    s_0=s_init
    o_0=o_init
    alpha_0=alpha_init
    c_0=c_init
    
    finishedSequences=[]
    currentSequences=[(s_0,c_0,o_0,[],0.0),]

    #print "Beam Search Start"
    while len(finishedSequences)<2*k:
        candidates=[]
        for currentSequence in currentSequences:
            scores=None
            if downstream:
                scores=dy.affine_transform([bias,R,dy.concatenate([currentSequence[2],currentSequence[1]])])
            else:
                scores=dy.affine_transform([bias,R,currentSequence[2]])
            topkTokens=topk(scores.npvalue(),k)
            for topkToken in topkTokens:
                loss=(dy.pickneglogsoftmax(scores,topkToken)).value()
                candidate_i_t=dy.concatenate([dy.lookup(decoder_lookup,topkToken),currentSequence[1]])
                candidate_s_t=currentSequence[0].add_input(candidate_i_t)
                candidate_o_t=candidate_s_t.output()
                candidate_alpha_t=dy.softmax(dy.concatenate([dy.dot_product(candidate_o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
                candidate_c_t=attend_vector(final_combined_outputs,candidate_alpha_t)
                candidate_loss=currentSequence[4]+loss
                candidate_sequence=copy.deepcopy(currentSequence[3])
                candidate_sequence.append(topkToken)
                candidate=(candidate_s_t,candidate_c_t,candidate_o_t,candidate_sequence,candidate_loss)
                if topkToken==STOP or len(candidate_sequence)>len(sentence_de)+10:
                    if len(candidate_sequence)>3 or len(candidate_sequence)>=len(sentence_de):
                        finishedSequences.append(candidate)
                else:
                    candidates.append(candidate)
        #Sort candidates by loss, lesser loss is better
        candidates.sort(key = lambda x: x[4])
        currentSequences=candidates[:k]

    #print "Beam Search End"

    finishedSequences.sort(key = lambda x:x[4])
    sentence_en=finishedSequences[0][3]      

    return loss,sentence_en



def greedyDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,downstream=False,GRU=False):
    dy.renew_cg()
    total_words=len(sentence_en)
    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])

    sentence_de_forward=sentence_de
    sentence_de_reverse=sentence_de[::-1]

    s=encoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    s_reverse=revcoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
    states_reverse=s_reverse.add_inputs(inputs)
    revcoder_outputs=[s.output() for s in states_reverse]

    final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
    final_state=states[-1].s()
    final_state_reverse=states_reverse[-1].s()

    if GRU:
        final_coding_state=final_state_reverse+final_state
    else:
        final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
    final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

    s_init=decoder.initial_state().set_s(final_state_reverse)
    o_init=s_init.output() 
    alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
    c_init=attend_vector(final_combined_outputs,alpha_init)
    
    s_0=s_init
    o_0=o_init
    alpha_0=alpha_init
    c_0=c_init
    

    losses=[]
    currentToken=None
    englishSequence=[]

    while currentToken!=STOP and len(englishSequence)<len(sentence_de)+10:
        #Calculate loss and append to the losses array
        scores=None
        if downstream:
            scores=R*dy.concatenate([o_0,c_0])+bias
        else:
            scores=R*o_0+bias
        currentToken=np.argmax(scores.npvalue())
        loss=dy.pickneglogsoftmax(scores,currentToken)
        losses.append(loss)
        englishSequence.append(currentToken)

        #Take in input
        i_t=dy.concatenate([dy.lookup(decoder_lookup,currentToken),c_0])
        s_t=s_0.add_input(i_t)
        o_t=s_t.output()
        alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_t=attend_vector(final_combined_outputs,alpha_t)
        
        #Prepare for the next iteration
        s_0=s_t
        o_0=o_t
        c_0=c_t
        alpha_0=alpha_t

    total_loss=dy.esum(losses)
    return total_loss,englishSequence



def do_one_example(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_en,downstream=False,GRU=False):
    dy.renew_cg()
    total_words=len(sentence_en)
    encoder_lookup=encoder_params["lookup"]
    decoder_lookup=decoder_params["lookup"]
    R=dy.parameter(decoder_params["R"])
    bias=dy.parameter(decoder_params["bias"])

    sentence_de_forward=sentence_de
    sentence_de_reverse=sentence_de[::-1]

    s=encoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
    states=s.add_inputs(inputs)
    encoder_outputs=[s.output() for s in states]

    s_reverse=revcoder.initial_state()
    inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
    states_reverse=s_reverse.add_inputs(inputs)
    revcoder_outputs=[s.output() for s in states_reverse]

    final_coding_output=encoder_outputs[-1]+revcoder_outputs[-1]
    final_state=states[-1].s()
    final_state_reverse=states_reverse[-1].s()

    if GRU:
        final_coding_state=final_state_reverse+final_state
    else:
        final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
    final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

    s_init=decoder.initial_state().set_s(final_state_reverse)
    o_init=s_init.output() 
    alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
    c_init=attend_vector(final_combined_outputs,alpha_init)

    
    s_0=s_init
    o_0=o_init
    alpha_0=alpha_init
    c_0=c_init
    

    losses=[]
    
    for en in sentence_en:
        #Calculate loss and append to the losses array
        scores=None
        if downstream:
            scores=R*dy.concatenate([o_0,c_0])+bias
        else:
            scores=R*o_0+bias
        loss=dy.pickneglogsoftmax(scores,en)
        losses.append(loss)

        #Take in input
        i_t=dy.concatenate([dy.lookup(decoder_lookup,en),c_0])
        s_t=s_0.add_input(i_t)
        o_t=s_t.output()
        alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_t=attend_vector(final_combined_outputs,alpha_t)
        
        #Prepare for the next iteration
        s_0=s_t
        o_0=o_t
        c_0=c_t
        alpha_0=alpha_t

    total_loss=dy.esum(losses)
    return total_loss,total_words

class Config:
    READ_OPTION="NORMAL"
    downstream=True
    sharing=False
    GRU=False

    def __init__(self,READ_OPTION="NORMAL",downstream=True,sharing=False,GRU=False):
        self.READ_OPTION=READ_OPTION
        self.downstream=downstream
        self.sharing=sharing
        self.GRU=GRU


READ_OPTION="NORMAL"
downstream=True
sharing=False
GRU=False

config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU)


if config.READ_OPTION=="NORMAL":
    train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(trainingPoints=700,validPoints=400)
elif config.READ_OPTION=="KNIGHTHOLDOUT":
    train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getDataKnightHoldOut(trainingPoints=1000)

reverse_wids=readData.reverseDictionary(wids)

print len(train_sentences_de)
print len(train_sentences_en)

print len(valid_sentences_de)
print len(valid_sentences_en)


VOCAB_SIZE_DE=len(wids)
VOCAB_SIZE_EN=VOCAB_SIZE_DE


train_sentences=zip(train_sentences_de,train_sentences_en)
valid_sentences=zip(valid_sentences_de,valid_sentences_en)

#Specify model
model=dy.Model()

if config.GRU:
    encoder=dy.GRUBuilder(LAYER_DEPTH,EMB_SIZE,HIDDEN_SIZE,model)
    revcoder=dy.GRUBuilder(LAYER_DEPTH,EMB_SIZE,HIDDEN_SIZE,model)
    decoder=dy.GRUBuilder(LAYER_DEPTH,EMB_SIZE+HIDDEN_SIZE,HIDDEN_SIZE,model)
else:
    encoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE,HIDDEN_SIZE,model)
    revcoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE,HIDDEN_SIZE,model)
    decoder=dy.LSTMBuilder(LAYER_DEPTH,EMB_SIZE+HIDDEN_SIZE,HIDDEN_SIZE,model)

encoder_params={}
encoder_params["lookup"]=model.add_lookup_parameters((VOCAB_SIZE_DE,EMB_SIZE))

decoder_params={}
if config.sharing:
    decoder_params["lookup"]=encoder_params["lookup"]
else:
    decoder_params["lookup"]=model.add_lookup_parameters((VOCAB_SIZE_EN,EMB_SIZE))

if config.downstream:
    decoder_params["R"]=model.add_parameters((VOCAB_SIZE_EN,2*HIDDEN_SIZE))
else:
    decoder_params["R"]=model.add_parameters((VOCAB_SIZE_EN,HIDDEN_SIZE))

decoder_params["bias"]=model.add_parameters((VOCAB_SIZE_EN))

trainer=dy.SimpleSGDTrainer(model)

totalSentences=0
for epochId in xrange(NUM_EPOCHS):    
    random.shuffle(train_sentences)
    for sentenceId,sentence in enumerate(train_sentences):
        totalSentences+=1
        sentence_de=sentence[0]
        sentence_en=sentence[1]
        loss,words=do_one_example(model,encoder,revcoder,decoder,encoder_params,decoder_params,sentence_de,sentence_en,downstream=config.downstream,GRU=config.GRU)
        loss.value()
        loss.backward()
        trainer.update()
        if totalSentences%100==0:
            #random.shuffle(valid_sentences)
            perplexity=0.0
            totalLoss=0.0
            totalWords=0.0
            for valid_sentence in valid_sentences:
                valid_sentence_de=valid_sentence[0]
                valid_sentence_en=valid_sentence[1]
                validLoss,words=do_one_example(model,encoder,revcoder,decoder,encoder_params,decoder_params,valid_sentence_de,valid_sentence_en,downstream=config.downstream,GRU=config.GRU)
                totalLoss+=float(validLoss.value())
                totalWords+=words
            print totalLoss
            print totalWords
            perplexity=math.exp(totalLoss/totalWords)
            print "Validation perplexity after epoch:",epochId,"sentenceId:",sentenceId,"Perplexity:",perplexity,"Time:",datetime.datetime.now()             
    trainer.update_epoch(1.0)


originalWordFile=open("originalWords.txt","w")
outputWordFile=open("outputWords.txt","w")
import editdistance

exactMatches=0
editDistance=0.0


for validSentenceId,validSentence in enumerate(valid_sentences):
    valid_sentence_de=validSentence[0]
    valid_sentence_en=validSentence[1]
    validLoss,valid_sentence_en_hat=greedyDecode(model,encoder,revcoder,decoder,encoder_params,decoder_params,valid_sentence_de,downstream=config.downstream,GRU=config.GRU)

    originalWord="".join([reverse_wids[c] for c in valid_sentence_en[:-1]])
    outputWord="".join([reverse_wids[c] for c in valid_sentence_en_hat[:-1]])
    
    if originalWord==outputWord:
        exactMatches+=1

    editDistance+=editdistance.eval(originalWord,outputWord)

    print "Input Word Pair:,","".join([reverse_wids[c] for c in valid_sentence_de])
    print "Original Word:,",originalWord    
    print "Output Word:,",outputWord

    originalWordFile.write(originalWord+"\n")
    outputWordFile.write(outputWord+"\n")

totalWords=len(valid_sentences)

print "Total Words",totalWords
print "Exact Matches",exactMatches
print "Average Edit Distance",(editDistance+0.0)/(totalWords+0.0)


originalWordFile.close()
outputWordFile.close()

#train_batch(model,encoder,revcoder,decoder,encoder_params,decoder_params,train_sentences,valid_sentences,NUM_EPOCHS,"Models/"+"Attentional"+"_"+str(HIDDEN_SIZE)+"_"+"Uni")
