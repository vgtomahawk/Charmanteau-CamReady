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
import loadEmbeddingsFile
import utilities
import aligner
import scipy.stats

def fuseDicts(featureDictList):
    keyList=featureDictList[0].keys()
    featureDims=len(featureDictList)
    fusedDict={}
    for key in keyList:
        fusedDict[key]=np.zeros(featureDims)
        for i,dictionary in enumerate(featureDictList):
            fusedDict[key][i]=dictionary[key]
    return fusedDict

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





class Config:
    READ_OPTION="NORMAL"
    downstream=True
    sharing=False
    GRU=False

    def __init__(self,READ_OPTION="NORMAL",downstream=True,sharing=False,GRU=False,preTrain=False,initFromFile=False,initFileName=None,foldId=None,TRAIN_REVERSE=False,trainMethod="NORMAL",ENSEMBLE=False,ENSEMBLE_METHOD="SOFT",ATTENTION_OFF=False):
        self.READ_OPTION=READ_OPTION
        self.downstream=downstream
        self.sharing=sharing
        self.GRU=GRU
        self.preTrain=preTrain
        self.initFromFile=initFromFile
        self.initFileName=initFileName
        self.foldId=foldId
        self.TRAIN_REVERSE=TRAIN_REVERSE
        self.trainMethod=trainMethod  
        self.ENSEMBLE=ENSEMBLE
        self.ENSEMBLE_METHOD=ENSEMBLE_METHOD
        self.ATTENTION_OFF=ATTENTION_OFF

class HyperParams:
    EMB_SIZE=None
    LAYER_DEPTH=None
    HIDDEN_SIZE=None
    HIDDEN_SIZE_DEC=None
    NUM_EPOCHS=None
    STOP=None
    SEPARATOR=None
    
    def __init__(self,EMB_SIZE=50,LAYER_DEPTH=1,HIDDEN_SIZE=100,HIDDEN_SIZE_DEC=100,NUM_EPOCHS=10,STOP=0,SEPARATOR=1,dMethod="REVERSE",OPTIMIZER="SGD",UPDATE_ENC_EMBEDDINGS=True,UPDATE_DEC_EMBEDDINGS=True):
        #Hyperparameter Definition
        self.EMB_SIZE=EMB_SIZE
        self.LAYER_DEPTH=LAYER_DEPTH
        self.HIDDEN_SIZE=HIDDEN_SIZE
        self.HIDDEN_SIZE_DEC=HIDDEN_SIZE_DEC
        self.NUM_EPOCHS=NUM_EPOCHS
        self.STOP=STOP
        self.SEPARATOR=SEPARATOR
        self.dMethod=dMethod
        self.OPTIMIZER=OPTIMIZER
        self.UPDATE_ENC_EMBEDDINGS=UPDATE_ENC_EMBEDDINGS
        self.UPDATE_DEC_EMBEDDINGS=UPDATE_DEC_EMBEDDINGS

def equalSequence(x,y):
    if len(x)!=len(y):
        return False

    for i,alpha in enumerate(x):
        if y[i]!=alpha:
            return False

    return True

class Model:

    def beamDecode(self,sentence_de,k=1):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

        part1=sentence_de[:sentence_de.index(hyperParams.SEPARATOR)]+[hyperParams.STOP,]
        part2=sentence_de[sentence_de.index(hyperParams.SEPARATOR)+1:]
        part1Length=len(part1)
        part2Length=len(part2)

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

        if GRU:
            final_coding_state=final_state_reverse+final_state
        else:
            final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
        final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
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
        while len(finishedSequences)<k:
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
                    if topkToken==hyperParams.STOP or len(candidate_sequence)>len(sentence_de)+20:
                        if len(candidate_sequence)>2:
                            finishedSequences.append(candidate)
                    else:
                        candidates.append(candidate)
            #Sort candidates by loss, lesser loss is better
            candidates.sort(key = lambda x: x[4])
            currentSequences=candidates[:k]

        #print "Beam Search End"

        prunedFinishedSequences=[]
        
        #Remove output sequences identical to parents. If that leaves nothing, fall back onto the original
        for finishedSequence in finishedSequences:
            if equalSequence(finishedSequence[3],part1) or equalSequence(finishedSequence[3],part2):
                continue
            else:
                prunedFinishedSequences.append(finishedSequence)

        if len(prunedFinishedSequences)!=0:
            finishedSequences=prunedFinishedSequences

        finishedSequences.sort(key = lambda x:x[4])
        sentence_en=finishedSequences[0][3]      
        loss_en=finishedSequences[0][4]

        return loss_en,sentence_en

    def greedyDecode(self,sentence_de):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

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

        if GRU:
            final_coding_state=final_state_reverse+final_state
        else:
            final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
        final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
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

        while currentToken!=hyperParams.STOP and len(englishSequence)<len(sentence_de)+10:
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



    def greedyDecode(self,sentence_de):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

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

        if GRU:
            final_coding_state=final_state_reverse+final_state
        else:
            final_coding_state=((final_state_reverse[0]+final_state[0]),(final_state_reverse[1]+final_state[1]))
        final_combined_outputs=[revcoder_output+encoder_output for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
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

        while currentToken!=hyperParams.STOP and len(englishSequence)<len(sentence_de)+10:
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


    def do_one_example_pretrain(self,sentence_de):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        encoder_params=self.encoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU

        dy.renew_cg()
        encoder_lookup=encoder_params["lookup"]
        R_DE=dy.parameter(encoder_params["R_DE"])
        bias_DE=dy.parameter(encoder_params["bias_DE"])

        sentence_de_forward=sentence_de
        sentence_de_reverse=sentence_de[::-1]

        s=encoder.initial_state()
        s=s.add_input(dy.lookup(encoder_lookup,self.hyperParams.SEPARATOR))
        
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_forward]
        states=[s,]+s.add_inputs(inputs)
        states=states[:-1]
        forward_scores=[]
        for s in states:
            o_0=s.output()
            forward_score=(R_DE)*(o_0)+(bias_DE)
            forward_scores.append(forward_score)

        forward_losses=[dy.pickneglogsoftmax(forward_scores[i],de) for i,de in enumerate(sentence_de_forward)]


        s_reverse=revcoder.initial_state()
        s_reverse=s_reverse.add_input(dy.lookup(encoder_lookup,self.hyperParams.SEPARATOR))
        inputs=[dy.lookup(encoder_lookup,de) for de in sentence_de_reverse]
        states_reverse=[s_reverse,]+s_reverse.add_inputs(inputs)
        states=states_reverse[:-1]
        backward_scores=[]
        for s in states_reverse:
            o_0=s.output()
            backward_score=(R_DE)*o_0+(bias_DE)
            backward_scores.append(backward_score)

        backward_losses=[dy.pickneglogsoftmax(backward_scores[i],de) for i,de in enumerate(sentence_de_reverse)]

        forward_loss=dy.esum(forward_losses)
        backward_loss=dy.esum(backward_losses)
        losses=[forward_loss,]+[backward_loss,]
        random.shuffle(losses)
        
        return losses

    def learnValPerceptron(self):
        self.w1=1.0
        self.w2=0.0
        self.topK=2

        #for valid_sentence in self.valid_sentences[:20]:
        #    sentence_de=valid_sentence[0]
        #    sentence_en=valid_sentence[1]
        #    part1=sentence_de[:sentence_de.index(self.hyperParams.SEPARATOR)]
        #    part2=sentence_de[sentence_de.index(self.hyperParams.SEPARATOR)+1:-1]
        #    part1=[self.reverse_wids[c] for c in part1]
        #    part2=[self.reverse_wids[c] for c in part2]
        #    part1=''.join(part1)
        #    part2=''.join(part2)
        #    parentSet=set()
        #    parentSet.add(part1)
        #    parentSet.add(part2)
        #    candidates=list(utilities.generateCandidates(part1,part2)-parentSet)
        #    candidates=[[self.wids[c] for c in candidate]+[self.hyperParams.STOP,] for candidate in candidates]
        #    prunedCandidates=[]
        #    for candidate in candidates:
        #        if len(candidate)>4:
        #            prunedCandidates.append(candidate)
        #    candidates=prunedCandidates
        #    losses=[]
        #    for candidate in candidates:
        #        loss,words=self.do_one_example(sentence_de,candidate)
        #        loss=np.sum(loss.npvalue())
        #        losses.append(loss)
        #    candidateLosses=zip(candidates,losses)
        #    candidateLosses.sort(key = lambda x: x[1])
        #    candidateLosses=candidateLosses[:self.topK]
        #    newCandidateLosses=[]
        #    for x in candidateLosses:
        #        loss,words=self.externalModel.do_one_example(sentence_de,x[0])
        #        loss=np.sum(loss.npvalue())
        #        newLoss=(x[0],x[1],loss,self.w1*x[1]+self.w2*loss)
        #        newCandidateLosses.append(newLoss)
        #    candidateLosses=newCandidateLosses
        #    candidateLosses.sort(key = lambda x:x[3])
        #    goldKey=sentence_en
        #    goldKeyFound=False
        #    goldKeyLoss=None
        #
        #    for x in candidateLosses:
        #         if x[0]==goldKey:
        #            goldKeyFound=True
        #            goldKeyLoss=x[3]
        #            break
        #
        #    if goldKeyFound and candidateLosses[0][0]!=goldKey:
        #        f1_y=
        #        f2_y=
        #        f1_yhat=
        #        f2_yhat=

    def genDecode(self,sentence_de,useBaseline=False,mixed=False,mode=None,topK=True,outFile=None,CANDIDATE_WRITE=True):
        if mixed==True:
            featureDict1=self.genDecode(sentence_de,mode="c")
            if topK==False:
                featureDict2=self.genDecode(valid_sentence_de,mode="c",useBaseline=True)
            else:
                keyValList=[(key,value) for key,value in featureDict1.items()]
                keyValList.sort(key = lambda x:x[1])
                keyValList=keyValList[:10]
                featureDict1={}
                for x in keyValList:
                    featureDict1[x[0]]=x[1]
                featureDict2={}
                for key in featureDict1:
                    featureDict2[key]=-self.lm_model.getSequenceScore(key)

            featureDictList=[featureDict1,featureDict2]
            featureDict=fuseDicts(featureDictList)
            maxKey=None
            maxScore=-float("inf")
            for key,value in featureDict.items():
                newScore=np.dot(self.weightVector,value)
                if newScore>maxScore:
                    maxScore=newScore
                    maxKey=key
            print maxKey
            charList=list(maxKey)
            charList=[self.wids[c] for c in charList]
            charList=charList+[self.hyperParams.STOP,]
            return 0.0,charList

        part1=sentence_de[:sentence_de.index(self.hyperParams.SEPARATOR)]
        part2=sentence_de[sentence_de.index(self.hyperParams.SEPARATOR)+1:-1]
        part1=[self.reverse_wids[c] for c in part1]
        part2=[self.reverse_wids[c] for c in part2]
        part1=''.join(part1)
        part2=''.join(part2)
        parentSet=set()
        parentSet.add(part1)
        parentSet.add(part2)
        candidates=list(utilities.generateCandidates(part1,part2)-parentSet)
        if useBaseline==False:
            candidates=[[self.wids[c] for c in candidate]+[self.hyperParams.STOP,] for candidate in candidates]
            prunedCandidates=[]
            for candidate in candidates:
                if len(candidate)>4:
                    prunedCandidates.append(candidate)
            candidates=prunedCandidates
            losses=[]
            for candidate in candidates:
                loss=None
                if self.config.trainMethod=="KEEPGEN" or self.config.trainMethod=="KEEPDELETE":
                    a=aligner.findAlignment(sentence_de,candidate,epsilon=self.hyperParams.SEPARATOR,SEPARATOR=self.hyperParams.SEPARATOR,mode=self.config.trainMethod,KEEP=2)
                    loss,words=self.do_one_example(sentence_de,a)
                    loss=np.sum(loss.npvalue())
                elif self.config.ENSEMBLE and self.config.ENSEMBLE_METHOD=="SOFT" and self.ensembleList!=None:
                    netLoss=0.0
                    for ensembleModel in self.ensembleList:
                        loss,words=ensembleModel.do_one_example(sentence_de,candidate)
                        netLoss+=np.sum(loss.npvalue())
                    netLoss/=len(self.ensembleList)
                    loss=netLoss
                elif self.config.ENSEMBLE and self.config.ENSEMBLE_METHOD=="PROBSPACE" and self.ensembleList!=None:
                    netLoss=0.0
                    for ensembleModel in self.ensembleList:
                        loss,words=ensembleModel.do_one_example(sentence_de,candidate)
                        netLoss+=math.exp(-np.sum(loss.npvalue()))
                    netLoss/=len(self.ensembleList)
                    netLoss=-math.log(netLoss)
                    loss=netLoss
                elif self.config.ENSEMBLE and self.config.ENSEMBLE_METHOD=="HSPACE" and self.ensembleList!=None:
                    netLoss=0.0
                    for ensembleModel in self.ensembleList:
                        loss,words=ensembleModel.do_one_example(sentence_de,candidate)
                        netLoss+=math.exp(np.sum(loss.npvalue()))
                    netLoss=math.log(len(self.ensembleList))-math.log(netLoss)
                    netLoss=-netLoss
                    loss=netLoss 
                else:
                    loss,words=self.do_one_example(sentence_de,candidate)
                    loss=np.sum(loss.npvalue())
                losses.append(loss)
            candidateLosses=zip(candidates,losses)
            
            if self.config.TRAIN_REVERSE:
                newCandidateLosses=[]
                Z_L=0.0
                for candidateLoss in candidateLosses:
                    candidate=candidateLoss[0]
                    Z_L+=self.probL.pdf(len(candidate)/len(sentence_de))

                for candidateLoss in candidateLosses:
                    candidate=candidateLoss[0]
                    loss=candidateLoss[1]
                    textCandidate=''.join([self.reverse_wids[c] for c in candidate[:-1]])
                    lmLoss=-self.lm_model.getSequenceScore(textCandidate)
                    lengthLoss=-math.log(self.probL.pdf(len(candidate)/len(sentence_de)))+math.log(Z_L)
                    newLoss=(candidate,loss+lmLoss+lengthLoss)
                    newCandidateLosses.append(newLoss)
                candidateLosses=newCandidateLosses
            """
            greedyLoss,greedySequence=self.greedyDecode(sentence_de)
            greedyLoss=np.sum(greedyLoss.npvalue())
            candidateLosses.append((greedySequence,greedyLoss))
            """
            candidateLosses.sort(key= lambda x:x[1])
            if self.externalModel!=None:
                candidateLosses=candidateLosses[:2]
                newCandidateLosses=[]
                for candidateLoss in candidateLosses:
                    loss,words=self.externalModel.do_one_example(sentence_de,candidateLoss[0])
                    loss=np.sum(loss.npvalue())
                    newCandidateLosses.append((candidateLoss[0],self.w*loss+candidateLoss[1]))
                candidateLosses=newCandidateLosses
                candidateLosses.sort(key= lambda x:x[1])

            Z=0.0
            for x in candidateLosses:
                Z=Z+math.exp(-x[1])
            if outFile!=None:
                if CANDIDATE_WRITE:
                    for candidateLoss in candidateLosses:
                        candidateWord=''.join([self.reverse_wids[c] for c in candidateLoss[0][:-1]])
                        candidateValue=candidateLoss[1]+math.log(Z)
                        outFile.write(candidateWord+":"+str(candidateValue)+"\n")
                else:
                    candidateWord=candidateLosses[0][0]
                    #candidateValue=candidateLoss[1]+math.log(Z)
                    part1=sentence_de[:sentence_de.index(self.hyperParams.SEPARATOR)]
                    part2=sentence_de[sentence_de.index(self.hyperParams.SEPARATOR)+1:-1]
                    part1=[self.reverse_wids[c] for c in part1]
                    part2=[self.reverse_wids[c] for c in part2]
                    part1=''.join(part1)
                    part2=''.join(part2)
                    outFile.write(part1+" "+part2+" "+self.blindCache[(part1,part2)]+"\n") 
            #print [self.reverse_wids[c] for c in candidateLosses[0][0]]
            #exit()
            if mode==None:
                return candidateLosses[0][1],candidateLosses[0][0]
            else:
                featureDict={}
                for candidateLoss in candidateLosses:
                    candidateWord=''.join([self.reverse_wids[c] for c in candidateLoss[0][:-1]])
                    candidateValue=candidateLoss[1]
                    featureDict[candidateWord]=candidateValue
                return featureDict

        else:
            prunedCandidates=[]
            for candidate in candidates:
                if len(candidate)>3:
                    prunedCandidates.append(candidate)
            candidates=prunedCandidates
            losses=[]
            for candidate in candidates:
                loss=-self.lm_model.getSequenceScore(candidate)
                losses.append(loss)
            candidateLosses=zip(candidates,losses)
            candidateLosses.sort(key=lambda x:x[1])

            #print candidateLosses[0][0]
            bestCandidate=[self.wids[c] for c in candidateLosses[0][0]]+[self.hyperParams.STOP,]
            bestCandidateLoss=candidateLosses[0][1]

            if mode==None:
                return bestCandidateLoss,bestCandidate
            else:
                featureDict={}
                for candidateLoss in candidateLosses:
                    candidateWord=candidateLoss[0]
                    candidateValue=candidateLoss[1]
                    featureDict[candidateWord]=candidateValue
                return featureDict




    def do_one_example(self,sentence_de,sentence_en,verbose=False):
        model=self.model
        encoder=self.encoder
        revcoder=self.revcoder
        decoder=self.decoder
        encoder_params=self.encoder_params
        decoder_params=self.decoder_params
        downstream=self.config.downstream
        GRU=self.config.GRU
        hyperParams=self.hyperParams

        if self.config.TRAIN_REVERSE:
            #Swap
            x=sentence_de
            sentence_de=sentence_en
            sentence_en=x

        dy.renew_cg()
        total_words=len(sentence_en)
        encoder_lookup=encoder_params["lookup"]
        decoder_lookup=decoder_params["lookup"]
        if self.config.trainMethod=="KEEPDELETE":
            decoder_action_lookup=decoder_params["action_lookup"]
        R=dy.parameter(decoder_params["R"])
        bias=dy.parameter(decoder_params["bias"])

        sentence_de_forward=sentence_de
        sentence_de_reverse=sentence_de[::-1]

        s=encoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de,update=hyperParams.UPDATE_ENC_EMBEDDINGS) for de in sentence_de_forward]
        #if hyperParams.FREEZE_EMBEDDINGS:
        #    inputs=[dy.lookup(encoder_lookup,de,update=False) for de in sentence_de_forward]
        states=s.add_inputs(inputs)
        encoder_outputs=[s.output() for s in states]

        s_reverse=revcoder.initial_state()
        inputs=[dy.lookup(encoder_lookup,de,update=hyperParams.UPDATE_ENC_EMBEDDINGS) for de in sentence_de_reverse]
        #if hyperParams.FREEZE_EMBEDDINGS:
        #    inputs=[dy.lookup(encoder_lookup,de,update=False) for de in sentence_de_reverse]
    
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

        if self.config.trainMethod=="KEEPDELETE":
            losses=[]
            final_combined_outputs=[dy.concatenate([revcoder_output,encoder_output]) for revcoder_output,encoder_output in zip(revcoder_outputs[::-1],encoder_outputs)]
            s=decoder.initial_state()
            s=s.add_input(dy.lookup(decoder_lookup,self.hyperParams.SEPARATOR,update=hyperParams.UPDATE_DEC_EMBEDDINGS))
            
            for enIndex,en in enumerate(sentence_en):
                scores=R*dy.concatenate([final_combined_outputs[enIndex],s.output()])+bias
                loss=dy.pickneglogsoftmax(scores,en)
                if en!=self.hyperParams.SEPARATOR:
                    s=s.add_input(dy.lookup(encoder_lookup,sentence_de[enIndex],update=hyperParams.UPDATE_ENC_EMBEDDINGS))
                losses.append(loss)
            total_loss=dy.esum(losses)
            return total_loss,total_words

        initDict={}
        initDict["FORWARD"]=final_state
        initDict["REVERSE"]=final_state_reverse
        initDict["BI"]=final_coding_state

        if self.config.ATTENTION_OFF:
            s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
            #s_init=decoder.initial_state().set_s(final_coding_state)
            o_init=s_init.output()
            s_0=s_init
            o_0=o_init
            losses=[]
            for enIndex,en in enumerate(sentence_en):
                #Calculate loss and append to the losses array
                scores=R*o_0+bias
                loss=dy.pickneglogsoftmax(scores,en)
                losses.append(loss)

                #Take in input
                i_t=None
                i_t=dy.lookup(decoder_lookup,en,update=hyperParams.UPDATE_DEC_EMBEDDINGS)
                s_t=s_0.add_input(i_t)
                o_t=s_t.output()
               
                #Prepare for the next iteration
                s_0=s_t
                o_0=o_t
                

            total_loss=dy.esum(losses)
            return total_loss,total_words

           

        s_init=decoder.initial_state().set_s(initDict[hyperParams.dMethod])
        #s_init=decoder.initial_state().set_s(final_coding_state)
        o_init=s_init.output() 
        alpha_init=dy.softmax(dy.concatenate([dy.dot_product(o_init,final_combined_output) for final_combined_output in final_combined_outputs]))
        c_init=attend_vector(final_combined_outputs,alpha_init)

        
        s_0=s_init
        o_0=o_init
        alpha_0=alpha_init
        c_0=c_init
        

        losses=[]

        if verbose:
            attentionDumpFile=open("outputDir4/attentionDump.txt","w")

        for enIndex,en in enumerate(sentence_en):
            if verbose:
                attentionDumpFile.write(" ".join([str(x) for x in alpha_0.npvalue()])+"\n")
            #Calculate loss and append to the losses array
            scores=None
            if downstream:
                scores=R*dy.concatenate([o_0,c_0])+bias
            else:
                scores=R*o_0+bias
            loss=dy.pickneglogsoftmax(scores,en)
            losses.append(loss)

            #Take in input
            i_t=None
            if self.config.trainMethod=="KEEPDELETE":
                i_t=dy.concatenate([dy.lookup(decoder_lookup,sentence_de[enIndex],update=hyperParams.UPDATE_DEC_EMBEDDINGS),dy.lookup(decoder_action_lookup,en),c_0])
            else:
                i_t=dy.concatenate([dy.lookup(decoder_lookup,en,update=hyperParams.UPDATE_DEC_EMBEDDINGS),c_0])
            s_t=s_0.add_input(i_t)
            o_t=s_t.output()
            alpha_t=dy.softmax(dy.concatenate([dy.dot_product(o_t,final_combined_output) for final_combined_output in final_combined_outputs]))
            c_t=attend_vector(final_combined_outputs,alpha_t)
            
            #Prepare for the next iteration
            s_0=s_t
            o_0=o_t
            c_0=c_t
            alpha_0=alpha_t

        if verbose:
            attentionDumpFile.close()

        total_loss=dy.esum(losses)
        return total_loss,total_words

    def save_model(self):
        print "Saving Model"
        if self.config.trainMethod=="KEEPDELETE":
            self.model.save(self.modelFile,[self.encoder,self.revcoder,self.decoder,self.encoder_params["lookup"],self.decoder_params["lookup"],self.decoder_params["action_lookup"],self.decoder_params["R"],self.decoder_params["bias"]])
        else:
            self.model.save(self.modelFile,[self.encoder,self.revcoder,self.decoder,self.encoder_params["lookup"],self.decoder_params["lookup"],self.decoder_params["R"],self.decoder_params["bias"]])
        print "Model Saved"

    def load_model(self):
        print "Loading Model"
        if self.config.trainMethod=="KEEPDELETE":
            (self.encoder,self.revcoder,self.decoder,self.encoder_params["lookup"],self.decoder_params["lookup"],self.decoder_params["action_lookup"],self.decoder_params["R"],self.decoder_params["bias"])=self.model.load(self.modelFile)
        else:
            (self.encoder,self.revcoder,self.decoder,self.encoder_params["lookup"],self.decoder_params["lookup"],self.decoder_params["R"],self.decoder_params["bias"])=self.model.load(self.modelFile)
        print "Model Loaded"

    def __init__(self,config,hyperParams,modelFile="Buffer/model"):
        self.config=config
        self.hyperParams=hyperParams
        self.modelFile=modelFile
        self.bestPerplexity=float("inf")
        self.weightVector=np.zeros(2)
        self.weightVector[0]=-1.0
        self.weightVector[1]=0.0
        self.externalModel=None
        if self.config.READ_OPTION=="NORMALCROSSVALIDATE":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(filterKnight=False,crossValidate=True,foldId=config.foldId)
 
        elif self.config.READ_OPTION=="KNIGHTCROSSVALIDATE":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(filterKnight=True,crossValidate=True,foldId=config.foldId)
        elif self.config.READ_OPTION=="NORMAL":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(trainingPoints=700,validPoints=400)
        elif self.config.READ_OPTION=="KNIGHTONLY":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getData(trainingPoints=320,validPoints=40,filterKnight=True) 
        elif self.config.READ_OPTION=="NORMALDISJOINT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getDataDisjoint(trainingPoints=500,validPoints=200)
        elif self.config.READ_OPTION=="KNIGHTHOLDOUT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids=readData.getDataKnightHoldOut(trainingPoints=1000)
        elif self.config.READ_OPTION=="PHONETICINPUT":
            train_sentences_de,train_sentences_en,valid_sentences_de,valid_sentences_en,test_sentences_de,test_sentences_en,wids,wids_phonetic=readData.getDataPhoneticInput(trainingPoints=700,validPoints=400)       

        self.train_sentences_de=train_sentences_de
        self.train_sentences_en=train_sentences_en
        self.valid_sentences_de=valid_sentences_de
        self.valid_sentences_en=valid_sentences_en
        self.test_sentences_de=test_sentences_de
        self.test_sentences_en=test_sentences_en
        
        
        self.wids=wids
        self.reverse_wids=readData.reverseDictionary(self.wids)
        if self.config.READ_OPTION=="PHONETICINPUT" or self.config.READ_OPTION=="PHONOLEXINPUT":
            self.wids_phonetic=wids_phonetic
            self.reverse_wids_phonetic=readData.reverseDictionary(self.wids_phonetic)

        print len(self.train_sentences_de)
        print len(self.train_sentences_en)

        print len(self.valid_sentences_de)
        print len(self.valid_sentences_en)

        if self.config.READ_OPTION=="PHONETICINPUT":
            self.VOCAB_SIZE_DE=len(self.wids_phonetic)
            self.VOCAB_SIZE_EN=len(self.wids)
        else:
            self.VOCAB_SIZE_DE=len(wids)
            self.VOCAB_SIZE_EN=self.VOCAB_SIZE_DE

        self.train_sentences=zip(self.train_sentences_de,self.train_sentences_en)
        self.valid_sentences=zip(self.valid_sentences_de,self.valid_sentences_en)
        self.test_sentences=zip(self.test_sentences_de,self.test_sentences_en)


        #Specify model
        self.model=dy.Model()

        config=self.config
        hyperParams=self.hyperParams
        model=self.model

        if self.config.GRU:
            self.encoder=dy.GRUBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.revcoder=dy.GRUBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.decoder=dy.GRUBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE+hyperParams.HIDDEN_SIZE,hyperParams.HIDDEN_SIZE,model)
        else:
            self.encoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.revcoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)
            self.decoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE+hyperParams.HIDDEN_SIZE,hyperParams.HIDDEN_SIZE,model)
            if self.config.trainMethod=="KEEPDELETE":
                #self.decoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE+10+hyperParams.HIDDEN_SIZE,hyperParams.HIDDEN_SIZE,model)
                self.decoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)


        if self.config.initFromFile:
            if self.config.READ_OPTION=="PHONETICINPUT":
                print "Cannot initialize phonetic embeddings from character embeddings"
                exit()
            preEmbeddings=loadEmbeddingsFile.loadEmbeddings(wids,self.config.initFileName,hyperParams.EMB_SIZE)
            
        #if self.config.trainMethod=="KEEPDELETE":
        #    VOCAB_SIZE_EN=3

        self.encoder_params={}
        self.encoder_params["lookup"]=model.add_lookup_parameters((self.VOCAB_SIZE_DE,hyperParams.EMB_SIZE))
        
        if self.config.initFromFile:
            self.encoder_params["lookup"].init_from_array(preEmbeddings)

        self.decoder_params={}
        if config.sharing:
            self.decoder_params["lookup"]=self.encoder_params["lookup"]
        else:
            self.decoder_params["lookup"]=model.add_lookup_parameters((self.VOCAB_SIZE_EN,hyperParams.EMB_SIZE))

        if self.config.trainMethod=="KEEPDELETE":
            self.decoder_params["action_lookup"]=model.add_lookup_parameters((3,10))

        if self.config.initFromFile:
            self.decoder_params["lookup"].init_from_array(preEmbeddings)

        if config.downstream:
            self.decoder_params["R"]=model.add_parameters((self.VOCAB_SIZE_EN,2*hyperParams.HIDDEN_SIZE))
            if self.config.trainMethod=="KEEPDELETE":
                self.decoder_params["R"]=model.add_parameters((3,3*hyperParams.HIDDEN_SIZE))
 
        else:
            self.decoder_params["R"]=model.add_parameters((self.VOCAB_SIZE_EN,hyperParams.HIDDEN_SIZE))
            if self.config.trainMethod=="KEEPDELETE":
                self.decoder_params["R"]=model.add_parameters((3,hyperParams.HIDDEN_SIZE))
     

        self.decoder_params["bias"]=model.add_parameters((self.VOCAB_SIZE_EN))
        if self.config.trainMethod=="KEEPDELETE":
            self.decoder_params["bias"]=model.add_parameters((3))


        if config.preTrain:
            self.encoder_params["R_DE"]=model.add_parameters((self.VOCAB_SIZE_DE,hyperParams.HIDDEN_SIZE))
            self.encoder_params["bias_DE"]=model.add_parameters((self.VOCAB_SIZE_DE))

        if self.config.ATTENTION_OFF:
            self.decoder_params["R"]=model.add_parameters((self.VOCAB_SIZE_EN,hyperParams.HIDDEN_SIZE))
            self.decoder=dy.LSTMBuilder(hyperParams.LAYER_DEPTH,hyperParams.EMB_SIZE,hyperParams.HIDDEN_SIZE,model)


        import example
        self.lm_model=example.lm_model

        print "Initializing Blind Cache"
        blindCache={}
        for line in open("../Data/dataAliyaScraped.csv"):
            words=line.split()
            #if (words[0],words[1]) in blindCache:
            #    print "Repeat"
            #    print words
            blindCache[(words[0],words[1])]=words[2]
        print "Initialized Blind Cache"
        print "Size of Blind Cache:",len(blindCache)
        self.blindCache=blindCache    

    def readBlindData(self):
        print "Reading Blind Data"
        trainInputs,trainOutputs,validInputs,validOutputs,testInputs,testOutputs,wids=readData.getData(removeKnight=True)
        blind_sentences_de=trainInputs+validInputs+testInputs
        blind_sentences_en=trainOutputs+validOutputs+testOutputs
        self.blind_sentences=zip(blind_sentences_de,blind_sentences_en)
        print "Finished Reading Blind Data"
        print "Size of Blind Data:",len(self.blind_sentences)

    def pretrain(self):
        trainer=dy.SimpleSGDTrainer(self.model)
        totalSentences=0
        for epochId in xrange(hyperParams.NUM_EPOCHS):    
            random.shuffle(self.train_sentences)
            for sentenceId,sentence in enumerate(self.train_sentences):
                totalSentences+=1
                sentence_de=sentence[0]
                losses=self.do_one_example_pretrain(sentence_de)
                for loss in losses:
                    loss.value()
                    loss.backward()
                    trainer.update()
            trainer.update_epoch(1.0)

    def trainLengthModel(self):
        lengths=[]
        for train_sentence in self.train_sentences:
            sentence_de=train_sentence[0]
            sentence_en=train_sentence[1]
            lr=(len(sentence_en)+0.0)/(len(sentence_de)+0.0)
            lengths.append(lr)
        muL=sum(lengths)/len(lengths)
        varianceL=sum([(lr-muL)*(lr-muL) for lr in lengths])/len(lengths)
        sigmaL=math.sqrt(varianceL)
        print "Mu",muL
        print "Sigma",sigmaL
        self.muL=muL
        self.sigmaL=sigmaL
        self.probL=scipy.stats.norm(muL,sigmaL)

    def train(self,interEpochPrinting=False):
        if self.config.preTrain:
            print "Pretraining"
            self.pretrain()

        if self.config.trainMethod=="KEEPGEN" or self.config.trainMethod=="KEEPDELETE":
            print "Trimming Training Set"
            aux_train_sentences=[]
            aux_valid_sentences=[]
            goodInstances=0
            badInstances=0
            for train_sentence in self.train_sentences:
                sentence_de=train_sentence[0]
                sentence_en=train_sentence[1]
                a=aligner.findAlignment(sentence_de,sentence_en,epsilon=self.hyperParams.SEPARATOR,SEPARATOR=self.hyperParams.SEPARATOR,mode=self.config.trainMethod,KEEP=2)
                if a==[]:
                    badInstances+=0
                else:
                    if len(sentence_de)!=len(a):
                        print "Assertion Failed" 
                    aux_train_sentences.append((sentence_de,a))
            self.train_sentences=aux_train_sentences

            for train_sentence in self.valid_sentences:
                sentence_de=train_sentence[0]
                sentence_en=train_sentence[1]
                a=aligner.findAlignment(sentence_de,sentence_en,epsilon=self.hyperParams.SEPARATOR,SEPARATOR=self.hyperParams.SEPARATOR,mode=self.config.trainMethod,KEEP=2)
                if a==[]:
                    badInstances+=0
                    badInstances+=1
                    #print [self.reverse_wids[c] for c in sentence_de]
                    #print [self.reverse_wids[c] for c in sentence_en]
                else:
                    if len(sentence_de)!=len(a):
                        print "Assertion Failed"
                    goodInstances+=1
                    aux_valid_sentences.append((sentence_de,a))
            self.train_sentences=aux_train_sentences
            self.aux_valid_sentences=aux_valid_sentences

            print "Good Instances",goodInstances
            print "Bad Instances",badInstances
            #exit()
            #self.aux_valid_sentences=[]

        if self.hyperParams.OPTIMIZER=="Adam":
            trainer=dy.AdamTrainer(self.model)
        else:
            trainer=dy.SimpleSGDTrainer(self.model)

        totalSentences=0
        for epochId in xrange(self.hyperParams.NUM_EPOCHS):    
            random.shuffle(self.train_sentences)
            for sentenceId,sentence in enumerate(self.train_sentences):
                totalSentences+=1
                sentence_de=sentence[0]
                sentence_en=sentence[1]
                loss,words=self.do_one_example(sentence_de,sentence_en)
                loss.value()
                loss.backward()
                trainer.update()
                if interEpochPrinting:
                    if totalSentences%100==0:
                        #random.shuffle(valid_sentences)
                        perplexity=0.0
                        totalLoss=0.0
                        totalWords=0.0
                        valid_sentences=None
                        if self.config.trainMethod=="KEEPGEN" or self.config.trainMethod=="KEEPDELETE":
                            valid_sentences=self.aux_valid_sentences
                        else:
                            valid_sentences=self.valid_sentences
                        for valid_sentence in valid_sentences:
                            valid_sentence_de=valid_sentence[0]
                            valid_sentence_en=valid_sentence[1]
                            validLoss,words=self.do_one_example(valid_sentence_de,valid_sentence_en)
                            totalLoss+=float(validLoss.value())
                            totalWords+=words
                        print totalLoss
                        print totalWords
                        perplexity=math.exp(totalLoss/totalWords)
                        print "Validation perplexity after epoch:",epochId,"sentenceId:",sentenceId,"Perplexity:",perplexity,"Time:",datetime.datetime.now()             
            trainer.update_epoch(1.0)
            perplexity=0.0
            totalLoss=0.0
            totalWords=0.0
            if self.config.trainMethod=="KEEPGEN" or self.config.trainMethod=="KEEPDELETE":
                valid_sentences=self.aux_valid_sentences
            else:
                valid_sentences=self.valid_sentences
            for valid_sentence in valid_sentences:
                valid_sentence_de=valid_sentence[0]
                valid_sentence_en=valid_sentence[1]
                validLoss,words=self.do_one_example(valid_sentence_de,valid_sentence_en)
                totalLoss+=float(validLoss.value())
                totalWords+=words
            perplexity=math.exp(totalLoss/totalWords)
            if epochId>=5 and perplexity<self.bestPerplexity:
                self.save_model()
                self.bestPerplexity=perplexity
                print "Best Perplexity:",self.bestPerplexity

        #self.save_model()
        #self.load_model()

    def stripStops(self,sentence):
        if sentence[-1]==self.hyperParams.STOP:
            return sentence[:-1]
        else:
            return sentence

    def testKnight(self,blind_sentences):
        reverse_wids=self.reverse_wids
        import editdistance
        exactMatches=0
        editDistance=0.0
        for blindSentenceId,blindSentence in enumerate(blind_sentences):
            blind_sentence_de=blindSentence[0]
            blind_sentence_en=blindSentence[1]

            part1=blind_sentence_de[:blind_sentence_de.index(self.hyperParams.SEPARATOR)]
            part2=blind_sentence_de[blind_sentence_de.index(self.hyperParams.SEPARATOR)+1:-1]

            part1="".join([self.reverse_wids[c] for c in part1])
            part2="".join([self.reverse_wids[c] for c in part2])

            
            knightWord=self.blindCache[(part1,part2)]
            originalWord="".join([reverse_wids[c] for c in blind_sentence_en[:-1]])
            if originalWord==knightWord:
                exactMatches+=1

            editDistance+=editdistance.eval(originalWord,knightWord)

        totalWords=len(blind_sentences)

        print "Total Words",totalWords
        print "Exact Matches",exactMatches
        print "Average Edit Distance",(editDistance+0.0)/(totalWords+0.0)

        return exactMatches,(editDistance+0.0)/(totalWords+0.0)

        


    def testOut(self,valid_sentences,verbose=True,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="greedy",beam_decode_k=1,outFileName=None,CANDIDATE_WRITE=True):
        reverse_wids=self.reverse_wids

        if self.config.READ_OPTION=="PHONETICINPUT" or self.config.READ_OPTION=="PHONOLEXINPUT":
            reverse_wids_phonetic=self.reverse_wids_phonetic

        originalWordFile=open(originalFileName,"w")
        outputWordFile=open(outputFileName,"w")
        if outFileName!=None:
            outFile=open(outFileName,"w")
        else:
            outFile=None
        import editdistance
        exactMatches=0
        editDistance=0.0
        

        for validSentenceId,validSentence in enumerate(valid_sentences):
            valid_sentence_de=validSentence[0]
            valid_sentence_en=validSentence[1]
            if decodeMethod=="greedy":
                validLoss,valid_sentence_en_hat=self.greedyDecode(valid_sentence_de)
            elif decodeMethod=="beam":
                validLoss,valid_sentence_en_hat=self.beamDecode(valid_sentence_de,k=beam_decode_k)
            elif decodeMethod=="gen":
                validLoss,valid_sentence_en_hat=self.genDecode(valid_sentence_de,outFile=outFile,CANDIDATE_WRITE=CANDIDATE_WRITE)
            elif decodeMethod=="genMixed":
                validLoss,valid_sentence_en_hat=self.genDecode(valid_sentence_de,mixed=True)
            elif decodeMethod=="genBase":
                validLoss,valid_sentence_en_hat=self.genDecode(valid_sentence_de,useBaseline=True)
 
            #valid_sentence_en_stripped=self.stripStops(valid_sentence_en)
            #valid_sentence_en_hat_stripped=self.stripStops(valid_sentence_en_hat)

            originalWord="".join([reverse_wids[c] for c in valid_sentence_en[:-1]])
            outputWord="".join([reverse_wids[c] for c in valid_sentence_en_hat[:-1]])
            
            if outFile!=None:
                if decodeMethod=="greedy":
                    part1=valid_sentence_de[:valid_sentence_de.index(self.hyperParams.SEPARATOR)]
                    part2=valid_sentence_de[valid_sentence_de.index(self.hyperParams.SEPARATOR)+1:-1]
                    part1=[self.reverse_wids[c] for c in part1]
                    part2=[self.reverse_wids[c] for c in part2]
                    part1=''.join(part1)
                    part2=''.join(part2)
                    outFile.write(part1+" "+part2+" "+self.blindCache[(part1,part2)]+"\n") 
                outFile.write("Original Word:"+originalWord+"\n")
                outFile.write("Output Word:"+outputWord+"\n")
                outFile.write("End of Example--------------------------------------"+"\n")
            
            if originalWord==outputWord:
                exactMatches+=1

            editDistance+=editdistance.eval(originalWord,outputWord)
            
            if verbose:
                if self.config.READ_OPTION=="PHONETICINPUT" or self.config.READ_OPTION=="PHONOLEXINPUT":
                    print "Input Word Pair:,","".join([reverse_wids_phonetic[c] for c in valid_sentence_de])
                else:
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

        if outFile!=None:
            outFile.close()

        return exactMatches,(editDistance+0.0)/(totalWords+0.0)

def query(queryWord1,queryWord2):
    random.seed(491)
    READ_OPTION="KNIGHTCROSSVALIDATE"
    preTrain=False
    downstream=True
    sharing=False
    GRU=False
    initFromFile=True
    initFileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"
    dMethod="REVERSE"
    decodeMethod="gen"
    TRAIN_REVERSE=True
    bothWays=False
    trainMethod="NORMAL"
    EMB_SIZE=50
    HIDDEN_SIZE=110
    OPTIMIZER="SGD"
    UPDATE_ENC_EMBEDDINGS=False
    UPDATE_DEC_EMBEDDINGS=True
    ENSEMBLE=True
    ENSEMBLE_METHOD="SOFT"
    RANDOM_SHUFFLE=True
    ENSEMBLE_SIZE=30
    QUERY=True

    if not ENSEMBLE:
        foldId=8
        config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId,TRAIN_REVERSE=TRAIN_REVERSE,trainMethod=trainMethod)
        hyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod,HIDDEN_SIZE=HIDDEN_SIZE,EMB_SIZE=EMB_SIZE,OPTIMIZER=OPTIMIZER,UPDATE_ENC_EMBEDDINGS=UPDATE_ENC_EMBEDDINGS,UPDATE_DEC_EMBEDDINGS=UPDATE_DEC_EMBEDDINGS)
        predictor=Model(config,hyperParams)
        predictor.train_sentences=predictor.train_sentences+predictor.test_sentences[:20]
        predictor.valid_sentences=predictor.valid_sentences+predictor.test_sentences[20:]
        predictor.train(interEpochPrinting=False)
        predictor.trainLengthModel()
        predictor.readBlindData()
        predictor.load_model()
    else:
        ensemble=[]
        for ensembleId in range(ENSEMBLE_SIZE):
            foldId=ensembleId%10
            print "Training fold ",foldId
            config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId,TRAIN_REVERSE=TRAIN_REVERSE,trainMethod=trainMethod,ENSEMBLE=ENSEMBLE,ENSEMBLE_METHOD=ENSEMBLE_METHOD)
            hyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod,HIDDEN_SIZE=HIDDEN_SIZE,EMB_SIZE=EMB_SIZE,OPTIMIZER=OPTIMIZER,UPDATE_ENC_EMBEDDINGS=UPDATE_ENC_EMBEDDINGS,UPDATE_DEC_EMBEDDINGS=UPDATE_DEC_EMBEDDINGS)
            predictor=Model(config,hyperParams,modelFile="Buffer/model_"+str(ensembleId))
            predictor.train_sentences=predictor.train_sentences+predictor.test_sentences[:20]
            predictor.valid_sentences=predictor.valid_sentences+predictor.test_sentences[20:]
            if RANDOM_SHUFFLE:
                predictor.train_sentences=predictor.train_sentences+predictor.valid_sentences
                random.shuffle(predictor.train_sentences)
                predictor.valid_sentences=predictor.train_sentences[341:]
                predictor.train_sentences=predictor.train_sentences[:341]
            if not QUERY: 
                predictor.train(interEpochPrinting=False)
            predictor.trainLengthModel()
            predictor.load_model()
            ensemble.append(predictor)

        predictor=ensemble[0]
        predictor.ensembleList=ensemble
        predictor.readBlindData()


    #queryWord1="bad"           
    #queryWord2="advice"
    queryWord1=[c for c in queryWord1]
    queryWord2=[c for c in queryWord2]
    queryWord=queryWord1+["SEPARATOR",]+queryWord2+["STOP",]
    print queryWord
    queryWord=[predictor.wids[c] for c in queryWord]
    print queryWord

    loss,answer=predictor.genDecode(queryWord,outFile=open("blah.txt","w"),CANDIDATE_WRITE=False)
    answer=[predictor.reverse_wids[c] for c in answer][:-1]
    return "".join(answer)


if __name__=="__main__":

    if sys.argv[1]=="PLAIN":
        random.seed(491)
        np.random.seed(6000)
        READ_OPTION="KNIGHTONLY"
        preTrain=False
        downstream=True
        sharing=False
        GRU=False
        initFromFile=True
        initFileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"

        config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName)
        hyperParams=HyperParams(NUM_EPOCHS=10)

        predictor=Model(config,hyperParams)
        predictor.train(interEpochPrinting=False)

        print "Greedy Decode"
        print "Validation Performance"
        predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")
        print "Test Performance"
        predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")

        
        #print "Beam Decode"
        #predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=5)
        """
        #predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=2)
        """

        predictor.load_model()

        print "Greedy Decode"
        print "Validation Performance"
        predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")
        #print "Test Performance"
        predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt")

        """
        print "Beam Decode"
        predictor.testOut(predictor.valid_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=5)
        #predictor.testOut(predictor.test_sentences,verbose=False,originalFileName="originalWords.txt",outputFileName="outputWords.txt",decodeMethod="beam",beam_decode_k=2)
        """

    elif sys.argv[1]=="HOLDOUTTEST":
        random.seed(491)
        READ_OPTION="KNIGHTCROSSVALIDATE"
        preTrain=False
        downstream=True
        sharing=False
        GRU=False
        initFromFile=True
        initFileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"
        dMethod="REVERSE"
        decodeMethod="gen"
        TRAIN_REVERSE=True
        bothWays=False
        trainMethod="NORMAL"
        EMB_SIZE=50
        HIDDEN_SIZE=110
        OPTIMIZER="SGD"
        UPDATE_ENC_EMBEDDINGS=False
        UPDATE_DEC_EMBEDDINGS=True
        ENSEMBLE=True
        ENSEMBLE_METHOD="SOFT"
        RANDOM_SHUFFLE=True
        ENSEMBLE_SIZE=30
        QUERY=True

        if not ENSEMBLE:
            foldId=8
            config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId,TRAIN_REVERSE=TRAIN_REVERSE,trainMethod=trainMethod)
            hyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod,HIDDEN_SIZE=HIDDEN_SIZE,EMB_SIZE=EMB_SIZE,OPTIMIZER=OPTIMIZER,UPDATE_ENC_EMBEDDINGS=UPDATE_ENC_EMBEDDINGS,UPDATE_DEC_EMBEDDINGS=UPDATE_DEC_EMBEDDINGS)
            predictor=Model(config,hyperParams)
            predictor.train_sentences=predictor.train_sentences+predictor.test_sentences[:20]
            predictor.valid_sentences=predictor.valid_sentences+predictor.test_sentences[20:]
            predictor.train(interEpochPrinting=False)
            predictor.trainLengthModel()
            predictor.readBlindData()
            predictor.load_model()
        else:
            ensemble=[]
            for ensembleId in range(ENSEMBLE_SIZE):
                foldId=ensembleId%10
                print "Training fold ",foldId
                config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId,TRAIN_REVERSE=TRAIN_REVERSE,trainMethod=trainMethod,ENSEMBLE=ENSEMBLE,ENSEMBLE_METHOD=ENSEMBLE_METHOD)
                hyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod,HIDDEN_SIZE=HIDDEN_SIZE,EMB_SIZE=EMB_SIZE,OPTIMIZER=OPTIMIZER,UPDATE_ENC_EMBEDDINGS=UPDATE_ENC_EMBEDDINGS,UPDATE_DEC_EMBEDDINGS=UPDATE_DEC_EMBEDDINGS)
                predictor=Model(config,hyperParams,modelFile="Buffer/model_"+str(ensembleId))
                predictor.train_sentences=predictor.train_sentences+predictor.test_sentences[:20]
                predictor.valid_sentences=predictor.valid_sentences+predictor.test_sentences[20:]
                if RANDOM_SHUFFLE:
                    predictor.train_sentences=predictor.train_sentences+predictor.valid_sentences
                    random.shuffle(predictor.train_sentences)
                    predictor.valid_sentences=predictor.train_sentences[341:]
                    predictor.train_sentences=predictor.train_sentences[:341]
                if not QUERY: 
                    predictor.train(interEpochPrinting=False)
                predictor.trainLengthModel()
                predictor.load_model()
                ensemble.append(predictor)

            predictor=ensemble[0]
            predictor.ensembleList=ensemble
            predictor.readBlindData()

    
        queryWord1="bad"           
        queryWord2="advice"
        queryWord1=[c for c in queryWord1]
        queryWord2=[c for c in queryWord2]
        queryWord=queryWord1+["SEPARATOR",]+queryWord2+["STOP",]
        print queryWord
        queryWord=[predictor.wids[c] for c in queryWord]
        print queryWord

        loss,answer=predictor.genDecode(queryWord,outFile=open("blah.txt","w"),CANDIDATE_WRITE=False)
        answer=[predictor.reverse_wids[c] for c in answer][:-1]
        print answer

        print "Blind Performance Best"
        blindMatches,blindDistance=predictor.testOut(predictor.blind_sentences,verbose=True,decodeMethod=decodeMethod,outFileName="output_reviews/"+"best"+"_"+"blind_forward_greedy.txt",CANDIDATE_WRITE=False)

        print "Blind Performance Knight"
        blindMatches,blindDistance=predictor.testKnight(predictor.blind_sentences)

    elif sys.argv[1]=="KNIGHTTEST":
        random.seed(491)
        READ_OPTION="KNIGHTCROSSVALIDATE"
        preTrain=False
        downstream=True
        sharing=False
        GRU=False
        initFromFile=True
        initFileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"
        dMethod="REVERSE"
        decodeMethod="gen"
        TRAIN_REVERSE=False
        bothWays=False
        trainMethod="NORMAL"
        EMB_SIZE=50
        HIDDEN_SIZE=100
        OPTIMIZER="SGD"
        UPDATE_ENC_EMBEDDINGS=True
        UPDATE_DEC_EMBEDDINGS=False
        ENSEMBLE=True
        ENSEMBLE_METHOD="SOFT"
        ATTENTION_OFF=False
        ENSEMBLE_SIZE=1

        averageValidMatches=0.0
        averageValidDistance=0.0
        averageTestMatches=0.0
        averageTestDistance=0
        for foldId in range(10):
            print "Training fold ",foldId
            ensemble=[]
            for ensembleId in range(ENSEMBLE_SIZE):
                print "Training ensemble ",ensembleId
                config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId,TRAIN_REVERSE=TRAIN_REVERSE,trainMethod=trainMethod,ENSEMBLE=ENSEMBLE,ENSEMBLE_METHOD=ENSEMBLE_METHOD,ATTENTION_OFF=ATTENTION_OFF)
                hyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod,HIDDEN_SIZE=HIDDEN_SIZE,EMB_SIZE=EMB_SIZE,OPTIMIZER=OPTIMIZER,UPDATE_ENC_EMBEDDINGS=UPDATE_ENC_EMBEDDINGS,UPDATE_DEC_EMBEDDINGS=UPDATE_DEC_EMBEDDINGS)
                predictor=Model(config,hyperParams)
                predictor.test_sentences=predictor.valid_sentences+predictor.test_sentences
                random.shuffle(predictor.train_sentences)
                predictor.valid_sentences=predictor.train_sentences[281:]
                predictor.train_sentences=predictor.train_sentences[:281]
                predictor.train(interEpochPrinting=False)
                predictor.trainLengthModel()
                predictor.load_model()
                ensemble.append(predictor)

            predictor=ensemble[0]
            predictor.ensembleList=ensemble
        
            print "Validation Performance Best"
            validMatches,validDistance=predictor.testOut(predictor.test_sentences[:40],beam_decode_k=2,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"best"+"_"+str(foldId)+"_"+"validation.txt")
            print "Test Performance Best"
            testMatches,testDistance=predictor.testOut(predictor.test_sentences[40:],beam_decode_k=2,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"best"+"_"+str(foldId)+"_"+"test.txt")
            
            averageValidMatches+=validMatches
            averageValidDistance+=validDistance
            averageTestMatches+=testMatches
            averageTestDistance+=testDistance

        print "---------Final Aggregates---------------"
        print "Average Valid Matches",averageValidMatches/10
        print "Average Test Matches",averageTestMatches/10
        print "Average Valid Distance",averageValidDistance/10
        print "Average Test Distance",averageTestDistance/10


               

    elif sys.argv[1]=="CROSSVALTRAIN":
        random.seed(491)

        READ_OPTION="KNIGHTCROSSVALIDATE"
        preTrain=False
        downstream=True
        sharing=False
        GRU=False
        initFromFile=True
        initFileName="../Pretrained/output_embeddings_134iter_lowestValLoss.txt"
        dMethod="REVERSE"
        decodeMethod="gen"
        TRAIN_REVERSE=False
        bothWays=False
        trainMethod="NORMAL"
        EMB_SIZE=50
        HIDDEN_SIZE=110
        OPTIMIZER="SGD"
        UPDATE_ENC_EMBEDDINGS=True
        UPDATE_DEC_EMBEDDINGS=False

        averageValidMatches=0.0
        averageTestMatches=0.0
        averageValidDistance=0.0
        averageTestDistance=0.0

        averageValidMatchesBest=0.0
        averageTestMatchesBest=0.0
        averageValidDistanceBest=0.0
        averageTestDistanceBest=0.0



        for foldId in range(10):
            config=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId,TRAIN_REVERSE=TRAIN_REVERSE,trainMethod=trainMethod)
            hyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod,HIDDEN_SIZE=HIDDEN_SIZE,EMB_SIZE=EMB_SIZE,OPTIMIZER=OPTIMIZER,UPDATE_ENC_EMBEDDINGS=UPDATE_ENC_EMBEDDINGS,UPDATE_DEC_EMBEDDINGS=UPDATE_DEC_EMBEDDINGS)

            predictor=Model(config,hyperParams)
            predictor.train(interEpochPrinting=False)
            predictor.trainLengthModel()
            predictor.readBlindData()
            if bothWays:
                extConfig=Config(READ_OPTION=READ_OPTION,downstream=downstream,sharing=sharing,GRU=GRU,preTrain=preTrain,initFromFile=initFromFile,initFileName=initFileName,foldId=foldId,TRAIN_REVERSE=False)
                extHyperParams=HyperParams(NUM_EPOCHS=10,dMethod=dMethod,HIDDEN_SIZE=HIDDEN_SIZE,EMB_SIZE=EMB_SIZE,OPTIMIZER=OPTIMIZER,UPDATE_ENC_EMBEDDINGS=True,UPDATE_DEC_EMBEDDINGS=False)
                extPredictor=Model(extConfig,extHyperParams,modelFile="Buffer/extModel")
                extPredictor.train(interEpochPrinting=False)
                predictor.externalModel=extPredictor 
            if decodeMethod=="genMixed":
                predictor.learnValPerceptron()

            print "Greedy Decode"
            print "Validation Performance"
            validMatches,validDistance=predictor.testOut(predictor.valid_sentences,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"final"+"_"+str(foldId)+"_"+"validation.txt")
            print "Test Performance"
            testMatches,testDistance=predictor.testOut(predictor.test_sentences,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"final"+"_"+str(foldId)+"_"+"test.txt")
            #print "Blind Performance Final"
            #blindMatches,blindDistance=predictor.testOut(predictor.blind_sentences,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"final"+"_"+str(foldId)+"_"+"blind.txt")



            averageValidMatches+=validMatches
            averageValidDistance+=validDistance
            averageTestMatches+=testMatches
            averageTestDistance+=testDistance
            
            print "Loading Best Decoder Model w.r.t Val Perplexity"
            predictor.load_model()
            if bothWays:
                predictor.externalModel.load_model()
            if decodeMethod=="genMixed":
                predictor.learnValPerceptron()

            print "Validation Performance Best"
            validMatches,validDistance=predictor.testOut(predictor.valid_sentences,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"best"+"_"+str(foldId)+"_"+"validation.txt")
            print "Test Performance Best"
            testMatches,testDistance=predictor.testOut(predictor.test_sentences,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"best"+"_"+str(foldId)+"_"+"test.txt")

            averageValidMatchesBest+=validMatches
            averageValidDistanceBest+=validDistance
            averageTestMatchesBest+=testMatches
            averageTestDistanceBest+=testDistance
            
            #print "Blind Performance Best"
            #blindMatches,blindDistance=predictor.testOut(predictor.blind_sentences,verbose=False,decodeMethod=decodeMethod,outFileName="outputDir4/"+"best"+"_"+str(foldId)+"_"+"blind.txt")

            #print "Blind Performance Knight"
            #blindMatches,blindDistance=predictor.testKnight(predictor.blind_sentences)
            
            #predictor.do_one_example(predictor.test_sentences[7][0],predictor.test_sentences[7][1],verbose=True)
            #break

        print "---------Final Aggregates---------------"
        print "Average Valid Matches",averageValidMatches/10
        print "Average Test Matches",averageTestMatches/10
        print "Average Valid Distance",averageValidDistance/10
        print "Average Test Distance",averageTestDistance/10

        
        print "Average Valid Matches Best",averageValidMatchesBest/10
        print "Average Test Matches Best",averageTestMatchesBest/10
        print "Average Valid Distance Best",averageValidDistanceBest/10
        print "Average Test Distance Best",averageTestDistanceBest/10
