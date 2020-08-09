import torch
import random
from collections import Counter
import argparse
import torch.nn as nn # use plan embeding 
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.util import preprocess, create_context_target, convert_one_hot
from w2vec.CBow import CustomCBOW
from w2vec.SkipGram import CustomSkipGram
import pickle # for save

def word2vec_trainer(corpus, word2ind, mode="CBOW", dimension=64, learning_rate=0.01, iteration=50000, batch_size=500, window_size=3):
    vocab_size = len(word2ind)
    contexts, target = create_context_target(corpus, window_size)
    target = convert_one_hot(target, vocab_size).float()
    contexts = convert_one_hot(contexts, vocab_size).float()
    batch_size = min(batch_size, len(target))
    optimizer = SGD(lr=learning_rate)
    losses = []
    parallel_num = contexts.shape[1]
    print("Number of words : %d" % (len(target)))
    #################### model initialization ####################
    if mode == "CBOW":
        model = CustomCBOW(vocab_size, dimension, vocab_size, parallel_num)
    elif mode == "SG":
        model = CustomSkipGram(vocab_size, dimension, vocab_size, parallel_num)
    else:
        print("Unkwnown mode : "+mode)
        exit()
    ##############################################################
    for i in range(iteration):
        ################## getRandomContext ##################
        index = torch.randperm(len(target))[0:batch_size]
        centerWord, contextWords = target[index], contexts[index]
        ################## learning ##################    
        loss = model.forward(contextWords, centerWord)
        model.backward()
        optimizer.update(model)
        W_emb = model.get_inputw()
        W_out = model.get_outputw()
        losses.append(loss)
        ################## learning rate decay ##################
        lr = learning_rate*(1-i/iteration)
        optimizer.set_lr(lr)
        #########################################################
        if i%1000==0:
        	avg_loss=sum(losses)/len(losses)
        	print("Loss : %f" %(avg_loss,))
        	losses=[]

    return W_emb, W_out

def main():
    with open('text8', 'r') as f:
       text = f.read()
	# Write your code of data processing, training, and evaluation
	# Full training takes very long time. We recommend using a subset of text8 when you debug
    corpus, word2ind, _ = preprocess(text, subset=1e-4)
    print("processing completed")
    W_emb, W_out = word2vec_trainer(corpus, word2ind, mode="SG", learning_rate=0.01, iteration=50000, window_size=1)
    
    # plot (not sure, skipgram 보고 나중에 수정할게요)
    # trainer.plot()

    # saved
    # params = {}
    # params['word_vecs'] = W_emb.astype(np.float16)
    # params['word_out'] = W_out.astype(np.float16)
    # if mode == 'CBOW':
    #     pkl_file = 'cbow_params.pkl'
    # elif mode == 'SG':
    #     pkl_file = 'skipgram_params.pkl'

    # with open(pkl_file, 'wb') as f:
    #     pickle.dump(params, f, -1)
    
main()
