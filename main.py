#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 16:21:02 2019
@author: fatimamh

FILE:
    SUMMARIZATION_PYTORCH
INPUT:
    CSV FILES (TRAIN/VAL/TEST)
OUTPUT:
    CSV FILES (TRAIN/VAL/TEST) - ADDED SYSTEM SUMMARIES
    ROUGE SCORE FILE
DESCRIPTION:
    THIS CODE TAKES CSV FILES (TRAIN/VAL/TEST) AND DOES THE FOLLOWING.
        -CREATE AN ENCODER-DECODER NETWORK
        -GENERATES SUMMARIES
        -CALCULATES ROUGE SCORES
"""
'''-----------------------------------------------------------------------
Import libraries
-----------------------------------------------------------------------'''
import argparse
import resource
import time
import os
import sys
from os import path
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
torch.backends.cudnn.deterministic = True

from lib.utils.file_utils import read_content
from lib.utils.memory_utils import get_size
from lib.utils.model_utils import model_param
from lib.model.s2s_model import S2SModel
from lib.model.encoder import Encoder
from lib.model.decoder import AttentionDecoder

from lib.training import train_model
from lib.inference import test_model
from lib.evaluation import rouge_evaluation
from rouge import Rouge

parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--t', type = bool,   default = False,   help = 'To train the model.')
parser.add_argument('--i', type = bool,   default = False,   help = 'To test the model on test data.')
parser.add_argument('--e', type = bool,   default = False,   help = 'To evaluate the model.')
parser.add_argument('--a', type = bool,   default = True,    help = 'To train, test and evaluate the model.')
parser.add_argument('--c', type = str,    default = '/hits/basement/nlp/fatimamh/s2s_summarization/configuration',   
                                                            help = 'Configuration file')
'''----------------------------------------------------------------
'''
if __name__ == "__main__":
    
    args       = parser.parse_args()     
    configure  = eval(read_content(args.c))
    print(type(configure), configure)
    is_trained = False
    is_tested  = False
    is_scored  = False

    bm_file    = os.path.join(configure['ckpt_folder'], configure["bm_f"])
    out_file   = os.path.join(configure['out_folder'], configure["out_f"])


    '''------------------------------------------------------------
    STEP 1: Prepare data tensors and language objects:
                As I have done processing on sample data and data 
                is saved, therefore I disabled the function calls. 
    ------------------------------------------------------------'''

    '''
    
    if path.exists(bm_file):
        is_trained = True

    if path.exists(out_file):
        is_tested = True

    
    if args.t or args.i or args.e:
        args.a = False

    prog_start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('\n----------------------Printing all arguments:--------------------\n{}\n---------------------------------------------------\n'.format(args)) 
    if args.a or args.t or args.i:

        torch.cuda.empty_cache()
        #Declare the hyperparameter
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        
        START_IDX  = 2 
        PAD_IDX    = 1 
        #print(PAD_IDX)
        #Declare the model
        encoder = 
        decoder = 
        encoder = EncoderRNN(input_size, emb_size, hidden_size).to(device)
    decoder = AttnDecoderRNN(emb_size, hidden_size, output_size, dropout=0.1).to(device)
        model = S2SModel(device, encoder, decoder)#.to(device) # it will make object of model including encoder + decoder  
        model = model.to(device)
        print('-------------Model layers-------------------------\n{}\
            \n-------------------------------------------------\n'.format(model))
        
        #Set model parameters
        optimizer, criterion, clip = model_param(model, configure, PAD_IDX)
        if args.a or args.t:  
            print('In Training')      
            is_trained = train_model(device, configure, model, criterion, optimizer, clip)    
        
        if args.a or args.i: 
            if is_trained:
                print('In Inference')
                is_tested = test_model(device, configure, model, criterion)
            else:
                print('Model is not trained.')
    
    if args.a or args.e: 
        if is_tested:
            print('In Evaluation')
            is_scored = rouge_evaluation(device, configure)
        else: 
            print('Summaries file \'{}\' is missing/ testing not completed.'.format(configure["out_f"]))
        if is_scored:
            print('Evaluation has been completed')

        # this gives size in kbs -- have to convert in bytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    memory = get_size(usage)

    print ('\n-------------------Memory and time usage:  {} in {}.--------------------\n'.format(memory, strftime("%H:%M:%S", gmtime(time.time() - prog_start_time))))

    '''

