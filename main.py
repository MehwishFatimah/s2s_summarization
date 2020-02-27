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
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import logging
torch.backends.cudnn.deterministic = True

# config file and memory utils
from lib.utils.file_utils import read_content
from lib.utils.memory_utils import get_size

from lib.utils.main_utils import data_processing
from lib.utils.main_utils import load_vocabs
from lib.utils.main_utils import check_model
from lib.utils.main_utils import check_summaries
from lib.utils.main_utils import check_config_files

from lib.utils.model_utils import model_param

from lib.model.s2s_model import S2SModel

from lib.training import train_model
#from lib.inference import test_model
#from lib.evaluation import rouge_evaluation
from rouge import Rouge
import numpy as np
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--p', type = bool,   default = False,   help = 'To process data.')
parser.add_argument('--v', type = bool,   default = False,   help = 'To load vocabularies.')
parser.add_argument('--t', type = bool,   default = False,   help = 'To train the model.')
parser.add_argument('--i', type = bool,   default = False,   help = 'To test the model on test data.')
parser.add_argument('--e', type = bool,   default = False,   help = 'To evaluate the model.')
parser.add_argument('--a', type = bool,   default = True,    help = 'To train, test and evaluate the model.')
parser.add_argument('--dc', type = str,    default = '/hits/basement/nlp/fatimamh/s2s_summarization/data_config_en_sub',   
                                                            help = 'Configuration file')
parser.add_argument('--mc', type = str,    default = '/hits/basement/nlp/fatimamh/s2s_summarization/model_config_sub',   
                                                            help = 'Configuration file')


  

'''----------------------------------------------------------------
'''
if __name__ == "__main__":
    args       = parser.parse_args()
    #print(args)   
    d_config   = eval(read_content(args.dc))
    m_config   = eval(read_content(args.mc))
    check_config_files(d_config, m_config) 
    
    '''------------------------------------------------------------
    Step 1: Prepare data tensors and language objects
    ------------------------------------------------------------'''
    if args.p:
        data_processing(d_config)

    input_vocab = None 
    output_vocab = None
    if args.v:
        input_vocab, output_vocab = load_vocabs(m_config)
    
    '''------------------------------------------------------------
    Step 2: Model and its parameters
    ------------------------------------------------------------'''        
    is_trained = check_model(m_config)
    is_tested = check_summaries(m_config)
    is_scored  = False

    if args.t or args.i or args.e:
        args.a = False

    prog_start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('\n----------------------Printing all arguments:--------------------\n\
        {}\n---------------------------------------------------\n'.format(args)) 
    if args.a or args.t or args.i:
        print(device)
        '''------------------------------------------------------------
        Step 2: Declare the model
        ------------------------------------------------------------'''
        model = S2SModel(device, m_config)
        model = model.to(device)

        print('-------------Model layers-------------------------\n{}\
            \n-----------------------------------------------\n'.format(model))

        '''------------------------------------------------------------
        Step 3: Declare the hyperparameter
        ------------------------------------------------------------'''
        #optimizer, criterion, clip = model_param(model, m_config)
        enc_optimizer, dec_optimizer, criterion = model_param(model, m_config)
        #print(enc_optimizer)
        #print(dec_optimizer)
        #print(criterion)
        
        
        if args.a or args.t:  
            print('In Training')      
            is_trained = train_model(device, m_config, model, criterion, enc_optimizer, dec_optimizer)    
        '''
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

