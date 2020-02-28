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

from datetime import datetime
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import logging
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True # for fast training
# config file and memory utils
from lib.utils.file_utils import read_content
from lib.utils.memory_utils import get_size

from lib.utils.main_utils import data_processing

from lib.utils.main_utils import check_model
from lib.utils.main_utils import check_summaries
from lib.utils.main_utils import check_config_files

from lib.utils.model_utils import model_param

from lib.model.s2s_model import S2SModel

from lib.training import train_model
from lib.inference import test_model
from lib.evaluation import rouge_evaluation
from rouge import Rouge
import numpy as np

'''----------------------------------------------------------------
'''
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
log_format = "-----------------------------------------\n"\
             "%(asctime)s::%(levelname)s\n"\
             "%(filename)s::%(lineno)d\n%(message)s\n"\
             "-----------------------------------------\n"

logging.basicConfig(filename='/hits/basement/nlp/fatimamh/s2s_summarization/logs.log', level='DEBUG', format=log_format)
'''----------------------------------------------------------------
'''

parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--p', type = bool,   default = False,   help = 'To process data.')
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
    logging.info('args: {}'.format(args))   
    d_config   = eval(read_content(args.dc))
    m_config   = eval(read_content(args.mc))
    check_config_files(d_config, m_config) 
    
    '''------------------------------------------------------------
    Step 1: Prepare data tensors and language objects
    ------------------------------------------------------------'''
    if args.p:
        data_processing(d_config)
    '''------------------------------------------------------------
    Step 2: Model and its parameters
    ------------------------------------------------------------'''        
    is_trained = check_model(m_config)
    is_tested  = check_summaries(m_config)
    is_scored  = False

    if args.t or args.i or args.e:
        args.a = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('device: {}'.format(device))

    print('\n----------------------Printing all arguments:--------------------\n\
        {}\n---------------------------------------------------\n'.format(args)) 
    if args.a or args.t or args.i:
        '''------------------------------------------------------------
        Step 2: Declare the model
        ------------------------------------------------------------'''
        model = S2SModel(device, m_config)
        model = model.to(device)

        print('-------------Model layers-------------------------\n{}\
            \n-----------------------------------------------\n'.format(model))
        logging.info('Model layers: {}'.format(model))
        '''------------------------------------------------------------
        Step 3: Declare the hyperparameter
        ------------------------------------------------------------'''
        #optimizer, criterion, clip = model_param(model, m_config)
        enc_optimizer, dec_optimizer, criterion = model_param(model, m_config)

        if args.a or args.t:  
            print('In Training')      
            is_trained = train_model(device, m_config, model, criterion, enc_optimizer, dec_optimizer)    
        
        if args.a or args.i: 
            if is_trained:
                print('In Testing')
                is_tested = test_model(device, m_config, model, criterion)
            else:
                print('Model is not trained.')
    
        if args.a or args.e: 
            if is_tested:
                print('In Evaluation')
                is_scored = rouge_evaluation(device, m_config)
            else: 
                print('Summaries file \'{}\' is missing/ testing not completed.'.format(m_config["output_summaries_file"]))
            if is_scored:
                print('Evaluation has been completed')

        # this gives size in kbs -- have to convert in bytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    memory = get_size(usage)

    print ('\n-------------------Memory and time usage:  {}.--------------------\n'.format(memory))
