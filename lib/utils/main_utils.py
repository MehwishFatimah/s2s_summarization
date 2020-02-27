#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:48:29 2020

@author: fatimamh
"""
import resource
import time
import os
import sys
from lib.utils.memory_utils import get_size

# data processing utils
from lib.utils.data_utils import process_data
from lib.utils.dataset_utils import process_vacabs
from lib.utils.dataset_utils import process_tensors
from lib.utils.dataset_utils import object_load
'''----------------------------------------------------------------
'''

def data_processing(config):
    # Step 1: Convert data from json to csv. CLEAN AND SHORT 
    start_time = time.time()
    print("cleaning data")
    process_data(config)
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
        format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
    
    # Step 2: Generate vocabs
    start_time = time.time()
    print("processing data")
    in_vocab, out_vocab = process_vacabs(config)
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
        format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))

    # Step 2:Generate tensors
    start_time = time.time()
    print("processing tensors")
    process_tensors(config, in_vocab = in_vocab, out_vocab = out_vocab)
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
        format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))

'''----------------------------------------------------------------
'''
def load_vocabs(config):
    
    file = os.path.join(config["dict_folder"], config["text_dict_c"])
    input_vocab = object_load(file)
    print(input_vocab.n_words)
    file = os.path.join(config["dict_folder"], config["sum_dict_c"])
    output_vocab = object_load(file)
    print(output_vocab.n_words)

    return input_vocab, output_vocab

'''----------------------------------------------------------------
'''
def check_model(config):
    is_trained = False
    bm_file    = os.path.join(config["check_point_folder"], config["best_model_file"])
    if os.path.exists(bm_file):
        is_trained = True
    return is_trained

'''----------------------------------------------------------------
'''
def check_summaries(config):
    is_tested  = False
    out_file    = os.path.join(config["out_folder"], config["output_summaries_file"])
    if os.path.exists(out_file):
        is_tested = True
    return is_tested
'''----------------------------------------------------------------
'''
def check_config_files(d_config, m_config):
    for key in d_config:
        if key == 'in_folder' or 'json_files' or 'csv_files': continue
        assert d_config[key] == m_config[key],"< Mismatch in config files: KEY = {}>".format(key)