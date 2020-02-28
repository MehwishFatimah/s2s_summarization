#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:28:08 2019
Modified on Wed Nov 6
Modified on Wed Feb 12
@author: fatimamh
"""

'''-----------------------------------------------------------------------
Import libraries and defining command line arguments
-----------------------------------------------------------------------'''
import os
import argparse
import time
import resource
import random
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import torch
import pickle
import dill

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_index = 0
UNK_index = 1
SP_index  = 2
EP_index  = 3   


# For freq
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "UNK", 2: "SP", 3: "EP"} # add UNK
        self.n_words = 4  # Count PAD, UNK, SP and EP
        
    '''---------------------------------------------'''
    def add_text(self, text):
        for word in text.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    '''---------------------------------------------'''
    def reset(self):
        self.word2index.clear()
        self.index2word.clear()
        self.word2count.clear()
        self.index2word = {0: "PAD", 1: "UNK", 2: "SP", 3: "EP"} # add UNK
        self.n_words = 4

    def refill(self, text):
        for word in text:
            self.add_word(word)
    
    '''---------------------------------------------'''
    def filter_most_common(self, ratio):
        sorted_list = Counter(OrderedDict(sorted(self.word2count.items(), key=lambda t: t[1], reverse=True)))
        print(len(sorted_list))
        sorted_list = sorted_list.most_common(ratio)
        sorted_list =  [i[0] for i in sorted_list]
        return sorted_list   
            
    def condensed_vocab(self, ratio):
        new_list = self.filter_most_common(ratio)
        self.reset()
        self.refill(new_list)    
    
    '''---------------------------------------------'''
    # Remove words below a certain threshold
    def filter_least_common(self, min_count):
        keep_words = list()
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)
        return keep_words 

    def trimmed_vocab(self, min_count):
        new_list = self.filter_least_common(min_count)
        self.reset()
        self.refill(new_list)     
        
        
'''
====================================================================
'''

def prepare_vocabs(input_vocab, output_vocab, df):
    source = df['text']
    target = df['summary']
    print(len(source), len(target))

    for i in range(len(source)):
        input_vocab.add_text(source[i])
        words = source[i].split()
        #print('Source: {} {}'.format(words, len(words)))
        #print('Vocab: {} {}'.format(input_vocab.word2index, input_vocab.n_words))
        #print()
        output_vocab.add_text(target[i])    
        words = target[i].split()
        #print('target: {} {}'.format(words, len(words)))
        #print('Vocab: {} {}'.format(output_vocab.word2index, output_vocab.n_words))

    #return input_vocab, output_vocab

'''--------------------------------------------------------
'''            
def object_save(name, obj, folder):
    
    f_name = name #+ '.pkl'
    file = os.path.join(folder, f_name)
    with open(file, 'wb') as f:
        dill.dump(obj, f) #, pickle.HIGHEST_PROTOCOL)
    return file
'''---------------------------------------------------------------'''

def object_load(file):#f_name, folder):
    #f_name = f_name + '.pkl'
    #file = os.path.join(folder, f_name)
    with open(file, 'rb') as f:
        obj = dill.load(f)
    return obj
'''---------------------------------------------------------------'''

def process_vacabs(config):
    #print(files)
    files = config["csv_files"] 
    in_folder = config["in_folder"] 
    dict_folder = config["dict_folder"]
    text_vocab = Lang('text')
    sum_vocab = Lang('sum')
        
    for file in files:
        
        file  = os.path.join(in_folder, file)
        df = pd.read_csv(file, encoding = 'utf-8')
        #df = df.head(5)
        #print('\n=========================================')
        print('Training data:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
        print('text_vocab: {}, sum_vocab: {}'.format(text_vocab.n_words, sum_vocab.n_words)) 
        print()
        prepare_vocabs(text_vocab, sum_vocab, df)
        print('text_vocab: {}, sum_vocab: {}'.format(text_vocab.n_words, sum_vocab.n_words)) 
        print('--------------------')
    
    # Store original full dictionaries
    print('text_vocab: {}, sum_vocab: {}'.format(text_vocab.n_words, sum_vocab.n_words)) 
    f_input = object_save(config["text_dict_f"], text_vocab, dict_folder)
    f_output = object_save(config["sum_dict_f"], sum_vocab, dict_folder)
    #print(f_input, f_output, dict_folder)
    #print(text_vocab.__dict__)
    #print(sum_vocab.__dict__)
    # Make condense dictionaries
    text_vocab.condensed_vocab(config["text_vocab"])
    sum_vocab.condensed_vocab(config["sum_vocab"])
    f_input = object_save(config["text_dict_c"], text_vocab, dict_folder)
    f_output = object_save(config["sum_dict_c"], sum_vocab, dict_folder)
    print()
    print('text_vocab: {}, sum_vocab: {}'.format(text_vocab.n_words, sum_vocab.n_words))
    # send condensed vocab, because these will use for tensors
    return f_input, f_output

'''
====================================================================
'''
# to do ... padding in loader
def padding(data, max_len):  
    # padding to max_len
    len_data = min([max_len, len(data)])
    padded_data = np.pad(data, (0,max(0, max_len-len_data)), 'constant', constant_values = (PAD_index))[:max_len]
    return padded_data

'''---------------------------------------------------------------'''    
# modified for handling UNK
def vectorize(vocab, text):
    vector = list()
    for word in text.split():
        if word in vocab.word2index:
            #print('word: {}, index: {}'.format(word, vocab.word2index[word]))
            vector.append(vocab.word2index[word])
        else:
            #print('word: {}, UNK_index: {}'.format(word, UNK_index))
            vector.append(UNK_index)
    #print(vector)
    return vector 


'''---------------------------------------------------------------'''
# TO DO: WHY actual vocab gives unk
def tensor_to_text(vocab, vector):
    
    #print(vocab.n_words)
    words = list()
    for i in range(len(vector)):
        idx = vector[i]
        if idx == PAD_index: continue
        words.append(vocab.index2word[idx])
    print(vector)
    words = ' '.join(map(str, words)) 
    #print(words)
    return words 


'''---------------------------------------------------------------'''
def text_to_tensor(vocab, text, max_len):
    #print(lang.name)
    indexes = vectorize(vocab, text)
    indexes.append(EP_index)
    #print(len(indexes))
    indexes = padding(indexes, max_len)
    #print(len(indexes))
    #print('vectorize: {}'.format(indexes))
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
'''---------------------------------------------------------------'''

def prepare_tensors(input_vocab, output_vocab, df, folder, text_len, summary_len):

    source = df['text']
    target = df['summary']
        
    for i in range(len(source)):
        input_tensor = text_to_tensor(input_vocab, source[i], text_len)
        #print('input_tensor: {}'.format(input_tensor.shape))
        f_name = 'input_' + str(i+1) + '.pt'
        file = os.path.join(folder, f_name)
        torch.save(input_tensor, file)

        target_tensor = text_to_tensor(output_vocab, target[i], summary_len)
        #print('target_tensor: {}'.format(target_tensor.shape))
        f_name = 'target_' + str(i+1) + '.pt'
        file = os.path.join(folder, f_name) 
        torch.save(target_tensor, file)

'''---------------------------------------------------------------'''

def process_tensors(config, in_vocab, out_vocab):    
    files = config['csv_files'] 
    in_folder = config['in_folder'] 
    text_len = config['max_text'] 
    sum_len = config['max_sum']

    text_vocab = object_load(in_vocab)
    sum_vocab = object_load(out_vocab)
    print(text_vocab.n_words)
    print(sum_vocab.n_words)
    #print(text_vocab.__dict__)
    #print(sum_vocab.__dict__)
    print()
    for file in files:
        folder = file.split('_')[1]
        #print(folder)
        file  = os.path.join(in_folder, file)
        df = pd.read_csv(file, encoding = 'utf-8')
        #df = df.head(5)
        #print('\n=========================================')
        print('Training data:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))
        folder = os.path.join(in_folder, folder)

        prepare_tensors(text_vocab, sum_vocab, df, folder, text_len, sum_len)
        print('--------------------')
