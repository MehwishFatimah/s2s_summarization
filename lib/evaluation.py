#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:15:48 2019
Updated: 02 Dec 19
@author: fatimamh
"""
import json
import os
import numpy as np
import pandas as pd

import torch
from rouge import Rouge

from lib.utils.loader_utils import data_loader

from .utils.file_utils import read_content

# to work on single document
'''----------------------------------------------------------------
'''
def get_text(data, index_word):
    
    text = []
    for idx in data:
        i = int(idx.item())
        if i == 1: continue # remove padding
        word = index_word[i]
        text.append(word)
        
    return text    

'''----------------------------------------------------------------
'''
def get_test_data(device, test_loader, batch_size, index_word):

    texts = []
    references = []    
    for batch_idx, batch in enumerate(test_loader):
        sources, targets = [i.type(torch.LongTensor).to(device) for i in batch]
        if sources.size(0) != batch_size: continue
        for source in sources:
            text = get_text(source, index_word)
            text = text[0:len(text)-1]
            text = ' '.join(text)
            texts.append(text)

        for target in targets:
            text = get_text(target, index_word)
            references.append(text)    

    return texts, references

'''----------------------------------------------------------------
'''
def get_system_summaries(file, index_word):

    #print('\n---------------Loading tensor-----{}-------------------\n'.format(file))
    outputs = torch.load(file)
    
    systems = []
    for batch_idx, batch in enumerate(outputs):
        for summary in batch:
            text = get_text(summary, index_word)
            systems.append(text)

    del outputs
    return systems


'''----------------------------------------------------------------
'''
def join(references, systems):

    for i in range(len(references)):
        references[i] = ' '.join(references[i])

    for i in range(len(systems)):
        systems[i] = ' '.join(systems[i])

    # no need to return -- list pass by ref
    #return references, systems

'''----------------------------------------------------------------
'''
def truncate_and_join(references, systems):
    
    for i in range(len(systems)):
        ref = references[i]
        ref = ref[0:len(ref)-1]

        sys = systems[i]
        sys = sys[0:len(ref)]
        
        sys = ' '.join(sys)
        ref = ' '.join(ref)
        references[i] = ref
        systems[i] = sys

    # no need to return -- list pass by ref
    #return references, systems

'''----------------------------------------------------------------
'''
def get_scores(references, systems):
    
    scores = []
    rouge = Rouge()
    for i in range(len(systems)):
        score = rouge.get_scores(systems[i], references[i])
        scores.append(score)

    return scores

'''----------------------------------------------------------------
'''
def get_average(x):
    return sum(x)/len(x)

'''----------------------------------------------------------------
'''
def get_results(df, folder, out_file):

    r1_f, r1_p, r1_r = ([] for _ in range(3))
    r2_f, r2_p, r2_r = ([] for _ in range(3))
    rl_f, rl_p, rl_r = ([] for _ in range(3))

    scores = df['score']
    for i in range(len(scores)):
        score = str(scores[i])
        score = score.replace('[', '')
        score = score.replace(']', '')
        score = score.replace("\'", "\"")
        dic = json.loads(score)
        #print(dic.keys())
        #print(dic.values())
        r1_f.append(dic['rouge-1']['f'])
        r1_p.append(dic['rouge-1']['p'])
        r1_r.append(dic['rouge-1']['r'])

        r2_f.append(dic['rouge-2']['f'])
        r2_p.append(dic['rouge-2']['p'])
        r2_r.append(dic['rouge-2']['r'])

        rl_f.append(dic['rouge-l']['f'])
        rl_p.append(dic['rouge-l']['p'])
        rl_r.append(dic['rouge-l']['r'])

    r1_f.append(get_average(r1_f))
    r1_p.append(get_average(r1_p))
    r1_r.append(get_average(r1_r))

    r2_f.append(get_average(r2_f))
    r2_p.append(get_average(r2_p))
    r2_r.append(get_average(r2_r))

    rl_f.append(get_average(rl_f))
    rl_p.append(get_average(rl_p))
    rl_r.append(get_average(rl_r))

    f_name = os.path.splitext(out_file)[0] + '_results.csv'
    file  = os.path.join(folder, f_name)
    
    rouge = pd.DataFrame(np.column_stack([r1_f, r1_p, r1_r, r2_f, r2_p, r2_r, rl_f, rl_p, rl_r]), columns=['r1_f', 'r1_p', 'r1_r', 'r2_f', 'r2_p', 'r2_r', 'rl_f', 'rl_p', 'rl_r'])
    print(len(rouge), rouge.columns, rouge.head(5))
    rouge.to_csv(file)

'''----------------------------------------------------------------
'''
def rouge_evaluation(device, configure):
	
    i_w_f = configure['i_w_file']
    folder = configure['out_folder']
    index_word = eval(read_content(i_w_f))
    
    batch_size   = configure['batch_size']
    test_folder   = configure['te_folder']
    test_size     = configure['te_size']
    test_list_ids = [*range(0, test_size, 1)]
    test_loader  = data_loader(test_folder, test_list_ids, batch_size = batch_size, shuffle = False)

    texts, references = get_test_data(device, test_loader, batch_size, index_word)    
    del test_loader

    
    file  = os.path.join(configure['out_folder'], configure['out_f'])
    systems = get_system_summaries(file, index_word)

    
    if len(texts) > len(systems): texts = texts[0:len(systems)]
    if len(references) > len(systems): references = references[0:len(systems)]
    

    #references, systems = 
    #truncate_and_join(references, systems)
    
    join(references, systems)
    print(type(systems[0]), type(references[0]))
    
    scores = get_scores(references, systems)
      
    df = pd.DataFrame({'text': texts, 'reference': references, 'system': systems, 'score': scores})
    
    
    f_name =  os.path.splitext(configure['out_f'])[0] + '_and_scores.csv'
    file = os.path.join(configure['out_folder'], f_name)
    df.to_csv(file)

    get_results(df, configure['out_folder'], configure['out_f'])

    return True

'''----------------------------------------------------------------
'''

'''
    unk = '<unk>'
    for system in systems:
        t = len(system)-1
        print('top: {} {}'.format(t,system[t]))
        for i in range(len(system)):
            if i == 0: continue # first    
            elif i == len(system)-1: continue# last
            else:
                if system[i-1] == unk and system[i] == unk and system[i+1] == unk:
                    print(system[i-1], system[i], system[i+1])
                    t = i-1
                    print('in if: {} {}'.format(t,system[t]))
                    break
        print('out: {} {}'.format(t, system[t]) )
    '''