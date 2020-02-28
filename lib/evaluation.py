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

from rouge import Rouge


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
def rouge_evaluation(device, config):
    test_file = config['output_summaries_file']
    file  = os.path.join(config['out_folder'], test_file)
    df = pd.read_csv(file)
    print(df.head())

    scores = get_scores(df['reference'], df['system'])      
    df['score'] = scores
    print(df.head())
    f_name =  os.path.splitext(test_file)[0] + '_and_scores.csv'
    file = os.path.join(config['out_folder'], f_name)
    df.to_csv(file)
    get_results(df, config['out_folder'], test_file)

    return True

'''----------------------------------------------------------------
'''
