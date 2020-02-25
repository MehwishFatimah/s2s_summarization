#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:05:55 2020

@author: fatimamh
"""
import time
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from random import seed
from random import randint

#from model.evaluation import evaluate

'''
====================================================================
0. Global def
====================================================================
'''
folder = '/hits/basement/nlp/fatimamh/s2s_translator/'
MAX_LENGTH = 10
'''
====================================================================
11. Plot functions
====================================================================
'''
def showPlot(epochs, train_loss, val_loss): #, val_loss):
    
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.AutoLocator()#MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    #ax.xaxis.set_major_locator(loc)
    
    x = epochs
    y = train_loss
    z = val_loss
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x, y, color='blue', linewidth=2, label='Train_loss')
    plt.plot(x, z, color='red', linewidth=2, label='Val_loss')
    
    plt.legend(loc='best')
    #plt.plot(points)
    seed(time.time())
    #print('seed: {}'.format(seed))
    v1 = randint(1,100)
    v2 = randint(1,100)

    file = 'plot_loss_' + str(v1) + str(v2) + '.png'
    file = os.path.join(folder, file)
    plt.savefig(file)
    print(file)

'''---------------------------------------------------------------'''
def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    seed(time.time())
    #print('seed: {}'.format(seed))
    v1 = randint(1,100)
    v2 = randint(1,100)

    file = 'attention_' + str(v1) + str(v2) + '.png'
    file = os.path.join(folder, file)
    plt.show()
    plt.savefig(file)


'''---------------------------------------------------------------'''
def evaluateAndShowAttention(input_lang, output_lang, encoder1, attn_decoder1, input_sentence):
    #output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
