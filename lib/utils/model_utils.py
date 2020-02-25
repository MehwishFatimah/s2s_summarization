#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:06:36 2019

@author: fatimamh
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

'''----------------------------------------------------------------
'''
def model_param(model, configure, PAD_IDX):
    # Optim connect it with model
    optimizer = optim.SGD(model.parameters(), lr = configure['learning_rate'], momentum = configure['momentum']) # define optim

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    
    # gradient clip
    clip = configure['grad_clip']

    return optimizer, criterion, clip

'''----------------------------------------------------------------
'''
def get_time(start_time, end_time):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60)

    return elapsed_mins, elapsed_secs

'''----------------------------------------------------------------
'''
def save_checkpoint(state, file, best):
	if best:
		print ('-------------Saving the new best model------')
	else:
		print ('-------------Saving the new ckpt model------')
    
	print('Epoch: {}, Best_loss: {}'.format(state['epoch'], state['loss']))
	torch.save(state, file)  # save checkpoint
    

'''----------------------------------------------------------------
'''
def load_checkpoint(file):
    print ('-------------Loading the model------')
    state = torch.load(file)  # save checkpoint
    print('Epoch: {}, Best_loss: {}'.format(state['epoch'], state['loss']))
    return state