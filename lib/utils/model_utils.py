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
def model_param(model, config):
    # Optim connect it with model
    enc_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), 
                                 lr = config["learning_rate"], 
                                 momentum = config["momentum"])
    dec_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), 
                                 lr = config["learning_rate"], 
                                 momentum = config["momentum"]) 
    #enc_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                              lr=learning_rate)
    #dec_optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
    #                              lr=learning_rate)

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index = config["PAD_index"])
    
    # gradient clip
    #clip = config["grad_clip"]

    return enc_optimizer, dec_optimizer, criterion#, clip

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

'''----------------------------------------------------------------
'''
# TO DO: VErify
def total_params(model):
    for parameter in model.parameters():
            print(parameter.size(), len(parameter)) 
            print()

def trainable_params(model):     
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
        
    print('params: {}'.format(params))