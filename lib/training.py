#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:07:57 2019

@author: fatimamh
"""
import os
from os import path
import sys
import time
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from lib.utils.loader_utils import data_loader
from lib.utils.file_utils import get_files
from lib.utils.model_utils import get_time
from lib.utils.model_utils import save_checkpoint
from lib.utils.model_utils import load_checkpoint


import torch
import torch.nn as nn
from torch import optim
import random
import time
import math
import os

import subprocess
from utils.plot_utils import showPlot
from torch.autograd import Variable

'''----------------------------------------------------------------
'''

def train(device, model, data_loader, enc_optimizer, dec_optimizer, criterion, clip): 
    ''' Training loop for the model to train.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        optimizer: Optimizer for the model.
        criterion: loss criterion.
        clip: gradient clip value.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''

    # set model into train mode
    model.train()
    # define epoch loss
    epoch_loss = 0
    for input_tensor, target_tensor in data_loader:
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device) 
        '''------------------------------------------------------------
        3: Clear old gradients from the last step              
        ------------------------------------------------------------'''
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        loss, predictions, attentions = model(input_tensor, target_tensor, criterion, teacher_forcing=True)
        #print('predictins: {}'.format(predictions.shape))
        #print('target_tensor: {}'.format(target_tensor.shape))
        '''------------------------------------------------------------
        13: Compute the derivative of the loss w.r.t. the parameters 
            (or anything requiring gradients) using backpropagation.             
        ------------------------------------------------------------'''
        #del predictions
        loss.backward()
        # clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        '''------------------------------------------------------------
        14: Update optimizer to take a step based on the gradients 
            of the parameters.             
        ------------------------------------------------------------'''
        enc_optimizer.step()
        dec_optimizer.step()
        epoch_loss += loss.item()
    
    epoch_loss /= len(data_loader)
    return epoch_loss   


'''----------------------------------------------------------------
'''

def evaluate(device, model, data_loader, criterion, max_length=MAX_LENGTH, testing= True):
    ''' Evaluation loop for the model to evaluate.
    Args:
        model: A Seq2Seq model instance.
        iterator: A DataIterator to read the data.
        criterion: loss criterion.
    Returns:
        epoch_loss: Average loss of the epoch.
    '''
	#set model in evaluation mode
    #  some layers have different behavior during train/and evaluation (like BatchNorm, Dropout) so setting it matters.
    model.eval()
	#declare loss
    epoch_loss = 0
    all_predictions = []
    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
       
        for input_tensor, target_tensor in data_loader:
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)    
            loss, predictions, attentions = model(input_tensor, target_tensor, criterion, teacher_forcing=False)
            if testing:
                all_predictions.append(predictions)
            epoch_loss += loss.item()

    epoch_loss /= len(data_loader)
    if testing:
        return epoch_loss, all_predictions
    else:
        return epoch_loss

'''----------------------------------------------------------------
'''

 
def train_model(device, configure, model, train_loader, val_loader, model, criterion, enc_optimizer, dec_optimizer):

    epochs          = configure['epocs']
    #learning_rate   = configure['learning_rate']
    clip            = configure['grad_clip']
    print_every     = configure['print_every']
    plot_every      = configure['plot_every']

    bm_file 	= os.path.join(configure['ckpt_folder'], configure["bm_f"])
    ckpt_file   = os.path.join(configure['ckpt_folder'], configure["ckpt_f"])
	#model_file = 'best-model.pth.tar'

    if path.exists(ckpt_file):
        checkpoint 		= load_checkpoint(ckpt_file)
        s_epoch 		= checkpoint['epoch']
        best_valid_loss = checkpoint['loss']
		
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
	
    elif path.exists(bm_file):
        checkpoint 		= load_checkpoint(bm_file)
        s_epoch 		= checkpoint['epoch']
        best_valid_loss = checkpoint['loss']
		
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    else:
        s_epoch = 1
        best_valid_loss = float('inf')

    e_epoch = configure['epochs']
    print('s_epoch: {} | e_epoch: {}'.format(s_epoch, e_epoch))

    e_count = []
    t_loss = []
    v_loss = []
    b_loss = []

    prev_gpu_memory_usage = 0
    plot_train_losses = []
    plot_valid_losses = []
    plot_epochs = []
    
    plot_train_loss_total = 0  
    plot_valid_loss_total = 0  

    # Training and evluation using train and val sets
    for epoch in range(s_epoch, e_epoch+1):
        print('--------Epoch:{} starts-----------------------------\n'.format(epoch))
        train_loss = 0
        valid_loss = 0
        '''------------------------------------------------------------
        5: Get batches from loader and pass to train             
        ------------------------------------------------------------'''
        start_time = time.time()
        train_loss = train(device, model, train_loader, enc_optimizer, dec_optimizer, criterion, clip)
        valid_loss = evaluate(device,model, val_loader, criterion, testing =False)
        end_time = time.time()
        
        plot_train_loss_total += train_loss
        plot_valid_loss_total += valid_loss
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        '''
        curr_gpu_memory_usage = get_gpu_memory_usage(device_id=torch.cuda.current_device())
        diff_gpu_memory_usage = curr_gpu_memory_usage - prev_gpu_memory_usage
        prev_gpu_memory_usage = curr_gpu_memory_usage
        print(curr_gpu_memory_usage)
        '''
        # add proper path for model -
        # TO DO per epoch - store - best model --resume training
        print('Epoch: {}/{} ==> {:.0f}% | Time: {}m {}s'.format(epoch, epochs, epoch/epochs*100, epoch_mins, epoch_secs))
        #print(torch.cuda.get_device_properties(device).total_memory)
        print('Train loss: {:.2f}'.format(train_loss))
        print('Valid loss: {:.2f}'.format(valid_loss))
        print('plot_train_loss_total: {:.2f}'.format(plot_train_loss_total))
        print('plot_valid_loss_total: {:.2f}'.format(plot_valid_loss_total))
        print()

        print('best vs current: {} | {}'.format(best_valid_loss, valid_loss))
        is_best = bool(valid_loss < best_valid_loss)
        best_valid_loss = (min(valid_loss, best_valid_loss))
        print('best vs current: {} | {}'.format(best_valid_loss, valid_loss))

        e_count.append(epoch)
        t_loss.append(train_loss)
        v_loss.append(valid_loss)
        b_loss.append(best_valid_loss)

        # Save checkpoint if is a new best
        if is_best:
            save_checkpoint({'epoch': epoch,
	    				 	 'state_dict': model.state_dict(),
	    				 	 'optimizer': optimizer.state_dict(),
	    				 	 'loss': best_valid_loss}, bm_file, True)  

        if epoch%5 == 0: 
            save_checkpoint({'epoch': epoch,
	    					 'state_dict': model.state_dict(),
	    				     'optimizer': optimizer.state_dict(),
	    				     'loss': best_valid_loss}, ckpt_file, False)  

        if epoch % plot_every == 0:
            plot_epochs.append(epoch)
            
            plot_train_loss_avg = plot_train_loss_total/plot_every
            print('plot_train_loss_avg {:.2f}'.format(plot_train_loss_avg))
            plot_train_losses.append(plot_train_loss_avg)
            
            plot_valid_loss_avg = plot_valid_loss_total/plot_every
            print('plot_valid_loss_avg {:.2f}'.format(plot_valid_loss_avg))
            plot_valid_losses.append(plot_valid_loss_avg)
            
            plot_train_loss_total = 0  
            plot_valid_loss_total = 0
        print()
        print('--------Epoch:{} ends-------------------------------------------\n'.format(epoch))
        
    showPlot(plot_epochs, plot_train_losses, plot_valid_losses)

    df = pd.DataFrame({'epoch': e_count, 't_loss': t_loss, 'v_loss': v_loss, 'b_loss': b_loss})
    f_name = str(int(epoch)) + '_' + configure['loss_f']
    file = os.path.join(configure['ckpt_folder'], f_name)
    df.to_csv(file)
    print(len(df), df.columns, df.head(5))
    return True



