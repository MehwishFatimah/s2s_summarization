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
from datetime import datetime
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

from lib.utils.loader_utils import get_data
from lib.utils.file_utils import get_files
from lib.utils.model_utils import get_time
from lib.utils.model_utils import save_checkpoint
from lib.utils.model_utils import load_checkpoint
from lib.utils.file_utils import print_writer
from lib.utils.file_utils import plot_writer
from lib.utils.file_utils import build_row

from lib.utils.plot_utils import showPlot

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
    batch_num = 1
    total_batch = len(data_loader)
    for input_tensor, target_tensor in data_loader:
        batch_size = input_tensor.size()[0]
        if batch_size == 1: continue
        print('\t---Train Batch: {}/{}---'.format(batch_num, total_batch))
        input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device) 
        '''------------------------------------------------------------
        3: Clear old gradients from the last step              
        ------------------------------------------------------------'''
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()
        #loss, predictions, attentions = model(input_tensor, target_tensor, criterion, teacher_forcing=True)
        loss, predictions = model(input_tensor, target_tensor, criterion, teacher_forcing=True)
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
        batch_num+=1
    
    epoch_loss /= len(data_loader)
    return epoch_loss   


'''----------------------------------------------------------------
'''
def evaluate(device, model, data_loader, criterion):
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
    # we don't need to update the model parameters. only forward pass.
    with torch.no_grad():
        batch_num = 1
        total_batch = len(data_loader)
       
        for input_tensor, target_tensor in data_loader:
            batch_size = input_tensor.size()[0]
            if batch_size == 1: continue
            print('\t---Eval Batch: {}/{}---'.format(batch_num, total_batch))
            input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)    
            #loss, predictions, attentions = model(input_tensor, target_tensor, criterion, teacher_forcing=False)
            loss, predictions = model(input_tensor, target_tensor, criterion, teacher_forcing=False)
            
            epoch_loss += loss.item()
            batch_num +=1

    epoch_loss /= len(data_loader)
    return epoch_loss

'''----------------------------------------------------------------
'''

def train_model(device, config, model, criterion, enc_optimizer, dec_optimizer):

    '''-----------------------------------------------
    Step 1: Get from config
    -----------------------------------------------'''
    s_epoch     = None
    e_epoch     = config["epochs"]
    clip        = config["grad_clip"]
    print_every = config["print_every"]
    plot_every  = config["plot_every"]
    print_file  = config["print_file"]
    plot_file   = config["plot_file"]

    '''-----------------------------------------------
    Step 2: Get data_loaders
    -----------------------------------------------'''
    train_loader = get_data(config, "train")
    val_loader   = get_data(config, "val")

    '''-----------------------------------------------
    Step 3: From scratch or resume
    -----------------------------------------------'''
    # TO DO how to make it in a function
    bm_file     = os.path.join(config["check_point_folder"], config["best_model_file"])
    ckpt_file   = os.path.join(config["check_point_folder"], config["check_point_file"]) 
    #model_file = "best-model.pth.tar"
    
    if path.exists(ckpt_file):
        checkpoint 		= load_checkpoint(ckpt_file)
        s_epoch 		= checkpoint["epoch"]
        best_valid_loss = checkpoint["loss"]
		
        model.load_state_dict(checkpoint["model_state"])
        enc_optimizer.load_state_dict(checkpoint["enc_optimizer"])
        dec_optimizer.load_state_dict(checkpoint["dec_optimizer"])
	
    elif path.exists(bm_file):
        checkpoint 		= load_checkpoint(bm_file)
        s_epoch 		= checkpoint["epoch"]
        best_valid_loss = checkpoint["loss"]
		
        model.load_state_dict(checkpoint["model_state"])
        enc_optimizer.load_state_dict(checkpoint["enc_optimizer"])
        dec_optimizer.load_state_dict(checkpoint["dec_optimizer"])

    else: # from scratch
        s_epoch = 1
        best_valid_loss = float('inf')
   
    plot_train_loss_total = 0  
    plot_valid_loss_total = 0  
    '''-----------------------------------------------
    Step 2: Get data_loaders
    -----------------------------------------------'''

    # Training and evluation using train and val sets
    for epoch in range(s_epoch, e_epoch+1):
        print('--------Epoch:{} starts--------\n'.format(epoch))
        train_loss = 0
        valid_loss = 0
        '''------------------------------------------------------------
        5: Get batches from loader and pass to train             
        ------------------------------------------------------------'''
        start_time = datetime.now()
        train_loss = train(device, model, train_loader, enc_optimizer, dec_optimizer, criterion, clip)
        valid_loss = evaluate(device, model, val_loader, criterion)
        end_time = datetime.now()
        time_diff = get_time(start_time, end_time)
        
        plot_train_loss_total += train_loss
        plot_valid_loss_total += valid_loss
        
        print('Epoch: {}/{} ==> {:.0f}% | Time: {}'.format(epoch, e_epoch, epoch/e_epoch*100, time_diff))
        
        is_best = bool(valid_loss < best_valid_loss)
        best_valid_loss = (min(valid_loss, best_valid_loss))

        print_row = build_row(epoch, train_loss, valid_loss, best_valid_loss, time_diff)
        print_writer(print_file, print_row)

        # Save checkpoint if is a new best
        if is_best:
            save_checkpoint({'epoch': epoch,
	    				 	 'model_state': model.state_dict(),
	    				 	 'enc_optimizer': enc_optimizer.state_dict(),
                             'dec_optimizer': dec_optimizer.state_dict(),                             
	    				 	 'loss': best_valid_loss}, bm_file, True)  

        if epoch%5 == 0: 
            save_checkpoint({'epoch': epoch,
	    					 'model_state': model.state_dict(),
                             'enc_optimizer': enc_optimizer.state_dict(),
                             'dec_optimizer': dec_optimizer.state_dict(),
	    				     'loss': best_valid_loss}, ckpt_file, False)  

        if epoch % plot_every == 0:
            
            plot_train_loss_avg = plot_train_loss_total/plot_every 
            plot_valid_loss_avg = plot_valid_loss_total/plot_every
            
            plot_row = build_row(epoch, plot_train_loss_avg, plot_valid_loss_avg)
            plot_writer(plot_file, plot_row)
            
            plot_train_loss_total = 0  
            plot_valid_loss_total = 0
        
    showPlot(config["out_folder"], plot_file)

    return True



