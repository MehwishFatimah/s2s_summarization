#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:50:39 2019

@author: fatimamh
"""

import os
import numpy as np
import pandas as pd

import torch
from lib.training import evaluate
from lib.utils.model_utils import load_checkpoint
from lib.utils.loader_utils import data_loader

'''----------------------------------------------------------------
'''
def test_model(device, configure, model, criterion):
	
	batch_size   = configure['batch_size']
	
	test_folder   = configure['te_folder']
	test_size     = configure['te_size']
	test_list_ids = [*range(0, test_size, 1)]
	test_loader = data_loader(test_folder, test_list_ids, batch_size = batch_size)

	file  = os.path.join(configure['ckpt_folder'], configure["bm_f"])
	best_model = load_checkpoint(file)
	model.load_state_dict(best_model['state_dict'])
	model.eval()

	test_loss, outputs = evaluate(device, model, test_loader, batch_size, criterion)
	del test_loader
	del model

	file  = os.path.join(configure['out_folder'], configure['out_f'])
	print('\n-----------------------Saving tensor------------------------\n{}\n-------------------------------------------------\n'.format(file))
	torch.save(outputs, file)

	return True


    
    
    