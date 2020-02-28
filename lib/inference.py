#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 18:50:39 2019

@author: fatimamh
"""

import os
import numpy as np
import pandas as pd
from csv import writer
import torch
from lib.training import evaluate
from lib.utils.model_utils import load_checkpoint
from lib.utils.loader_utils import get_data
from lib.utils.dataset_utils import object_load
from lib.utils.dataset_utils import tensor_to_text

'''----------------------------------------------------------------
'''
def load_vocabs(config):
    '''
    file = os.path.join(config["dict_folder"], config["text_dict_f"])
    in_v_f = object_load(file)
    file = os.path.join(config["dict_folder"], config["sum_dict_f"])
    out_v_f = object_load(file)
    '''
    file = os.path.join(config["dict_folder"], config["sum_dict_c"])
    out_v_c = object_load(file)
    
    #return in_v_f, out_v_f, out_v_c
    return out_v_c

'''-----------------------------------------------------------------
'''
def write_csv(file, content):
    with open(file, 'a+', newline='') as obj:
    	#headers = ['text', 'reference', 'system']
        headers = ['reference', 'system']
        csv_writer = writer(obj)
        is_empty = os.stat(file).st_size == 0
        if is_empty:
            csv_writer.writerow(headers)
        csv_writer.writerow(content)
'''-----------------------------------------------------------------
'''
def build_row(*args):
    row =[]
    for i in range(len(args)):
        row.append(args[i])
    return row

'''-----------------------------------------------------------------
'''
def get_text(file, config, reference, system):
	'''
	in_v_f, out_v_f, out_v_c = load_vocabs(config)
	text = text.tolist()
	text = [int(i) for i in text]
	text = tensor_to_text(in_v_f, text)
	'''
	# same vocab used in tensors. Thats why dont use full vocab here
	#print('in get test')
	out_v_c = load_vocabs(config)
	reference = reference.tolist()
	reference = [int(i) for i in reference]
	reference = tensor_to_text(out_v_c, reference)
	
	system = system.tolist()
	system  = [int(i) for i in system]
	system = tensor_to_text(out_v_c, system)	
	#row = build_row(text, reference, system)
	row = build_row(reference, system)
	print('row: {}\n'.format(row))
	write_csv(file, row)
	

'''----------------------------------------------------------------
'''
def test_model(device, config, model, criterion):
	
	data_loader   = get_data(config, "test")
	# test loader must be shuffle false
	#print("In test model")
	file  = os.path.join(config["check_point_folder"], config["best_model_file"])
	best_model = load_checkpoint(file)
	model.load_state_dict(best_model["model_state"])
	model.eval()
	file  = os.path.join(config['out_folder'], config['output_summaries_file'])
	with torch.no_grad():
		batch_num = 1
		total_batch = len(data_loader)
		for input_tensor, target_tensor in data_loader:
			batch_size = input_tensor.size()[0]
			if batch_size == 1: continue # for time being
			print('\t---Test Batch: {}/{}---'.format(batch_num, total_batch))
			input_tensor, target_tensor = input_tensor.to(device), target_tensor.to(device)

			#loss, predictions, attentions = model(input_tensor, target_tensor, criterion, teacher_forcing=False)
			loss, predictions = model(input_tensor, target_tensor, criterion, teacher_forcing=False)
			
			for i in range(batch_size):
				#get_text(file, config, input_tensor[i].squeeze(), target_tensor[i].squeeze(), predictions[i].squeeze())
				get_text(file, config, target_tensor[i].squeeze(), predictions[i].squeeze())
			# dont make change in model; change here -- unpack batch and predictions as well 
			# then for each example -- run lang index to word and get sentence
			# store it as csv row : text ref sys
			batch_num +=1

	return True


    
    
    