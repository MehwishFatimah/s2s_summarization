#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:20:08 2019
Modified on Thu Nov 7
FINALIZED
@author: fatimamh
"""

import torch
import os
from torch.utils import data

class WikiDataset(data.Dataset):
    def __init__(self, folder, list_IDs):
                
        # 1.Read the content of file
        self.folder = folder
        self.list_IDs = list_IDs
        #print('*****In init********')
        #print('folder: {}, list_IDs: {}'.format(self.folder, self.list_IDs))
               
    def __len__(self):

        #print('*****In len********')
        #print('len: {}'.format(len(self.list_IDs)))
        return len(self.list_IDs)

    def __getitem__(self, index):

        ID = self.list_IDs[index]
        print('ID: {}'.format(ID))
        # Load data and get label
        #print('*****In get_item********')
        f_name = 'input_' + str(ID) + '.pt'
        x_file = os.path.join(self.folder, f_name)
        X = torch.load(x_file)
        #X = nn.ConstantPad1d((0, MAX_LENGTH-X.size(0)),0)(X.t()).t()
        
        f_name = 'target_' + str(ID) + '.pt'
        y_file = os.path.join(self.folder, f_name)
        y = torch.load(y_file)
        #print('X: \n{}\n------\ny: \n{}\n'.format(X, y))
        return X, y

'''===========================================================================
'''        
def data_loader(folder, list_IDs, batch_size=1, shuffle = True, num_workers = 0): #6
    
    # Declare the dataset pipline
    dataset = WikiDataset(folder, list_IDs)    
    loader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    #print('LOADER:{}---------------------\n{}'.format(type(loader), loader))
    
    return loader


def get_data(config, type="train", shuffle = True, num_workers = 0): 
    
    if type == "train":
        batch_size = config["batch_size"]
        folder   = config["train_folder"]
        size     = config["train_docs"] + 1    
        list_ids = [*range(1, size, 1)]
        loader   = data_loader(folder, list_ids, batch_size = batch_size)
        return loader

    if type == "val":
        batch_size   = config["batch_size"]
        folder   = config["val_folder"]
        size     = config["val_docs"] + 1
        list_ids = [*range(1, size, 1)]
        loader   = data_loader(folder, list_ids, batch_size = batch_size)
        return loader
        
    if type == "test": # test
        batch_size   = config["batch_size"]
        folder   = config["test_folder"]
        size     = config["test_docs"] + 1
        list_ids = [*range(1, size, 1)]
        loader   = data_loader(folder, list_ids, batch_size = batch_size, shuffle = False)
        return loader
