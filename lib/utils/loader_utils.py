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
        #print('ID: {}'.format(ID))
        # Load data and get label
        #print('*****In get_item********')
        f_name = 'text_' + str(ID) + '.pt'
        x_file = os.path.join(self.folder, f_name)
        X = torch.load(x_file)

        f_name = 'sum_' + str(ID) + '.pt'
        y_file = os.path.join(self.folder, f_name)
        y = torch.load(y_file)
        #print('X: \n{}\n------\ny: \n{}\n'.format(X, y))
        return X, y

'''===========================================================================
'''        
def data_loader(folder, list_IDs, batch_size = 8, shuffle = True, num_workers = 6):
    
    # Declare the dataset pipline
    dataset = WikiDataset(folder, list_IDs)    
    loader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    print('LOADER:{}---------------------\n{}'.format(type(loader), loader))
    
    return loader

