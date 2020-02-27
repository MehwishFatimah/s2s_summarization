"""
Created on Fri Feb 21 18:45:14 2020

@author: fatimamh
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import inspect


'''-----------------------------------------------------------------------
'''
class Encoder(nn.Module):
    def __init__(self, device, config):
        super(Encoder, self).__init__()

        # Declare the hyperparameter
        self.device = device
        self.input_dim = config["text_vocab"]
        self.embed_dim = config["emb_dim"]
        self.hidden_dim = config["hid_dim"]
        self.n_layers = config["num_layers"]
        self.dropout = config["enc_drop"]
        
        self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(self.embed_dim, self.hidden_dim, self.n_layers, dropout=self.dropout, bidirectional=False)
        self.dropout = nn.Dropout(self.dropout)


    def forward(self, input, hidden):
        print('---------------------------')
        #print('Encoder:\tinput: {}\n\t\t\thidden: {}'.format(input.shape, hidden.shape))
        batch_size = input.size(0)
        embedded = self.embedding(input).view(1, batch_size, self.embed_dim)
        print('Encoder:\tembedding: {}'.format(embedded.shape))
        output = embedded # self.dropout(embedded)
        for i in range(self.n_layers):
            output, hidden = self.gru(output, hidden)
            #print('Encoder:\tgru-output: {}\n\t\t\tgru-hidden: {}'.format(output.shape, hidden.shape))
            #hidden = torch.cat((hidden[:1, :, :], hidden[1:, :, :]), 2) # bidirectional
        print('Encoder:\tgru-output: {}\n\t\t\tgru-hidden: {}'.format(output.shape, hidden.shape))
        print('---------------------------')
        return output, hidden

    def initHidden(self, batch_size):

        return Variable(torch.zeros(1, batch_size, self.hidden_dim, device= self.device))

   
'''-----------------------------------------------------------------------
'''
