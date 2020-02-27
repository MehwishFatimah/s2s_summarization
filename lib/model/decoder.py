#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 18:45:14 2020

@author: fatimamh
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, device, config):
        super(Decoder, self).__init__()

        # Declare the hyperparameter
        self.device = device
        self.emb_dim      = config["emb_dim"]
        self.hidden_dim   = config["hid_dim"]#*2
        self.output_dim   = config["sum_vocab"]
        self.n_layers     = config["num_layers"]
        self.dropout      = config["dec_drop"]
        if self.n_layers == 1:
            self.dropout = 0
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim) # emb dim
        #self.embedding.weight.data.copy_(torch.eye(hidden_dim))
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)  

    def forward(self, input, hidden):
        #print('---------------------------')
        #print('Decoder:\tinput: {}\n\t\t\thidden: {}'.format(input.shape, hidden.shape))
        batch_size = input.size(0)
        output = self.embedding(input).view(1, batch_size, self.hidden_dim)
        #print('Decoder:\tembedding: {}'.format(output.shape))
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.gru(output, hidden)
        
        #print('Decoder:\tgru-output: {}\n\t\t\tgru-hidden: {}'.format(output.shape, hidden.shape))
        output = self.softmax(self.out(output[0]))
        #output = self.softmax(self.out(output.squeeze(0)))
        #print('Decoder:\tsoftmax: {}'.format(output.shape))
        #print('---------------------------')
        return output, hidden

    def initHidden(self,batch_size):
        return Variable(torch.zeros(1, batch_size, self.hidden_dim), device=self.device)

'''-------------------------------------------------------------------
'''
'''----------------------------------------------------------------
Att-class
'''
class Attn(nn.Module):
    def __init__(self, device, method, hidden_dim):
        super(Attn, self).__init__()
        self.device = device
        self.method = method
        self.hidden_dim = hidden_dim
        
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_dim, hidden_dim)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_dim * 2, hidden_dim)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_dim))

    def forward(self, hidden, encoder_outputs):
        
        max_len = encoder_outputs.size(0)
        #print('Att:\tmax_len: {}'.format(max_len))
        
        this_batch_size = encoder_outputs.size(1)
        #print('Att:\tthis_batch_size: {}'.format(this_batch_size))

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S
        attn_energies = attn_energies.to(self.device)
        #print('Att:\tattn_energies: {}'.format(attn_energies.shape))
        
        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                #print('Att:\thidden[:, b]: {} encoder_outputs[i, b].unsqueeze(0) {}'.\
                #    format(hidden[:, b].shape, encoder_outputs[i, b].unsqueeze(0).shape))
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        attn_energies = F.softmax(attn_energies, dim=1).unsqueeze(1)
        #print('Att:\tattn_energies: {}'.format(attn_energies.shape))
        return attn_energies
    
    def score(self, hidden, encoder_output):
        
        if self.method == 'dot':
            energy = torch.dot(hidden.view(-1), encoder_output.view(-1))#hidden.dot(encoder_output)
            #print('Att:\tenergy: {}'.format(energy.shape))
            return energy
        
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = torch.dot(hidden.view(-1), energy.view(-1))#hidden.dot(energy)
            #print('Att:\tenergy: {}'.format(energy.shape))
            return energy
        
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = torch.dot(self.v.view(-1), eenergy.view(-1)) #self.v.dot(energy)
            #print('Att:\tenergy: {}'.format(energy.shape))
            return energy


'''----------------------------------------------------------------
Att-decoder
LuongAttnDecoderRNN
''' 
class AttentionDecoder(nn.Module):
    
    def __init__(self, device, config, attn_model='general'):
        super(AttentionDecoder, self).__init__()

        # Declare the hyperparameter
        self.device       = device
        self.emb_dim      = config["emb_dim"]
        self.hidden_dim   = config["hid_dim"]#*2
        self.output_dim   = config["sum_vocab"]
        self.n_layers     = config["num_layers"]
        self.dropout      = config["dec_drop"]
        if self.n_layers == 1:
            self.dropout = 0
        self.attn_model = attn_model
        
        # Define layers
        self.embedding = nn.Embedding(self.output_dim, self.emb_dim)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(self.emb_dim, self.hidden_dim, self.n_layers, dropout=self.dropout)
        self.concat = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(self.device, self.attn_model, self.hidden_dim)

    def forward(self, input, hidden, encoder_outputs):
        # Note: we run this one step at a time
        print('---------------------------')
        #print('Decoder:\tinput: {}\n\t\t\thidden: {}\n\t\t\tencoder_outputs: {}'\
        #    .format(input.shape, hidden.shape, encoder_outputs.shape))
        # Get the embedding of the current input word (last output word)
        batch_size = input.size(0)
        #print('Decoder:\tbatch_size: {}'.format(batch_size))
        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        #print('Decoder:\tembedded : {}'.format(embedded.shape))
        embedded = embedded.view(1, batch_size, self.emb_dim) # S=1 x B x N
        print('Decoder:\tembedded : {}'.format(embedded.shape))
        
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, hidden)
        #print('Decoder:\trnn_output: {}'.format(rnn_output.shape))
        #print('Decoder:\thidden: {}'.format(hidden.shape))
        
        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        #print('Decoder:\tattn_weights: {}'.format(attn_weights.shape))
        #print('Decoder:\tencoder_outputs.transpose(0, 1): {}'.format(encoder_outputs.transpose(0, 1).shape))
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
        #print('Decoder:\tcontext: {}'.format(context.shape))

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        #print('Decoder:\trnn_output: {}'.format(rnn_output.shape))

        context = context.squeeze(1)       # B x S=1 x N -> B x N
        #print('Decoder:\tcontext: {}'.format(context.shape))
        
        concat_input = torch.cat((rnn_output, context), 1)
        #print('Decoder:\tconcat_input: {}'.format(concat_input.shape))
        concat_output = torch.tanh(self.concat(concat_input))
        #print('Decoder:\tconcat_output: {}'.format(concat_output.shape))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        print('Decoder:\toutput: {}'.format(output.shape))
        # Return final output, hidden state, and attention weights (for visualization)
        print('---------------------------')
        return output, hidden #, attn_weights
        

    def initHidden(self):
        return Variable(torch.zeros(1, batch_size, self.hidden_dim), device=self.device)