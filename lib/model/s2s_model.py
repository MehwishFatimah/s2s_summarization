#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:35:53 2020

@author: fatimamh
"""


import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from lib.model.encoder import Encoder
from lib.model.decoder import AttentionDecoder

teacher_forcing_ratio = 0.5

'''
Model class connects encoder and decoder object as 1 model
'''
class S2SModel(nn.Module):
    
    def __init__(self, device, config):
        super().__init__()
        #Declare hyperparameters
        self.device      = device
        #self.encoder     = EncoderRNN(input_size, emb_size, hidden_size).to(self.device)
        self.encoder     = Encoder(device, config).to(self.device)

        #self.decoder     = AttnDecoderRNN(emb_size, hidden_size, output_size, dropout=0.1).to(self.device)
        self.decoder     = AttentionDecoder(device, config).to(self.device) 
            
        self.max_text    = config["max_text"]
        self.max_sum     = config["max_sum"]
        self.SP_index    = config["SP_index"]
    
    # if have to use simple decoder : change code here   
    def forward(self, input_tensor, target_tensor, criterion, teacher_forcing=False):
        #print(self.encoder.parameters()) 
        #print(self.decoder.parameters()) 
        '''------------------------------------------------------------
        1: Get batch size from input             
        ------------------------------------------------------------'''
        batch_size = input_tensor.size()[0]
        '''------------------------------------------------------------
        2: Call encoder.initHidden and pass batch size             
        ------------------------------------------------------------'''
        encoder_hidden = self.encoder.initHidden(batch_size)
        
        '''------------------------------------------------------------
        4: Take transpose so length should be 1st dimension and 
            batch should be 2nd dimension now             
        ------------------------------------------------------------'''
        #print('\tinput_tensor: {}'.format(input_tensor.shape))
        input_tensor = Variable(input_tensor.transpose(0, 1))
        target_tensor = Variable(target_tensor.transpose(0, 1))
        #print('\tinput_tensor: {}'.format(input_tensor.shape))
        '''------------------------------------------------------------
        5: Take input and output lengths so encoder and decoder can 
            iterate over tokens step by step             
        ------------------------------------------------------------'''
        input_length = input_tensor.size()[0]
        target_length = target_tensor.size()[0]
        '''------------------------------------------------------------
        6: Initialize encoder_outputs tensor with zeros             
        ------------------------------------------------------------'''
        encoder_outputs = torch.zeros(self.max_text, batch_size, self.encoder.hidden_dim)
        encoder_outputs = encoder_outputs.to(self.device)
        #print('\tencoder_outputs: {}'.format(encoder_outputs.shape))
        '''------------------------------------------------------------
        7: Start encoding:
            Each token (in input tensor of all batch items) will be 
            passed to encoder (fwd is called).
            It will return encoder outputs and hidden states.              
        ------------------------------------------------------------'''
        for ei in range(input_length):
            #print('\tInput_tensor[ei]: {}'.format(input_tensor[ei].shape))
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0]

        '''------------------------------------------------------------
        8: Initialize first to input token for decoder and set h0 with 
        last encoder_hidden --decoder hidden            
        ------------------------------------------------------------'''
        decoder_input = Variable(torch.LongTensor([self.SP_index] * batch_size))
        decoder_input = decoder_input.to(self.device)
        decoder_hidden = encoder_hidden
        all_decoder_outputs = Variable(torch.zeros(self.max_sum, batch_size, 1))
        all_decoder_outputs = all_decoder_outputs.to(self.device)

        #all_attentions = Variable(torch.zeros(self.max_sum, batch_size, self.max_text))
        #all_attentions = all_attentions.to(self.device)
        '''------------------------------------------------------------
        9: Start decoding:
            Each token (in target tensor of all batch items) will be
            passed to decoder (fwd is called).              
        ------------------------------------------------------------'''
        loss = 0
        for di in range(target_length):
            #decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs) 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs) 
            #print('all_decoder_outputs[di] {}'.format(all_decoder_outputs[di].shape))
            topv, topi = decoder_output.data.topk(1)
            #print(topi.shape)
            all_decoder_outputs[di] = topi 
            #print('\t\tdecoder_output: {}'.format(decoder_output.shape))           
            #print('decoder_attention: {} all_attentions[di]: {}'.format(decoder_attention.shape, all_attentions[di].shape))    
            #all_attentions[di] = decoder_attention.data.squeeze()
            #print('all_attentions[di]: {}'.format(all_attentions[di].shape))

            # if not use squeeze --- throws error
            loss += criterion(decoder_output, target_tensor[di].squeeze())
            '''------------------------------------------------------------
            10: Teacher forcing: select token from target tensor             
            ------------------------------------------------------------'''
            if teacher_forcing and random.random() < teacher_forcing_ratio:
                decoder_input = target_tensor[di]  # Teacher forcing: Feed the target as the next input
            else:
                '''------------------------------------------------------------
                11: Not teacher forcing: select predicted token from decoder
                    output by topk. topi is index value with highest probability.             
                ------------------------------------------------------------'''        
                # Without teacher forcing: use its own predictions as the next input
                topv, topi = decoder_output.data.topk(1)
                #print('CL: {}\t\ttopi: {} topi.squeeze: {}'.format(line_numb(), topi.shape, topi.squeeze().shape))
                decoder_input = topi.squeeze().detach()  # detach from history as input
        
        loss /= target_length       
        #print('\t\tall_decoder_outputs: {}'.format(all_decoder_outputs.shape))
        all_decoder_outputs = all_decoder_outputs.transpose(0, 1)
        #print('\t\tall_decoder_outputs: {}'.format(all_decoder_outputs.shape))
        #print('\t\tall_attentions: {}'.format(all_attentions.shape))
        #all_attentions = all_attentions.transpose(0,1)
        #print('\t\tall_attentions: {}'.format(all_attentions.shape))
        #print('all_attentions[:di + 1] {}'.format(all_attentions[:di+1, :len(encoder_outputs)].shape))
    
        return loss, all_decoder_outputs#, all_attentions
        
            


   