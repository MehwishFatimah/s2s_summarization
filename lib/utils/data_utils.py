#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 14:48:19 2019
Modified on Wed Nov 06
Modified on Wed Feb 12

@author: fatimamh

"""

'''-----------------------------------------------------------------------
Import libraries and defining command line arguments
-----------------------------------------------------------------------'''
import argparse
import os
import re
import pandas as pd
import time
import resource

'''-----------------------------------------------------------------------
Takes text of an article containing following tags from dataframe and returns the text
after removing those tags.
  Args:
    text  : str
  Returns:
    text  : str
'''
def replace_tags_a(text):

    text = text.replace('<ARTICLE>', ' ')
    text = text.replace('</ARTICLE>', ' ')
    text = text.replace('<TITLE>', ' ')
    text = text.replace('</TITLE>', ' ')
    text = text.replace('<HEADING>', ' ')
    text = text.replace('</HEADING>', ' ')
    text = text.replace('<SECTION>', ' ')
    text = text.replace('</SECTION>', ' ')
    text = text.replace('<S>', ' ')
    text = text.replace('</S>', ' ')
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)

    return text

'''-----------------------------------------------------------------------
Takes text of a summary containing following tags from dataframe and returns the text
after removing those tags.
  Args:
    text   : str
  Returns:
    text   : str
'''
def replace_tags_s(text):

    text = text.replace('<SUMMARY>', ' ')
    text = text.replace('</SUMMARY>', ' ')
    text = text.replace('<S>', ' ')
    text = text.replace('</S>', ' ')
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)

    return text

'''-----------------------------------------------------------------------
Takes text from dataframe and returns the tokens of text split by space.
  Args:
    text   : str
  Returns:
    tokens : list
'''
def tokenize(text):

    tokens = text.split()

    return tokens

'''-----------------------------------------------------------------------
Takes text and max length (in terms of tokens). Splits the text into tokens,
if text is shoret than length, it will be all included,
else it will be truncated to the specified length.
After that tokens will be joined again to make text string.
  Args:
    text        : str
    length      : int
  Returns:
    short_text  : str
'''
def make_text_short(text, length):

      text = text.split()
      #print(len(text))

      short_text = text[0]
      if len(text) >= length:
          end = length-1 # 1 token less for adding EP token
      else:
          end = len(text)
      #print(end)
      for i in range(1, end):
          short_text = short_text + ' ' + text[i]

      return short_text

'''-----------------------------------------------------------------------
Takes a dataframe. Print its size, columns and head.
If clean_text is true, then calls cleaning method for article and summary.
If index is stored in file then delete this column.
If short_text is true, then calls short text method for article and summary.
retuns dataframe after specified operations.

  Args:
    df          : dataframe
    clean_text  : bool (default: True)
    short text  : bool (default: False)
    t_len       : int (default: 5000)
    s_len       : int (default: 1000)
  Returns:
    df          : dataframe
'''
def clean_data(df, clean_text, short_text, t_len, s_len):

    #print(clean_text, short_text, t_len, s_len)
    #print('Data before cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))

    if clean_text:
        print('cleaning text')
        df['text']    = df['text'].apply(lambda x: replace_tags_a(x))
        df['summary'] = df['summary'].apply(lambda x: replace_tags_s(x))

    if 'index' in df.columns:
        del df['index']

    if short_text:
        print('shortening text')
        df['text']    = df['text'].apply(lambda x: make_text_short(x, t_len))
        df['summary'] = df['summary'].apply(lambda x: make_text_short(x, s_len))

    print('Data after cleaning:\nSize: {}\nColumns: {}\nHead:\n{}'.format(len(df), df.columns, df.head(5)))

    return df

'''-----------------------------------------------------------------------
'''
def process_data(config, clean_text = True, short_text = True, ext = '.csv'):
  files = config['json_files']
  folder = config['in_folder'] 
  text_len = config['max_text'] 
  summary_len = config['max_sum'] 
  
  #print(folder)
  #print(files)
  for file in files:
      file = os.path.join(folder, file)
      df   = pd.read_json(file, encoding = 'utf-8')
      #print(df.head(2))
      df   = clean_data(df, clean_text, short_text, text_len, summary_len)
      print('\n--------------------------------------------')
      file_name  = os.path.splitext(file)[0]
      file       = os.path.join(folder, file_name + ext)
      print(file)
      df.to_csv(file, index = False)
      print('\n======================================================================================')

