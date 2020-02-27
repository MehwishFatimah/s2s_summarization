import argparse
import resource
import time
import os
import sys
from os import path


from lib.utils.file_utils import read_content
from lib.utils.memory_utils import get_size

# data processing utils
from lib.utils.data_utils import process_data
from lib.utils.dataset_utils import process_vacabs
from lib.utils.dataset_utils import object_load
from lib.utils.dataset_utils import object_save

parser = argparse.ArgumentParser(description = 'Help for the main module')
parser.add_argument('--dc', type = str,    default = '/hits/basement/nlp/fatimamh/s2s_summarization/data_config_en_sub',   
                                                            help = 'Configuration file')
parser.add_argument('--mc', type = str,    default = '/hits/basement/nlp/fatimamh/s2s_summarization/model_config_sub',   
                                                            help = 'Configuration file')

'''----------------------------------------------------------------
'''
def data_processing(config):
    # Step 1: Convert data from json to csv. CLEAN AND SHORT 
    start_time = time.time()
    process_data(files = config['json_files'], 
                 folder = config['in_folder'], 
                 clean_text = True, 
                 short_text = True, 
                 text_len = config['max_text'], 
                 summary_len = config['max_sum'], 
                 ext = '.csv')
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
        format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
    
    
    # Step 2: Generate vocabs
    start_time = time.time()
    in_vocab, out_vocab = process_vacabs(files = config['csv_files'], 
                                         in_folder = config['in_folder'], 
                                         dict_folder = config['test_folder'])
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
        format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))

    '''
    # Step 2:Generate tensors
    start_time = time.time()
    process_tensors(files = config['csv_files'], 
                    in_folder = config['in_folder'], 
                    in_vocab = in_vocab,
                    out_vocab = out_vocab,
                    text_len = config['max_text'], 
                    sum_len = config['max_sum'])
    print ('\n-------------------Memory and time usage: {:.2f} MBs in {:.2f} seconds.--------------------\n'.\
        format((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), (time.time() - start_time)))
    '''
'''----------------------------------------------------------------
'''
def load_vocabs(config):
    
    file = os.path.join(config["test_folder"], config["text_dict"])
    input_vocab = object_load(file)
    print(input_vocab.n_words)
    file = os.path.join(config["test_folder"], config["sum_dict"])
    output_vocab = object_load(file)
    print(output_vocab.n_words)

    return input_vocab, output_vocab

if __name__ == "__main__":
    args       = parser.parse_args()
    #print(args)   
    d_config   = eval(read_content(args.dc))
    m_config   = eval(read_content(args.mc))
    
    '''------------------------------------------------------------
    Step 1: Prepare data tensors and language objects
    ------------------------------------------------------------'''
    data_processing(d_config)

    input_vocab = None 
    output_vocab = None
    
    input_vocab, output_vocab = load_vocabs(m_config)
    #print(input_vocab.__dict__)
    ## provide vocab len -- to do add in config
    input_vocab.condensed_vocab(20000)
    output_vocab.condensed_vocab(10000)
    f_input = object_save('en_text_vocab_cond', input_vocab, d_config["test_folder"])
    f_output = object_save('en_sum_vocab_cond', output_vocab, d_config["test_folder"])