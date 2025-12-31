import os
import sys
import pickle
from tqdm import tqdm

from utils.tokenizer import Tokenizer

from pdb import set_trace

def generate_dict(params, input_data_paths, dictionary_dir, min_occurrences=5, use_sos_eos=False):
    print('\n*Generating dictionary*')
    tokenizer = Tokenizer(None, params.unit_type, False, not params.keep_punctuation)
    dictionary = {}
    for path in input_data_paths:
        print(path)
        with open(path, 'r') as f:
            for line in tqdm(f):
                input_text = line.split('\t')[3]
                input_units = tokenizer.preprocess_text(input_text)
                for unit in input_units:
                    if unit not in dictionary:
                        dictionary[unit] = 1
                    else:
                        dictionary[unit] += 1
    sorted_items = sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
    if use_sos_eos:
        sorted_items = [('<blank>', 100), ('<sos>', 100), ('<eos>', 100), ('<unk>', 100)] + sorted_items
    else:
        sorted_items = [('<blank>', 100), ('<unk>', 100)] + sorted_items
    reverse_dictionary = {}
    keep_idxs = []
    for idx,(key,value) in enumerate(sorted_items):
        if value<min_occurrences:
            dictionary.pop(key)
        else:
            keep_idxs.append(idx)
    for nn,idx in enumerate(keep_idxs):
        key = sorted_items[idx][0]
        dictionary[key] = nn
        reverse_dictionary[nn] = key
    
    if os.path.exists(dictionary_dir):
        os.system(f'rm -rf {dictionary_dir}')
    os.mkdir(dictionary_dir)
    with open(os.path.join(dictionary_dir, 'unit2idx.pkl'), 'wb') as f:
        pickle.dump(dictionary, f)
    with open(os.path.join(dictionary_dir, 'idx2unit.pkl'), 'wb') as f:
        pickle.dump(reverse_dictionary, f)
    print()
    return dictionary, reverse_dictionary

if __name__=='__main__':
    from params import get_params
    params = get_params()
    generate_dict(params,
                  ['/data/cv-corpus-23.0-2025-09-05/en/train_short.tsv', 
                        '/data/cv-corpus-23.0-2025-09-05/en/test_short.tsv', 
                        '/data/cv-corpus-23.0-2025-09-05/en/dev_short.tsv'],
                  '/home/marklind/dictionary')