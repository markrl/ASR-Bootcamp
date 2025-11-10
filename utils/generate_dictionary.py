import os
import sys
import pickle
from tqdm import tqdm

from utils.tokenizer import Tokenizer
from utils.params import get_params

from pdb import set_trace

def generate_dictionary(input_data_paths, dictionary_dir):
    params = get_params()
    tokenizer = Tokenizer(params.text_encoding, None, params.unit_type)
    dictionary = {}
    for path in input_data_paths:
        print(path)
        with open(path, 'r') as f:
            for line in tqdm(f):
                input_text = line.split('\t')[3]
                input_units = tokenizer.preprocess_text(input_text)                    
                for unit in input_units:
                    # Skip special tokens to ensure they are placed at the beginning of the dictionary
                    if '<' in unit or len(unit)==0:
                        continue
                    if unit not in dictionary:
                        dictionary[unit] = 1
                    else:
                        dictionary[unit] += 1
    sorted_items = sorted(dictionary.items(), key=lambda item: item[1], reverse=True)
    sorted_items = [('<sos>', 100), ('<eos>', 100), ('<unk>', 100)] + sorted_items
    reverse_dictionary = {}
    keep_idxs = []
    for idx,(key,value) in enumerate(sorted_items):
        if value<5:
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

if __name__=='__main__':
    generate_dictionary(['/data/cv-corpus-23.0-2025-09-05/en/train.tsv', 
                        '/data/cv-corpus-23.0-2025-09-05/en/test.tsv', 
                        '/data/cv-corpus-23.0-2025-09-05/en/dev.tsv'],
                        '/home/marklind/dictionary')