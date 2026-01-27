import os
import torch
import torch.nn as nn
import pickle
import argparse

from pdb import set_trace

class Tokenizer(nn.Module):
    def __init__(self,
                 dictionary_dir: str,
                 token_type: str,
                 use_sos: bool = False,
                 use_eos: bool = False,
                 remove_punctuation: bool = False) -> None:
        super().__init__()

        if dictionary_dir is None:
            self.dictionary = {'<blank>': 0, '<sos>': 1, '<eos>': 2}
            self.text_map = {0: '<blank>', 1: '<sos>', 2: '<eos>'}
        else:
            with open(os.path.join(dictionary_dir, 'unit2idx.pkl'), 'rb') as f:
                self.dictionary = pickle.load(f)
            with open(os.path.join(dictionary_dir, 'idx2unit.pkl'), 'rb') as f:
                self.text_map = pickle.load(f)
                
        self.token_type = token_type
        self.remove_punctuation = remove_punctuation
        self.use_sos = use_sos
        self.use_eos = use_eos

        self.remove_symbols = ['.', '?', ',', '!', ':', ';', '"', '`', '(', ')', '…', '/', '[', ']', 
                                '«', '»', '·', '→', '¡', '#', '„']
        self.alternative_apostrophes = ['\u2018', '\u2019', '\u201C', '\u201D', '´']
        self.space_symbols = ["'"]
        self.other_replacements = {'&': ' and ',
                                   '=': ' equals ',
                                   '~': ' ',
                                   '-': ' ',
                                   '–': ' ',
                                   '—': ' ',
                                   '€': ' euros ',
                                   '%': ' percent ',
                                   '+': ' plus '}
        if token_type=='character':
            self.remove_symbols += self.alternative_apostrophes + self.space_symbols
        self.special_tokens = {'unknown': '<unk>',
                               'start': '<sos>',
                               'end': '<eos>',
                               'blank': '<blank>'}

    def forward(self, text: str) -> torch.Tensor:
        return self.tokenize(text)

    def preprocess_text(self, text: str) -> list:
        text = text.strip()
        text = text.lower()
        for symbol in self.alternative_apostrophes:
            text = text.replace(symbol, "'")
        for symbol in self.other_replacements:
            text = text.replace(symbol, self.other_replacements[symbol])
        if self.remove_punctuation:
            for symbol in self.remove_symbols + self.space_symbols:
                text = text.replace(symbol, ' ')
        else:
            for symbol in self.remove_symbols:
                text = text.replace(symbol, ' '+symbol+' ')
        tokens = self.split_text(text)
        tokens = [token for token in tokens if token!='' and token!="'"]
        if self.use_sos:
            tokens = [self.special_tokens['start']] + tokens
        if self.use_eos:
            tokens += [self.special_tokens['end']]
        return tokens

    def split_text(self, text: str) -> str:
        if self.token_type=='word':
            return text.split(' ')
        elif self.token_type=='character':
            return list(text)
        else:
            raise NotImplementedError

    def tokenize(self, text: str) -> torch.Tensor:
        tokens = self.preprocess_text(text)
        idxs = []
        for token in tokens:
            if token in self.dictionary:
                idxs.append(self.dictionary[token])
            else:
                idxs.append(self.dictionary[self.special_tokens['unknown']])
        idxs = torch.LongTensor(idxs)
        return idxs

    def decode(self, encoding: torch.Tensor) -> str:
        if self.use_sos:
            encoding = encoding[1:]
        if self.use_eos:
            encoding = encoding[:-1]
        if len(encoding.shape) > 1:
            idxs = torch.argmax(encoding, dim=-1)
        else:
            idxs = encoding
        tokens = []
        for idx in idxs:
            token = self.text_map[idx.item()]
            if token==self.special_tokens['blank'] or token==self.special_tokens['start']:
                continue
            elif token==self.special_tokens['end']:
                break
            else:
                tokens.append(self.text_map[idx.item()])
        # tokens = [self.text_map[idx.item()] for idx in idxs if self.text_map[idx.item()]!=self.special_tokens['blank']]
        if self.token_type=='word':
            output = ' '.join(tokens)
        elif self.token_type=='character':
            output = ''.join(tokens)
        return output

    def collapse_ctc(self, encoding: torch.Tensor) -> torch.Tensor:
        if len(encoding.shape) > 1:
            idxs = torch.argmax(encoding, dim=-1)
        else:
            idxs = encoding
        collapsed = []
        prev_idx = -1
        for idx in idxs:
            if idx!=prev_idx:
                collapsed.append(idx)
                prev_idx = idx
        return torch.Tensor([idx for idx in collapsed if self.text_map[idx.item()]!=self.special_tokens['blank']])

    def remove_special(self, encoding: torch.Tensor) -> torch.Tensor:
        if self.use_sos:
            encoding = encoding[1:]
        if self.use_eos:
            first_eos_idx = torch.where(encoding==self.dictionary[self.special_tokens['end']])[0]
            if len(first_eos_idx)>1:
                encoding = encoding[:first_eos_idx[0]]
            elif len(first_eos_idx)==1:
                encoding = encoding[:first_eos_idx]
        return torch.Tensor([idx for idx in encoding if self.text_map[idx.item()]!=self.special_tokens['blank']]).long()


if __name__=='__main__':
    tokenizer = Tokenizer('asr_dict_5occurrences', 'word', False, True)
    encoding = tokenizer.tokenize("Hi, my name is Mark. I live in the United States.")
    print(encoding)
    decoded = tokenizer.decode(encoding)
    print(decoded)