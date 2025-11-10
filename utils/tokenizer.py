import os
import torch
import torch.nn as nn
import pickle

class Tokenizer(nn.Module):
    def __init__(self,
                 text_encoding: str,
                 dictionary_dir: str,
                 token_type: str):
        super().__init__()

        if dictionary_dir is None:
            self.dictionary = {'<sos>':0}
            self.text_map = {0:'<sos>'}
        else:
            with open(os.path.join(dictionary_dir, 'unit2idx.pkl'), 'rb') as f:
                self.dictionary = pickle.load(f)
            with open(os.path.join(dictionary_dir, 'idx2unit.pkl'), 'rb') as f:
                self.text_map = pickle.load(f)
        
        if text_encoding=='onehot':
            num_classes = len(self.dictionary)
            self.encoder = lambda x: nn.functional.one_hot(x, num_classes)
        else:
            return NotImplementedError
        
        self.token_type = token_type

        self.remove_symbols = ['.', '?', ',', '!', ':', ';', '"', '-', '`', '(', ')']
        self.alternative_apostrophes = ['\u2018', '\u2019', '\u201C', '\u201D']
        self.space_symbols = ["'"]
        if token_type=='character':
            self.remove_symbols += self.alternative_apostrophes + self.space_symbols
        self.unknown_token = '<unk>'
        self.start_token = '<sos>'
        self.stop_token = '<eos>'

    def preprocess_text(self, text: str) -> list:
        text = text.strip()
        text = text.lower()
        for symbol in self.alternative_apostrophes:
            text = text.replace(symbol, "'")
        for symbol in self.space_symbols:
            text = text.replace(symbol, ' ' + symbol)
        for symbol in self.remove_symbols:
            text = text.replace(symbol, '')
        tokens = self.split_text(text)
        if "'" in tokens:
            tokens.remove("'")
        return [self.start_token] + tokens + [self.stop_token]

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
                idxs.append(self.dictionary[self.unknown_token])
        idxs = torch.LongTensor(idxs)
        encoding = self.encoder(idxs)
        return encoding

    def decode(self, encoding):
        idxs = torch.argmax(encoding[1:-1], dim=-1)
        tokens = [self.text_map[idx.item()] for idx in idxs]
        if self.token_type=='word':
            output = ' '.join(tokens)
        elif self.token_type=='character':
            output = ''.join(tokens)
        return output

if __name__=='__main__':
    tokenizer = Tokenizer('onehot', '/home/marklind/asr_dict_5occurrences', 'word')
    encoding = tokenizer.tokenize("Hi, my name is Mark. I don't really like spaghetti.")
    decoded = tokenizer.decode(encoding)
    print(decoded)