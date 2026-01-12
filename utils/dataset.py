import os
import glob
import argparse
import torch
from itertools import islice
import soundfile as sf

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from lightning.pytorch import LightningDataModule

from utils.tokenizer import Tokenizer
from utils.generate_dictionary import generate_dict

from pdb import set_trace


class AsrDataModule(LightningDataModule):
    def __init__(self, 
                 params: argparse.Namespace) -> None:
        super().__init__()
        self.params = params
        if params.dictionary_dir is None:
            params.dictionary_dir = 'tmp_dict'
            if os.path.exists(params.dictionary_dir):
                os.system(f'rm -rf {params.dictionary_dir}')
            generate_dict(params,
                          ['/data/cv-corpus-23.0-2025-09-05/en/train_short.tsv',
                           '/data/cv-corpus-23.0-2025-09-05/en/test_short.tsv',
                           '/data/cv-corpus-23.0-2025-09-05/en/dev_short.tsv'],
                          params.dictionary_dir,
                          params.min_occurrences,
                          params.model_type!='ctc',
                          params.model_type=='aed')
        self.tokenizer = Tokenizer(params.dictionary_dir,
                                   params.unit_type,
                                   params.model_type!='ctc',
                                   params.model_type=='aed',
                                   not params.keep_punctuation)
        self.vocab_size = len(self.tokenizer.dictionary)
    
    def setup(self, stage: str = None) -> None:
        if stage == 'fit':
            self.train_data = AsrData(self.params, self.tokenizer, 'train')
            self.val_data = AsrData(self.params, self.tokenizer, 'dev')
        if stage == 'test':
            self.test_data = AsrData(self.params, self.tokenizer, 'test')
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.params.batch_size,
            num_workers=self.params.n_workers,
            shuffle=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.params.batch_size,
            num_workers=self.params.n_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.params.batch_size,
            num_workers=self.params.n_workers,
            shuffle=False,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_lens = torch.LongTensor([len(bb[0]) for bb in batch])
        Y_lens = torch.LongTensor([len(bb[1]) for bb in batch])
        X = pad_sequence([bb[0] for bb in batch], batch_first=True)
        Y = pad_sequence([bb[1] for bb in batch], batch_first=True)
        X = pack_padded_sequence(X, X_lens, batch_first=True, enforce_sorted=False)
        Y = pack_padded_sequence(Y, Y_lens, batch_first=True, enforce_sorted=False)
        fs = torch.LongTensor([bb[2] for bb in batch])
        return X,Y,fs
    

class AsrData(Dataset):
    def __init__(self, 
                 params: argparse.Namespace, 
                 tokenizer: Tokenizer,
                 fold: str) -> None:
        super().__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.fold = fold
        self.tsv_path = os.path.join(params.data_root, f'{fold}_short.tsv')
        self.length = len(open(self.tsv_path, 'r').readlines())-1
        self.clips_dir = os.path.join(params.data_root, 'clips_16k')

    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        # Skip the header
        index = index + 1
        with open(self.tsv_path, 'r') as f:
            line = next(islice(f, index, index+1)).split('\t')
        filepath = line[1]
        sentence = line[3]
        x,fs = sf.read(os.path.join(self.clips_dir, filepath))
        x = torch.from_numpy(x).float()
        y = self.tokenizer(sentence)
        return x,y,fs


if __name__=='__main__':
    from params import get_params
    params = get_params()
    data_module = AsrDataModule(params)
    data_module.setup('fit')
    train_data = data_module.train_data
    print(train_data[0])
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        set_trace()