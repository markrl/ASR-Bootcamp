import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule

from utils.tokenizer import Tokenizer

class AsrDataModule(LightningDataModule):
    def __init__(self, 
                 params):
        super().__init__()
        self.params = params
        tokenizer = Tokenizer(params.text_encoding,
                                   params.dictionary_dir,
                                   params.token_type)
        self.train_data = AsrData(params, tokenizer, 'train')
        self.val_data = AsrData(params, tokenizer, 'val')
        self.test_data = AsrData(params, tokenizer, 'test')
    
    def setup(self, stage):
        return
    
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.params.batch_size,
            num_workers=self.params.n_workers,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.params.batch_size,
            num_workers=self.params.n_workers,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.params.batch_size,
            num_workers=self.params.n_workers,
            shuffle=False,
        )
    

class AsrData(Dataset):
    def __init__(self, 
                 params, 
                 tokenizer,
                 fold):
        super().__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.fold = fold
        self.tsv_path = os.path.join(params.data_root, f'{fold}.tsv')
        self.length = len(open(self.tsv_path, 'r').readlines())-1

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return