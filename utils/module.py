import os
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pad_sequence

from lightning.pytorch import LightningModule

from typing import Any

from pdb import set_trace

class AsrModule(LightningModule):
    def __init__(self, 
                 params: argparse.Namespace,
                 model: nn.Module) -> None:
        super().__init__()
        torch.manual_seed(params.seed)
        self.params = params
        self.model = model
        self.criterion = nn.CTCLoss()

    def training_step(self, 
                      batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                      batch_idx: int) -> Any:
        X,Y,fs = batch
        # X,X_lens = pad_packed_sequence(X)
        Y,Y_lens = pad_packed_sequence(Y)
        Y_hat, Y_hat_lens = self.model(X)
        loss = self.criterion(pad_sequence(Y_hat), pad_sequence(Y), Y_hat_lens, Y_lens)
        self.log('train/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, 
                      batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                      batch_idx: int) -> Any:
        X,Y,fs = batch
        # X,X_lens = pad_packed_sequence(X)
        Y,Y_lens = pad_packed_sequence(Y)
        Y_hat, Y_hat_lens = self.model(X)
        loss = self.criterion(pad_sequence(Y_hat), pad_sequence(Y), Y_hat_lens, Y_lens)
        self.log('val/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, 
                  batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                  batch_idx: int) -> Any:
        X,Y,fs = batch
        # X,X_lens = pad_packed_sequence(X)
        Y,Y_lens = pad_packed_sequence(Y)
        Y_hat, Y_hat_lens = self.model(X)
        loss = self.criterion(pad_sequence(Y_hat), pad_sequence(Y), Y_hat_lens, Y_lens)
        self.log('test/loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.params.lr,
                                weight_decay=self.params.wd)
        return opt