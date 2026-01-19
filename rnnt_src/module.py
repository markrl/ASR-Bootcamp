import os
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from lightning.pytorch import LightningModule

from speechbrain.nnet.loss.transducer_loss import TransducerLoss

from utils.functional import edit_distance
from rnnt_src.model import RnntModel
from utils.tokenizer import Tokenizer
from utils.processor import AudioProcessor
# from utils.losses import RnntLoss

from typing import Any

from pdb import set_trace

class RnntModule(LightningModule):
    def __init__(self, 
                 params: argparse.Namespace,
                 vocab_size: int,
                 tokenizer: Tokenizer) -> None:
        super().__init__()
        torch.manual_seed(params.seed)
        self.params = params
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.criterion = TransducerLoss()
        self.val_n_tokens = 0
        self.val_edit_dist = 0
        self.test_n_tokens = 0
        self.test_edit_dist = 0

    def configure_model(self):
        self.model = RnntModel(self.params, self.vocab_size)

    def on_train_epoch_start(self):
        if self.params.augment:
            if self.current_epoch==self.params.augment_epoch:
                self.model.processor.start_augment()
                print('Now applying SpecAugment')

    def training_step(self, 
                      batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                      batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        H,X_lens = self.model(X, Y)
        loss = self.criterion(H, Y, X_lens.to(H.device), Y_lens.to(H.device))
        self.log('train/loss', loss.item(), on_step=True, sync_dist=True,
                 batch_size=self.params.batch_size, prog_bar=True)
        if self.params.train_wer:
            edit_dist, n_tokens = 0, 0
            self.model.eval()
            pred_seqs = model.greedy_search(X)
            for ii in range(len(pred_seqs)):
                pred_seq = pred_seqs[ii]
                targ_seq = Y[ii,:Y_lens[ii]]
                edit_dist += edit_distance(targ_seq, pred_seqs)
                n_tokens += Y_lens[ii]
                if batch_idx%100==0 and ii==0:
                    print(f'TARGET: "{self.tokenizer.decode(targ_seq)}"')
                    print(f'PREDICTED: "{self.tokenizer.decode(pred_seq)}"')
            wer = edit_dist/n_tokens
            self.log('train/wer', wer, on_step=True, sync_dist=True,
                     batch_size=self.params.batch_size, prog_bar=True)
        return loss

    def validation_step(self, 
                      batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                      batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        H,X_lens = self.model(X, Y)
        set_trace()
        loss = self.criterion(H, Y, X_lens.to(H.device), Y_lens.to(H.device))
        self.log('val/loss', loss.item(), on_step=False, sync_dist=True,
                 batch_size=self.params.batch_size, on_epoch=True, prog_bar=not self.params.val_wer)
        if self.params.val_wer:
            pred_seqs = model.greedy_search(X)
            for ii in range(len(pred_seqs)):
                pred_seq = pred_seqs[ii]
                targ_seq = Y[ii,:Y_lens[ii]]
                self.val_edit_dist += edit_distance(targ_seq, pred_seq)
                self.val_n_tokens += Y_lens[ii]
        return loss

    def on_validation_epoch_end(self):
        if self.params.val_wer:
            wer = self.val_edit_dist/self.val_n_tokens
            self.log('val/wer', wer, on_step=False, prog_bar=True, 
                    sync_dist=True, on_epoch=True)
            val_edit_dist = 0
            val_n_tokens = 0

    def test_step(self, 
                  batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                  batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        H,X_lens = self.model(X, Y)
        loss = self.criterion(H, Y, X_lens, Y_lens)
        self.log('test/loss', loss.item(), on_step=False, sync_dist=True, 
                 batch_size=self.params.batch_size, on_epoch=True)
        pred_seqs = model.greedy_search(X)
        for ii in range(len(pred_seqs)):
            pred_seq = pred_seqs[ii]
            targ_seq = Y[ii,:Y_lens[ii]]
            self.test_edit_dist += edit_distance(targ_seq, pred_seq)
            self.test_n_tokens += Y_lens[ii]
        return loss

    def on_test_epoch_end(self):
        wer = self.test_edit_dist/self.test_n_tokens
        self.log('test/wer', wer, on_step=False, 
                 sync_dist=True, on_epoch=True)
        test_edit_dist = 0
        test_n_tokens = 0

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.params.lr,
                            weight_decay=self.params.wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[self.params.finetune_epoch],
            gamma=self.params.finetune_lr_mult
        )
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }