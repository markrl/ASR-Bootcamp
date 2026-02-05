import os
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from lightning.pytorch import LightningModule

import k2

from utils.functional import edit_distance
from rnnt_src.model import RnntModel, RnntPredictor, RnntTranscriber
from utils.tokenizer import Tokenizer
from utils.processor import AudioProcessor
from utils.losses import SmoothCtcLoss

from typing import Any

from pdb import set_trace

class RnntModule(LightningModule):
    def __init__(self, 
                 params: argparse.Namespace,
                 vocab_size: int,
                 tokenizer: Tokenizer,
                 transcriber: RnntTranscriber=None,
                 predictor: RnntPredictor=None) -> None:
        super().__init__()
        torch.manual_seed(params.seed)
        self.params = params
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.val_n_tokens = 0
        self.val_edit_dist = 0
        self.test_n_tokens = 0
        self.test_edit_dist = 0
        self.predictor = predictor
        self.transcriber = transcriber

    def configure_model(self):
        blank_idx = self.tokenizer.dictionary['<blank>']
        sos_idx = self.tokenizer.dictionary['<sos>']
        eos_idx = self.tokenizer.dictionary['<eos>'] if self.params.use_eos_token else blank_idx
        self.model = RnntModel(self.params, 
                               self.vocab_size, 
                               blank_idx, 
                               sos_idx, 
                               eos_idx,
                               self.transcriber,
                               self.predictor)
        if self.params.finetune_epoch > 0:
            self.model.freeze_predictor()
            self.model.freeze_transcriber()

    def on_train_epoch_start(self):
        if self.current_epoch==self.params.finetune_epoch:
            self.model.unfreeze_predictor()
            self.model.unfreeze_transcriber()
            print('Now fine tuning E2E')
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
        boundary = torch.tensor([[0,0,Y_len-2,X_len] for Y_len,X_len in zip(Y_lens,X_lens)]).to(Y.device)
        loss = k2.rnnt_loss(H.float(), Y[:,1:], termination_symbol=self.model.eos_idx, boundary=boundary.to(Y.device))
        self.log('train/loss', loss.item(), on_step=True, sync_dist=True,
                 batch_size=self.params.batch_size, prog_bar=True)
        if self.params.train_wer and batch_idx%10==0:
            edit_dist, n_tokens = 0, 0
            if self.current_epoch >= 1:
                set_trace()
            with torch.no_grad():
                pred_seqs = self.model.greedy_search(X)
            for ii in range(len(pred_seqs)):
                pred_seq = self.tokenizer.remove_special(pred_seqs[ii])
                targ_seq = self.tokenizer.remove_special(Y[ii,:Y_lens[ii]])
                edit_dist += edit_distance(targ_seq, pred_seq)
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
        boundary = torch.tensor([[0,0,Y_len-2,X_len] for Y_len,X_len in zip(Y_lens,X_lens)]).to(Y.device)
        loss = k2.rnnt_loss(H.float(), Y[:,1:], termination_symbol=self.model.eos_idx, boundary=boundary.to(Y.device))

        self.log('val/loss', loss.item(), on_step=False, sync_dist=True,
                 batch_size=self.params.batch_size, on_epoch=True, prog_bar=True)
        if self.params.val_wer and batch_idx%10==0:
            pred_seqs = self.model.greedy_search(X)
            for ii in range(len(pred_seqs)):
                pred_seq = self.tokenizer.remove_special(pred_seqs[ii])
                targ_seq = self.tokenizer.remove_special(Y[ii,:Y_lens[ii]])
                self.val_edit_dist += edit_distance(targ_seq, pred_seq)
                self.val_n_tokens += Y_lens[ii]
                if batch_idx%50==0 and ii==0:
                    print(f'TARGET: "{self.tokenizer.decode(targ_seq)}"')
                    print(f'PREDICTED: "{self.tokenizer.decode(pred_seq)}"')
        return loss

    def on_validation_epoch_end(self):
        if self.params.val_wer:
            wer = self.val_edit_dist/self.val_n_tokens
            self.log('val/wer', wer, on_step=False, prog_bar=True, 
                    sync_dist=True, on_epoch=True)
            self.val_edit_dist = 0
            self.val_n_tokens = 0

    def test_step(self, 
                  batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                  batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        H,X_lens = self.model(X, Y)
        boundary = torch.tensor([[0,0,Y_len-2,X_len] for Y_len,X_len in zip(Y_lens,X_lens)]).to(Y.device)
        loss = k2.rnnt_loss(H.float(), Y[:,1:], termination_symbol=self.model.eos_idx, boundary=boundary.to(Y.device))
        self.log('test/loss', loss.item(), on_step=False, sync_dist=True, 
                 batch_size=self.params.batch_size, on_epoch=True)
        pred_seqs = self.model.greedy_search(X)
        for ii in range(len(pred_seqs)):
            pred_seq = self.tokenizer.remove_special(pred_seqs[ii])
            targ_seq = self.tokenizer.remove_special(Y[ii,:Y_lens[ii]])
            self.test_edit_dist += edit_distance(targ_seq, pred_seq)
            self.test_n_tokens += Y_lens[ii]
            if batch_idx%50==0 and ii==0:
                print(f'TARGET: "{self.tokenizer.decode(targ_seq)}"')
                print(f'PREDICTED: "{self.tokenizer.decode(pred_seq)}"')
        return loss

    def on_test_epoch_end(self):
        wer = self.test_edit_dist/self.test_n_tokens
        self.log('test/wer', wer, on_step=False, 
                 sync_dist=True, on_epoch=True)
        self.test_edit_dist = 0
        self.test_n_tokens = 0

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


class AmModule(LightningModule):
    def __init__(self, 
                 params: argparse.Namespace,
                 vocab_size: int,
                 tokenizer: Tokenizer) -> None:
        super().__init__()
        torch.manual_seed(params.seed)
        self.params = params
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.criterion = SmoothCtcLoss(params.smoothing*1.5)
        self.val_n_tokens = 0
        self.val_edit_dist = 0
        self.test_n_tokens = 0
        self.test_edit_dist = 0
        self.lr = params.lr

    def configure_model(self):
        self.model = RnntTranscriber(self.params, self.vocab_size, individual=True)

    def training_step(self, 
                      batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                      batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        Y = Y[:,1:]
        Y_lens = Y_lens-1
        if self.params.use_eos_token:
            Y_lens = Y_lens-1
        Y_flat = torch.cat([yy[:yy_lens] for yy,yy_lens in zip(Y,Y_lens)])
        Y_hat,Y_hat_lens,_ = self.model(X)
        loss = self.criterion(Y_hat.permute(1,0,2), Y_flat, Y_hat_lens, Y_lens)
        self.log('train/loss', loss.item(), on_step=True, sync_dist=True,
                 batch_size=self.params.batch_size, prog_bar=True)
        if self.params.train_wer:
            edit_dist, n_tokens = 0, 0
            for ii in range(Y_hat.shape[0]):
                pred_seq = torch.argmax(Y_hat[ii,:Y_hat_lens[ii]], dim=-1)
                pred_seq = self.tokenizer.collapse_ctc(pred_seq)
                targ_seq = Y[ii,:Y_lens[ii]]
                edit_dist += edit_distance(targ_seq, pred_seq)
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
        Y = Y[:,1:]
        Y_lens = Y_lens-1
        if self.params.use_eos_token:
            Y_lens = Y_lens-1
        Y_flat = torch.cat([yy[:yy_lens] for yy,yy_lens in zip(Y,Y_lens)])
        Y_hat,Y_hat_lens,_ = self.model(X)
        loss = self.criterion(Y_hat.permute(1,0,2), Y_flat, Y_hat_lens, Y_lens)
        self.log('val/loss', loss.item(), on_step=False, sync_dist=True, 
                 batch_size=self.params.batch_size, on_epoch=True, prog_bar=True)
        if self.params.val_wer:
            for ii in range(Y_hat.shape[0]):
                pred_seq = torch.argmax(Y_hat[ii,:Y_hat_lens[ii]], dim=-1)
                pred_seq = self.tokenizer.collapse_ctc(pred_seq)
                targ_seq = Y[ii,:Y_lens[ii]]
                self.val_edit_dist += edit_distance(targ_seq, pred_seq)
                self.val_n_tokens += Y_lens[ii]
                if batch_idx%50==0 and ii==0:
                    print(f'TARGET: "{self.tokenizer.decode(targ_seq)}"')
                    print(f'PREDICTED: "{self.tokenizer.decode(pred_seq)}"')
        return loss

    def on_validation_epoch_end(self):
        if self.params.val_wer:
            wer = self.val_edit_dist/self.val_n_tokens
            self.log('val/wer', wer, on_step=False, prog_bar=True, 
                    sync_dist=True, on_epoch=True)
            self.val_edit_dist = 0
            self.val_n_tokens = 0

    def test_step(self, 
                  batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                  batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        Y = Y[:,1:]
        Y_lens = Y_lens-1
        if self.params.use_eos_token:
            Y_lens = Y_lens-1
        Y_flat = torch.cat([yy[:yy_lens] for yy,yy_lens in zip(Y,Y_lens)])
        Y_hat,Y_hat_lens,_ = self.model(X)
        loss = self.criterion(Y_hat.permute(1,0,2), Y_flat, Y_hat_lens, Y_lens)
        self.log('test/loss', loss.item(), on_step=False, sync_dist=True, 
                 batch_size=self.params.batch_size, on_epoch=True)
        for ii in range(Y_hat.shape[0]):
            pred_seq = torch.argmax(Y_hat[ii,:Y_hat_lens[ii]], dim=-1)
            pred_seq = self.tokenizer.collapse_ctc(pred_seq)
            targ_seq = Y[ii,:Y_lens[ii]]
            self.test_edit_dist += edit_distance(targ_seq, pred_seq)
            self.test_n_tokens += Y_lens[ii]
            if batch_idx%50==0 and ii==0:
                print(f'TARGET: "{self.tokenizer.decode(targ_seq)}"')
                print(f'PREDICTED: "{self.tokenizer.decode(pred_seq)}"')
        return loss
    
    def on_test_epoch_end(self):
        wer = self.test_edit_dist/self.test_n_tokens
        self.log('test/wer', wer, on_step=False, 
                 sync_dist=True, on_epoch=True)
        self.test_edit_dist = 0
        self.test_n_tokens = 0

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), 1e-5,
                            weight_decay=self.params.wd)
        return opt
        # scheduler = torch.optim.lr_scheduler.LinearLR(
        #     opt,
        #     start_factor=0.1,
        #     end_factor=1.0,
        #     total_iters=5
        # )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     opt,
        #     milestones=[22, 36],
        #     gamma=0.1
        # )
        # return {
        #     'optimizer': opt,
        #     'lr_scheduler': scheduler
        # }


class LmModule(LightningModule):
    def __init__(self, 
                 params: argparse.Namespace,
                 vocab_size: int,
                 tokenizer: Tokenizer) -> None:
        super().__init__()
        torch.manual_seed(params.seed)
        self.params = params
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.dictionary['<blank>'],
                                             label_smoothing=params.smoothing)
        self.val_n_tokens = 0
        self.val_edit_dist = 0
        self.test_n_tokens = 0
        self.test_edit_dist = 0

    def configure_model(self):
        self.model = RnntPredictor(self.params, self.vocab_size, individual=True)

    def training_step(self, 
                      batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                      batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        Y_hat,_ = self.model(Y[:,:-1], None)
        loss = self.criterion(Y_hat.permute(0,2,1), Y[:,1:])
        self.log('train/loss', loss.item(), on_step=True, sync_dist=True,
                 batch_size=self.params.batch_size, prog_bar=True)
        if self.params.train_wer:
            edit_dist, n_tokens = 0, 0
            for ii in range(Y_hat.shape[0]):
                pred_seq = torch.argmax(Y_hat[ii,:Y_lens[ii]-1], dim=-1)
                targ_seq = Y[ii,1:Y_lens[ii]]
                edit_dist += edit_distance(pred_seq, targ_seq)
                n_tokens += len(targ_seq)
                if batch_idx%20==0 and ii==0:
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
        Y_hat,_ = self.model(Y[:,:-1], None)
        loss = self.criterion(Y_hat.permute(0,2,1), Y[:,1:])
        self.log('val/loss', loss.item(), on_step=False, sync_dist=True, 
                 batch_size=self.params.batch_size, on_epoch=True, prog_bar=True)
        if self.params.val_wer:
            for ii in range(Y_hat.shape[0]):
                pred_seq = torch.argmax(Y_hat[ii,:Y_lens[ii]-1], dim=-1)
                targ_seq = Y[ii,1:Y_lens[ii]]
                self.val_edit_dist += edit_distance(pred_seq, targ_seq)
                self.val_n_tokens += len(targ_seq)
                if batch_idx%10==0 and ii==0:
                    print(f'TARGET: "{self.tokenizer.decode(targ_seq)}"')
                    print(f'PREDICTED: "{self.tokenizer.decode(pred_seq)}"')
        return loss

    def on_validation_epoch_end(self):
        if self.params.val_wer:
            wer = self.val_edit_dist/self.val_n_tokens
            self.log('val/wer', wer, on_step=False, prog_bar=True, 
                    sync_dist=True, on_epoch=True)
            self.val_edit_dist = 0
            self.val_n_tokens = 0

    def test_step(self, 
                  batch: tuple[PackedSequence, PackedSequence, torch.Tensor],
                  batch_idx: int) -> Any:
        X,Y,fs = batch
        Y,Y_lens = pad_packed_sequence(Y, batch_first=True)
        Y_hat,_ = self.model(Y[:,:-1], None)
        loss = self.criterion(Y_hat.permute(0,2,1), Y[:,1:])
        self.log('test/loss', loss.item(), on_step=False, sync_dist=True, 
                 batch_size=self.params.batch_size, on_epoch=True)
        for ii in range(Y_hat.shape[0]):
            pred_seq = torch.argmax(Y_hat[ii,:Y_lens[ii]-1], dim=-1)
            targ_seq = Y[ii,1:Y_lens[ii]]
            self.test_edit_dist += edit_distance(pred_seq, targ_seq)
            self.test_n_tokens += len(targ_seq)
            if batch_idx%10==0 and ii==0:
                print(f'TARGET: "{self.tokenizer.decode(targ_seq)}"')
                print(f'PREDICTED: "{self.tokenizer.decode(pred_seq)}"')
        return loss

    def on_test_epoch_end(self):
        wer = self.test_edit_dist/self.test_n_tokens
        self.log('test/wer', wer, on_step=False, 
                 sync_dist=True, on_epoch=True)
        self.test_edit_dist = 0
        self.test_n_tokens = 0

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), 5e-4,
                            weight_decay=self.params.wd)
        return opt