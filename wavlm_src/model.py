import argparse
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from transformers import AutoModel

from utils.processor import AudioProcessor

from pdb import set_trace


class WavlmModel(nn.Module):
    def __init__(self, 
                 params: argparse.Namespace,
                 vocab_size: int) -> None:
        super().__init__()
        self.params = params
        self.processor = AudioProcessor(params)
        self.wavlm = AutoModel.from_pretrained('microsoft/wavlm-large', 
                                                trust_remote_code=True)
        if params.freeze_fm:
            self.freeze_fm()
        if params.freeze_extractor:
            self.freeze_feat_extractor()
        self.linear = nn.Linear(self.wavlm.config.output_hidden_size, vocab_size)
        self.dropout = nn.Dropout(params.logit_dropout_p)
        self.log_smax = nn.LogSoftmax(dim=-1)

    def freeze_fm(self):
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def freeze_feat_extractor(self):
        self.wavlm.feature_extractor._freeze_parameters()

    def forward(self, x: PackedSequence) -> tuple[torch.Tensor, torch.Tensor]:
        x,x_lens = pad_packed_sequence(x, batch_first=True)
        x = self.processor(x)
        mask = self.generate_mask(x.shape, x_lens).to(x.device)
        x = self.wavlm(input_values=x, 
                       attention_mask=mask).last_hidden_state
        x_lens = self.wavlm._get_feat_extract_output_lengths(x_lens)
        x = self.linear(x)
        x = self.log_smax(x)
        return x, x_lens

    def generate_mask(self, x_size, x_lens: torch.Tensor) -> torch.tensor:
        mask = torch.zeros(x_size)
        for ii in range(x_size[0]):
            mask[ii,:x_lens[ii]] = 1
        return mask
    

if __name__=='__main__':
    from params import get_params
    params = get_params()
    module = WavlmModel(params)