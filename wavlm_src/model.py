import argparse
import torch
from torch import nn
from transformers import AutoModel

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
        self.linear = nn.Linear(self.wavlm.config.output_hidden_size, vocab_size)
        self.log_smax = nn.LogSoftmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wavlm(x)
        x = self.linear(x)
        x = self.log_smax(x)
        return x
    

if __name__=='__main__':
    from params import get_params
    params = get_params()
    module = WavlmModel(params)