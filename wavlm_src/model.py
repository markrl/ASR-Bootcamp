import argparse
import torch
from torch import nn
from transformers import AutoModel


class WavlmModel(nn.Module):
    def __init__(self, params: argparse.Namespace) -> None:
        self.wavlm = AutoModel.from_pretrained('microsoft/wavlm-large', 
                                                trust_remote_code=True)
        self.linear = nn.Linear()
        self.log_smax = nn.LogSoftmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wavlm(x)
        x = self.linear(x)
        x = self.log_smax(x)
        return x