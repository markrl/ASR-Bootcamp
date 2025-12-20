import argparse
import torch
import torch.nn as nn
import torchaudio

class AudioProcessor(nn.Module):
    def __init__(self, params: argparse.Namespace) -> None:
        super().__init__()
        self.params = params
        self.input_feature_type = params.input_feature_type
        self.normalization_fn = params.normalization_fn

        if params.preemphasis_alpha is not None:
            self.preempasis = torchaudio.transforms.Preemphasis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preemphasis is not None:
            x = torchaudio.lfilter(x)

        if self.params.normalization_fn is not None:
            if self.params.normalization_fn == 'power':
                raise NotImplementedError
            elif self.params.normalization_fn == 'amp':
                raise NotImplementedError

        if self.input_feature_type == 'waveform':
            return x
