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
        desired_fs = 16000
        if params.dataset_fs!=desired_fs:
            self.resample = torchaudio.transforms.Resample(params.dataset_fs, desired_fs)
        else:
            self.resample = None

        if params.preemphasis_alpha is not None:
            self.preemphasis = torchaudio.transforms.Preemphasis(params.preemphasis_alpha)
        else:
            self.preemphasis = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample is not None:
            x = self.resample(x)
        if self.preemphasis is not None:
            x = self.Preemphasis(x)

        if self.params.normalization_fn is not None:
            if self.params.normalization_fn == 'power':
                raise NotImplementedError
            elif self.params.normalization_fn == 'amp':
                raise NotImplementedError
            else:
                raise NotImplementedError

        if self.input_feature_type == 'waveform':
            if self.params.augment:
                # Spec -> augment -> Wav
                raise NotImplementedError
            return x
        elif self.input_feature_type == 'melspec':
            if self.params.augment:
                # Spec -> augment -> Melspec
                raise NotImplementedError
            raise NotImplementedError
        elif self.input_feature_type == 'spec':
            if self.params.augment:
                # Spec -> augment
                raise NotImplementedError
            raise NotImplementedError
        else:
            raise NotImplementedError
