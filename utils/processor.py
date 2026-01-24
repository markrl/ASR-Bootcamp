import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

from pdb import set_trace

class AudioProcessor(nn.Module):
    def __init__(self, 
                 params: argparse.Namespace
                 ) -> None:
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

        if params.input_feature_type in ['spec', 'melspec'] or params.augment:
            self.stft = T.Spectrogram(n_fft=params.n_fft,
                                      power=None)
            self.istft = T.InverseSpectrogram(n_fft=params.n_fft)
        if params.input_feature_type == 'melspec':
            self.mel_scale = T.MelScale(n_mels=params.n_mels,
                                        sample_rate=params.dataset_fs,
                                        n_stft=params.n_fft//2+1)
        self.augment = False
        self.time_mask = T.TimeMasking(time_mask_param=params.time_mask_param)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=params.freq_mask_param)
        self.random_shift = int(params.max_shift_dur*params.dataset_fs/1000)

    def _get_new_lens(self, 
                      x_lens: torch.Tensor
                      ) -> torch.Tensor:
        x_lens = torch.ceil(x_lens/self.stft.hop_length)
        return x_lens.long()

    def forward(self, 
                x: torch.Tensor,
                x_lens: torch.Tensor
                ) -> torch.Tensor:
        if self.resample is not None:
            x = self.resample(x)
        if self.preemphasis is not None:
            x = self.Preemphasis(x)

        if self.random_shift > 0 and self.training:
            random_shift = torch.randint(self.random_shift, (1,)).item()
            x = torch.cat((torch.zeros(x.shape[0], random_shift).to(x.device), x), dim=-1)
            x_lens += random_shift

        if self.params.center or self.params.normalization_fn is not None:
            x = x.T
            if self.params.center:
                x = x-torch.mean(x, dim=0)

            if self.params.normalization_fn is not None:
                if self.params.normalization_fn == 'power':
                    powers = torch.sum(x**2, dim=0)
                    x = x/torch.sqrt(powers)
                elif self.params.normalization_fn == 'amp':
                    maxs = torch.max(torch.abs(x), dim=0)[0]
                    x = x/maxs
                elif self.params.normalization_fn == 'std':
                    stds = torch.std(x, dim=0)
                    x = x/stds
                else:
                    raise NotImplementedError
            x = x.T

        if self.input_feature_type == 'waveform':
            if self.augment and self.training and torch.rand(1).item()<self.params.p_augment:
                # Spec -> augment -> Wav
                spec = self.stft(x)
                mask = torch.ones(spec.shape).to(spec.device)
                mask = self.time_mask(mask)
                mask = self.freq_mask(mask)
                spec = mask*spec
                x = self.istft(spec)
        elif self.input_feature_type == 'melspec':
            x = torch.abs(self.stft(x))**2
            if self.augment and self.training and torch.rand(1).item()<self.params.p_augment:
                # Spec -> augment -> Melspec
                x = self.time_mask(x)
                x = self.freq_mask(x)
            x = self.mel_scale(x)
            x = x.permute(0,2,1)
            x_lens = self._get_new_lens(x_lens)
        elif self.input_feature_type == 'spec': # *complex* spectrogram
            x = self.stft(x)
            if self.augment and self.training and torch.rand(1).item()<self.params.p_augment:
                # Spec -> augment
                mask = torch.ones(spec.shape).to(spec.device)
                mask = self.time_mask(mask)
                mask = self.freq_mask(mask)
                x = mask*x
            x = x.permute(0,2,1)
            x_lens = self._get_new_lens(x_lens)
        elif self.input_feature_type == 'powerspec':
            x = self.stft(x)**2
            if self.augment and self.training and torch.rand(1).item()<self.params.p_augment:
                # Spec -> augment
                x = self.time_mask(x)
                x = self.freq_mask(x)
            x = x.permute(0,2,1)
            x_lens = self._get_new_lens(x_lens)
        else:
            raise NotImplementedError
            
        return x, x_lens

    def start_augment(self) -> None:
        self.augment = True

    def stop_augment(self) -> None:
        self.augment = False
