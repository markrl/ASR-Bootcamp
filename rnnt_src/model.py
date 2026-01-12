import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from transformers import AutoModel

from utils.processor import AudioProcessor

from typing import Tuple
from argparse import Namespace

from pdb import set_trace


class RnntModel(nn.Module):
    def __init__(self, 
                 params: Namespace,
                 vocab_size: int) -> None:
        super().__init__()
        self.params = params
        self.processor = AudioProcessor(params)
        self.transcriber = None
        self.predictor = RnntPredictor(params)
        self.joiner = RnntJoiner(params, vocab_size)

    def forward(self, 
                x: PackedSequence, 
                y: PackedSequence, state=None):
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        y, y_lens = pad_packed_sequence(y, batch_first=True)
        x = self.processor(x)
        x_lens = torch.clamp(x_lens, 0, x.shape[-1])
        x, x_lens = self.transcriber(x, x_lens)
        y, y_lens, state = self.predictor(y, y_lens, state)
        y_hat, x_lens, y_lens = self.joiner(x, x_lens, y, y_lens)
        return y_hat, x_lens, y_lens, state

    def transcribe(self, x):
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        return self.transcriber(x, x_lens)

    def transcribe_streaming(self, x):
        raise NotImplementedError


class CnnFeatureExtractor(nn.Module):
    def __init__(self,
                 params: Namespace) -> None:
        self.k_size = params.n_fft
        self.stride = int(params.n_fft/2)
        self.cnn = nn.Conv1D(in_channels=1,
                             out_channels=params.n_mels,
                             kernel_size=self.k_size,
                             stride=self.stride)
        
    def _get_new_lens(self, lens):
        return torch.floor((lens - self.k_size)/self.stride + 1).long()
        
    def forward(self, x, x_lens) -> None:
        x = self.cnn(x)
        x_lens = self._get_new_lens(x)
        return x, x_lens


class TimeDownsampler(nn.Module):
    def __init__(self,
                 downsample_time_factor: int) -> None:
        super().__init__()
        self.stride = downsample_time_factor

    def forward(self, x, x_lens):
        # Think about aliasing problems with this
        x_lens = (x_lens/self.stride).long()
        x = x[:, :self.stride:]
        return x, x_lens


class RnntTranscriber(nn.Module):
    def __init__(self,
                 params: Namespace) -> None:
        super().__init__()
        self.params = params
        if params.cnn_feat_extractor:
            self.cnn_feat_extractor = CnnFeatureExtractor(params)
        self.input_layer_norm = nn.LayerNorm(params.n_mel)
        if params.transcriber_type=='gru':
            self.s2s_model = nn.GRU(params.n_mel,
                                params.transcriber_s2s_out_dim,
                                params.n_transcriber_layers,
                                batch_first=True,
                                dropout=params.transcriber_s2s_dropout_p,
                                bidirectional=params.transcriber_bidirectional)
        elif params.transcriber_type=='lstm':
            self.s2s_model = nn.LSTM(params.n_mel,
                                params.transcriber_s2s_out_dim,
                                params.n_transcriber_layers,
                                batch_first=True,
                                dropout=params.transcriber_s2s_dropout_p,
                                bidirectional=params.transcriber_bidirectional)
        elif params.transcriber_type=='mamba':
            raise NotImplementedError
        self.linear = nn.Linear(params.transcriber_s2s_out_dim, params.transcriber_out_dim)

    def forward(self, 
                x: torch.Tensor, 
                x_lens: torch.Tensor,
                state: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.params.cnn_feat_extractor:
            x, x_lens = self.cnn_feat_extractor(x, x_lens)
        if self.params.downsample_time_factor > 0:
            x, x_lens = self.time_downsampler(x, x_lens)
        x = self.input_layer_norm(x)
        x = self.s2s_model(x, state)
        x = self.linear(x)
        return x, x_lens

class RnntPredictor(nn.Module):
    def __init__(self,
                 params: Namespace,
                 vocab_size: int) -> None:
        super().__init__()
        self.params = params
        self.embedding = nn.Embedding(vocab_size, params.embedding_dim)
        self.input_layer_norm = nn.LayerNorm(params.embedding_dim)
        if params.predictor_type=='gru':
            self.s2s_model = nn.GRU(params.embedding_dim,
                                params.predictor_s2s_out_dim,
                                params.n_predictor_layers,
                                batch_first=True,
                                dropout=params.predictor_s2s_dropout_p)
        elif params.predictor_type=='lstm':
            self.s2s_model = nn.LSTM(params.embedding_dim,
                                params.predictor_s2s_out_dim,
                                params.n_predictor_layers,
                                batch_first=True,
                                dropout=params.predictor_s2s_dropout_p)
        elif params.predictor_type=='mamba':
            raise NotImplementedError
        self.linear_out = nn.Linear(params.predictor_s2s_out_dim, params.joiner_in_dim)
        self.output_layer_norm = nn.LayerNorm(params.joiner_in_dim)
        self.dropout = nn.Dropout(params.predictor_out_dropout_p)

    def forward(self, 
                y: torch.Tensor, 
                y_lens: torch.Tensor, 
                state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.embedding(y)
        y = self.input_layer_norm(y)
        y, state = self.s2s_model(y, state)[:2]
        return y, y_lens, state


class RnntJoiner(nn.Module):
    def __init__(self, 
                 params: Namespace, 
                 vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(params.joiner_in_dim, vocab_size)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                x: torch.Tensor, 
                x_lens: torch.Tensor,
                y: torch.Tensor, 
                y_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = x.unsqueeze(2) + y.unsqueeze(1)
        xy = self.activation(xy)
        xy = self.linear(xy)
        xy = self.softmax(xy)
        return xy, x_lens, y_lens