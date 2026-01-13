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
                 vocab_size: int,
                 blank_idx: int=0,
                 sos_idx: int=1
                 ) -> None:
        super().__init__()
        self.params = params
        self.processor = AudioProcessor(params)
        self.transcriber = RnntTranscriber(params)
        self.predictor = RnntPredictor(params)
        self.joiner = RnntJoiner(params, vocab_size)
        self.blank_idx = blank_idx
        self.sos_idx = sos_idx

    def forward(self, 
                x: PackedSequence, 
                y: PackedSequence, 
                state: torch.Tensor=None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        y, y_lens = pad_packed_sequence(y, batch_first=True)
        x = self.processor(x)
        x_lens = torch.clamp(x_lens, 0, x.shape[-1])
        x, x_lens = self.transcriber(x, x_lens)
        y, y_lens, state = self.predictor(y, y_lens, state)
        h = self.joiner(x, y)
        return h, x_lens, y_lens, state

    def greedy_search(self, 
                      x: torch.Tensor
                      ) -> torch.Tensor:
        x, x_lens = self.transcribe(x)
        batch_size = x.shape[0]
        max_out_len = 50
        output = []
        for b in range(batch_size):
            t = 0
            u = 0
            y_hat = [self.sos_idx]
            state = None
            while t < x_lens[b] and u < max_out_len:
                predictor_input = torch.tensor([y_hat[-1]], device=x.device)
                g_u, state = self.predict(predictor_input, state)
                f_t = x[b,t]
                h = self.joiner(f_t, x_lens[b])
                idx = h.max(-1)[1].item()
                if idx==self.blank_idx:
                    t += 1
                else:
                    u += 1
                    y_hat.append(idx)
            output.append(torch.Tensor(y_hat[1:]))
    return output

    def transcribe(self, 
                   x: PackedSequence
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        return self.transcriber(x, x_lens)

    def predict(self,
                y: torch.Tensor,
                state: torch.Tensor):
        


class CnnFeatureExtractor(nn.Module):
    def __init__(self,
                 params: Namespace
                 ) -> None:
        self.k_size = params.n_fft
        self.stride = int(params.n_fft/2)
        self.cnn = nn.Conv1D(in_channels=1,
                             out_channels=params.n_mels,
                             kernel_size=self.k_size,
                             stride=self.stride)
        
    def _get_new_lens(self, 
                      lens: torch.Tensor
                      ) -> torch.Tensor:
        return torch.floor((lens - self.k_size)/self.stride + 1).long()
        
    def forward(self, 
                x: torch.Tensor, 
                x_lens: torch.Tensor
                ) -> None:
        x = self.cnn(x)
        x_lens = self._get_new_lens(x)
        return x, x_lens


class TimeDownsampler(nn.Module):
    def __init__(self,
                 downsample_time_factor: int
                 ) -> None:
        super().__init__()
        self.stride = downsample_time_factor

    def forward(self, 
                x: torch.Tensor, 
                x_lens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Think about aliasing problems with this
        x_lens = (x_lens/self.stride).long()
        x = x[:, :self.stride:]
        return x, x_lens


class RnntTranscriber(nn.Module):
    def __init__(self,
                 params: Namespace
                 ) -> None:
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
                state: torch.Tensor=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
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
                 vocab_size: int
                 ) -> None:
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
                state: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.embedding(y)
        y = self.input_layer_norm(y)
        y, state = self.s2s_model(y, state)[:2]
        y = self.linear_out(y)
        y = self.output_layer_norm(y)
        y = self.dropout(y)
        return y, y_lens, state

    def step(self,
             y: torch.Tensor,
             state: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.input_layer_norm(y)
        state = self.s2s_model(y, state)[1]
        out = self.linear_out(state)
        out = self.output_layer_norm(y)
        return out, state
        

class RnntJoiner(nn.Module):
    def __init__(self, 
                 params: Namespace, 
                 vocab_size: int
                 ) -> None:
        super().__init__()
        self.linear = nn.Linear(params.joiner_in_dim, vocab_size)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                x: torch.Tensor, 
                y: torch.Tensor, 
                ) -> torch.Tensor:
        h = x.unsqueeze(2) + y.unsqueeze(1)
        h = self.activation(h)
        h = self.linear(h)
        h = self.softmax(h)
        return h