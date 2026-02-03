import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence

from utils.processor import AudioProcessor

from typing import Tuple
from argparse import Namespace

from pdb import set_trace


class ResidualGru(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 n_layers: int, 
                 dropout: float=0.2, 
                 bidirectional: bool=False, 
                 return_states: bool=False
                 ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.return_states = return_states
        self.in_projection = nn.Linear(input_dim, hidden_dim)
        self.rnns = [nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional
        ) for _ in range(n_layers)]
        if bidirectional:
            self.projections = [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(n_layers)]

        self.layer_norms = [nn.LayerNorm(hidden_dim) for _ in range(n_layers)]
        self.dropout = nn.Dropout(dropout)

    def forward(self, 
                x: torch.Tensor, 
                in_states: torch.Tensor=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_projection(x)
        out_states = []
        for layer in range(self.n_layers):
            residual = x
            
            if in_states is not None:
                output, state = self.rnns[layer](x, in_states[layer])
            else:
                output, state = self.rnns[layer](x)
            if self.return_states:
                out_states.append(state)
            if self.bidirectional:
                output = self.projections[layer](output)
            output = self.dropout(output)
        
            x = self.layer_norms[layer](output + residual)

        return x, out_states

class CnnFrontend(nn.Module):
    def __init__(self, 
                 input_channels: int=1, 
                 output_channels: int=32,
                 stride: int=2
                )  -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(stride, stride), padding=1),
            nn.BatchNorm2d(output_channels),
            nn.Hardtanh(0, 20, inplace=True),
            
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=(1, 1), padding=1),
            nn.BatchNorm2d(output_channels),
            nn.Hardtanh(0, 20, inplace=True)
        )

    def forward(self, 
                x: torch.Tensor, 
                x_lens: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1) 
        x = self.conv(x)
        batch, channels, time, freq = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch, time, channels * freq)
        return x, (x_lens/self.stride).long()

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
        x = F.avg_pool1d(x.transpose(1,2), kernel_size=self.stride, stride=self.stride).transpose(1,2)
        return x, x_lens


class RnntTranscriber(nn.Module):
    def __init__(self,
                 params: Namespace,
                 vocab_size: int=None,
                 individual: bool=False
                 ) -> None:
        super().__init__()
        self.params = params
        self.individual = individual
        self.processor = AudioProcessor(params)
        if params.time_downsample_factor > 0:
            self.downsampler = TimeDownsampler(params.time_downsample_factor)
        if params.cnn_feat_extractor:
            self.cnn_feat_extractor = CnnFeatureExtractor(params)
        if params.cnn_frontend:
            self.cnn_frontend = CnnFrontend(stride=max([params.time_downsample_factor, 2]))
            rnn_in_dim = int(self.cnn_frontend.output_channels*params.n_mels/self.cnn_frontend.stride)
        else:
            rnn_in_dim = params.n_mels
        self.input_layer_norm = nn.LayerNorm(rnn_in_dim)
        if params.transcriber_type=='gru':
            self.s2s_model = nn.GRU(rnn_in_dim,
                                params.transcriber_s2s_out_dim,
                                params.n_transcriber_layers,
                                batch_first=True,
                                dropout=params.transcriber_s2s_dropout_p,
                                bidirectional=params.transcriber_bidirectional)
        elif params.transcriber_type=='residual_gru':
            self.s2s_model = ResidualGru(rnn_in_dim,
                                params.transcriber_s2s_out_dim,
                                params.n_transcriber_layers,
                                dropout=params.transcriber_s2s_dropout_p,
                                bidirectional=params.transcriber_bidirectional)
        elif params.transcriber_type=='lstm':
            self.s2s_model = nn.LSTM(rnn_in_dim,
                                params.transcriber_s2s_out_dim,
                                params.n_transcriber_layers,
                                batch_first=True,
                                dropout=params.transcriber_s2s_dropout_p,
                                bidirectional=params.transcriber_bidirectional)
        elif params.transcriber_type=='mamba':
            raise NotImplementedError
        self.linear = nn.Linear(params.transcriber_s2s_out_dim, params.joiner_in_dim)
        self.output_layer_norm = nn.LayerNorm(params.joiner_in_dim)
        self.act_out = nn.ReLU()
        if individual:
            self.vocab_project = nn.Linear(params.joiner_in_dim, vocab_size)
            self.smax = nn.LogSoftmax(dim=-1)

    def forward(self, 
                x: PackedSequence,
                state: torch.Tensor=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, x_lens = pad_packed_sequence(x, batch_first=True)
        x, x_lens = self.processor(x, x_lens)
        if self.params.cnn_feat_extractor:
            x, x_lens = self.cnn_feat_extractor(x, x_lens)
        if self.params.time_downsample_factor > 0 and not self.params.cnn_frontend:
            x, x_lens = self.downsampler(x, x_lens)
        else:
            x, x_lens = self.cnn_frontend(x, x_lens)
        x = self.input_layer_norm(x)
        x, state = self.s2s_model(x, state)
        x = self.linear(x)
        x = self.act_out(x)
        x = self.output_layer_norm(x)
        if self.individual:
            x = self.vocab_project(x)
            # blank_bias = torch.zeros_like(x)
            # blank_bias[:,:,0] = -5.0
            # x = x - blank_bias
            x = self.smax(x)
        return x, x_lens, state


class RnntPredictor(nn.Module):
    def __init__(self,
                 params: Namespace,
                 vocab_size: int,
                 individual: bool=False
                 ) -> None:
        super().__init__()
        self.params = params
        self.individual = individual
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
        self.act_out = nn.ReLU()
        self.output_layer_norm = nn.LayerNorm(params.joiner_in_dim)
        self.dropout = nn.Dropout(params.predictor_out_dropout_p)
        if individual:
            self.vocab_project = nn.Linear(params.joiner_in_dim, vocab_size)

    def forward(self, 
                y: torch.Tensor, 
                state: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = self.embedding(y)
        y = self.input_layer_norm(y)
        y, state = self.s2s_model(y, state)
        y = self.linear_out(y)
        y = self.act_out(y)
        y = self.output_layer_norm(y)
        y = self.dropout(y)
        if self.individual:
            y = self.vocab_project(y)
        return y, state

    def step(self,
             y: torch.Tensor,
             state: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.embedding(y)
        y = self.input_layer_norm(y)
        state = self.s2s_model(y, state)[1]
        out = self.linear_out(state)
        out = self.act_out(out)
        out = self.output_layer_norm(out)
        return out, state


class RnntJoiner(nn.Module):
    def __init__(self, 
                 params: Namespace, 
                 vocab_size: int
                 ) -> None:
        super().__init__()
        self.linear = nn.Linear(params.joiner_in_dim, vocab_size)
        self.relu = nn.ReLU()
        self.transcriber_weight = params.transcriber_weight
        self.predictor_weight = params.predictor_weight

    def forward(self,
                x: torch.Tensor, 
                y: torch.Tensor, 
                ) -> torch.Tensor:
        h = self.transcriber_weight*x.unsqueeze(2).contiguous() + \
                self.predictor_weight*y.unsqueeze(1).contiguous()
        h = self.relu(h)
        h = self.linear(h)
        return h


class RnntModel(nn.Module):
    def __init__(self, 
                 params: Namespace,
                 vocab_size: int,
                 blank_idx: int=0,
                 sos_idx: int=1,
                 eos_idx: int=2,
                 transcriber: RnntTranscriber=None,
                 predictor: RnntPredictor=None,
                 max_out_len: int=50
                 ) -> None:
        super().__init__()
        self.params = params
        if transcriber is None:
            self.transcriber = RnntTranscriber(params, vocab_size)
        else:
            self.transcriber = transcriber
            self.transcriber.individual = False
        if predictor is None:
            self.predictor = RnntPredictor(params, vocab_size)
        else:
            self.predictor = predictor
            self.predictor.individual = False
        self.joiner = RnntJoiner(params, vocab_size)
        self.blank_idx = blank_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.max_out_len = max_out_len

    def forward(self, 
                x: PackedSequence, 
                y: torch.Tensor, 
                state: torch.Tensor=None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, x_lens, transcriber_state = self.transcriber(x)
        y, predictor_state = self.predictor(y, state)
        h = self.joiner(x, y)
        return h, x_lens

    def greedy_search(self, 
                      x: torch.Tensor
                      ) -> torch.Tensor:
        with torch.no_grad():
            x, x_lens, transcriber_state = self.transcribe(x)
        batch_size = x.shape[0]
        output = []
        for b in range(batch_size):
            t = 0
            u = 0
            y_hat = [self.sos_idx]
            predictor_state = None
            while t < x_lens[b] and u < self.max_out_len:
                with torch.no_grad():
                    predictor_input = torch.tensor([y_hat[-1]], device=x.device)
                    g_u, predictor_state = self.predict(predictor_input, predictor_state)
                    f_t = x[b,t]
                    h = self.joiner(f_t.unsqueeze(0).unsqueeze(0), g_u.unsqueeze(0))
                    h = nn.functional.log_softmax(h, dim=-1)
                    # h[0,0,0,self.blank_idx] = h[0,0,0,self.blank_idx]-4
                idx = h.max(-1)[1].item()
                if idx==self.blank_idx:
                    t += 1
                else:
                    u += 1
                    y_hat.append(idx)
            y_hat.append(self.eos_idx)
            output.append(torch.Tensor(y_hat).long())
        return output

    def transcribe(self, 
                   x: PackedSequence
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transcriber(x)

    def predict(self,
                y: torch.Tensor,
                state: torch.Tensor):
        return self.predictor.step(y, state)

    def freeze_transcriber(self):
        for p in self.transcriber.parameters():
            p.requires_grad = False

    def freeze_predictor(self):
        for p in self.predictor.parameters():
            p.requires_grad = False

    def unfreeze_transcriber(self):
        for p in self.transcriber.parameters():
            p.requires_grad = True

    def unfreeze_predictor(self):
        for p in self.predictor.parameters():
            p.requires_grad = True