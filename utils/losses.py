import torch
import torch.nn as nn

from argparse import Namespace

class RnntLoss(nn.Module):
    def __init__(self,
                 blank: int = 0,
                 sos: int = 0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.blank = blank
        self.sos = sos
        self.reduction = reduction
        self.log_smax = nn.LogSoftmax(dim=1)
    
    def forward(self, h, targets, h_lens, target_lens):
        batch_size, n_frames, n_tokens, vocab_size = h.shape
        h = self.log_smax(h)
        alphas = torch.zeros(batch_size, n_frames, n_tokens+1)
        alphas[:,0,0] = 1

        probs = 0
        if self.reduction=='none':
            return probs
        elif self.reduction=='mean':
            return torch.mean(probs)
        elif self.reduction=='sum':
            return torch.sum(probs)


class CtcLoss(nn.Module):
    def __init__(self, 
                 params: Namespace) -> None:
        super().__init__()
        self.params = params

    def forward(self, predictions, targets, prediction_lens, target_lens):
        return