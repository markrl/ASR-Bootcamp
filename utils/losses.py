import torch
import torch.nn as nn

from argparse import Namespace

from pdb import set_trace


class RnntLoss(nn.Module):
    def __init__(self,
                 blank_idx: int = 0,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.blank_idx = blank_idx
        self.reduction = reduction
        self.log_smax = nn.LogSoftmax(dim=1)
    
    def forward(self, h, targets, input_lens, target_lens):
        batch_size, n_frames, n_tokens, vocab_size = h.shape
        h = self.log_smax(h)
        log_alpha = torch.zeros(batch_size, n_frames, n_tokens).to(h.device)
        for t in range(n_frames):
            for u in range(n_tokens):
                if u == 0:
                    if t == 0:
                        log_alpha[:, t, u] = 0.

                    else: #t > 0
                        log_alpha[:, t, u] = log_alpha[:, t-1, u] + h[:, t-1, 0, self.blank_idx] 
                        
                else: #u > 0
                    if t == 0:
                        log_alpha[:, t, u] = log_alpha[:, t,u-1] + torch.gather(h[:, t, u-1], dim=1, index=targets[:,u-1].view(-1,1) ).reshape(-1)
                    
                    else: #t > 0
                        log_alpha[:, t, u] = torch.logsumexp(torch.stack([
                        log_alpha[:, t-1, u] + h[:, t-1, u, self.blank_idx],
                        log_alpha[:, t, u-1] + torch.gather(h[:, t, u-1], dim=1, index=targets[:,u-1].view(-1,1) ).reshape(-1)]), dim=0)
        log_probs = [log_alpha[b, input_lens[b]-1, target_lens[b]-1] + h[b, input_lens[b]-1, target_lens[b]-1, self.blank_idx] for b in range(batch_size)]
        neg_log_probs = -torch.stack(log_probs, dim=0)

        if self.reduction=='none':
            return neg_log_probs
        elif self.reduction=='mean':
            return torch.mean(neg_log_probs)
        elif self.reduction=='sum':
            return torch.sum(neg_log_probs)


class CtcLoss(nn.Module):
    def __init__(self, 
                 params: Namespace) -> None:
        super().__init__()
        self.params = params

    def forward(self, predictions, targets, prediction_lens, target_lens):
        return