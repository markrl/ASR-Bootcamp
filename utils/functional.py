import torch
import torch.nn as nn

from numpy.typing import ArrayLike
from argparse import Namespace

def edit_distance(s1: ArrayLike, s2: ArrayLike) -> int:
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m+1):
        for j in range(n+1):
            if i==0:
                dp[i][j] = j
            elif j==0:
                dp[i][j] = i
            else:
                if s1[i-1]==s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[-1][-1]

class RnntLoss(nn.Module):
    def __init__(self, params: Namespace) -> None:
        self.super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rnnt_loss()

def rnnt_loss(x: torch.Tensor):
    return x