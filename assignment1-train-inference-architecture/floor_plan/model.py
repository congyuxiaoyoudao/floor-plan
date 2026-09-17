import os
import typing
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Int
from einops import rearrange

class Encoder(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_model: int,
            device: str = None,
            dtype: str = None
    ):
        super().__init__()
        self.mlp = nn.Linear(d_in, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class Decoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_out: int,
            device: str = None,
            dtype: str = None
    ):
        super().__init__()
        self.mlp = nn.Linear(d_model, d_out, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class MLPAutoEncoder(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_model: int,
            d_out: int,
            device: str = None,
            dtype: str = None
    ):
        super().__init__()
        self.encoder = Encoder(
            d_in=d_in,
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.decoder = Decoder(
            d_model=d_model,
            d_out=d_out,
            device=device,
            dtype=dtype
        )

    def forward(self, x: Int[Tensor, "batch seq_len dim"]) -> Tensor:
        x = rearrange(x, "batch seq_len dim -> batch (seq_len dim)", dim=2)
        out = self.decoder(self.encoder(x))
        out = rearrange(out, "batch (seq_len dim) -> batch seq_len dim", dim=2)
        return out


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """
    Dump all the state from the model, optimizer and iteration into 
    the file-like object out.
    """
    model_state = model.state_dict()
    optim_state = optimizer.state_dict()

    checkpoint = {
        "model": model_state,
        "optimizer": optim_state,
        "iteration": iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]