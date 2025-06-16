from typing import Optional

import torch
from torch import Tensor

from tsl.nn.utils import maybe_cat_exog


maybe_cat_exog = maybe_cat_exog


def maybe_cat_emb(x: Tensor, emb: Optional[Tensor]):
    if emb is None:
        return x
    if emb.ndim < x.ndim:
        emb = emb[[None] * (x.ndim - emb.ndim)]
    emb = emb.expand(*x.shape[:-1], -1)
    return torch.cat([x, emb], dim=-1)


def maybe_cat_v(u: Tensor, v: Optional[Tensor]):
    return maybe_cat_emb(x=u, emb=v)
