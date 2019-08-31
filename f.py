
from __future__ import division

import warnings
import math
import types
from torch.nn import functional as F
import torch
from torch._C import _infer_size, _add_docstr
torch.nn.MultiLabelSoftMarginLoss
@torch._jit_internal.weak_script
def multilabel_soft_margin_loss(input, target, weight=None, size_average=None,
                                reduce=None, reduction='mean'):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str) -> Tensor
    r"""multilabel_soft_margin_loss(input, target, weight=None, size_average=None) -> Tensor

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    loss = -(target * torch.log(input) + (1 - target) *  torch.log(-input))

    if weight is not None:
        loss = loss * torch.jit._unwrap_optional(weight)
    loss = loss.sum(dim=1) / input.size(1)  # only return N loss values
    #loss = loss.sum(dim=1)
    if reduction == 'none':
        ret = loss
    elif reduction == 'mean':
        ret = loss.mean()
    elif reduction == 'sum':
        ret = loss.sum()
    else:
        ret = input
        raise ValueError(reduction + " is not valid")
    return ret

def focal_cross_entropy(input, target, weight=None, ignore_index=-100,reduction='mean'):
    input=torch.mul(torch.mul((1-F.softmax(input, 1)),(1-F.softmax(input, 1))),(F.log_softmax(input, 1)))
    return  F.nll_loss(input, target, weight, None, ignore_index, None, reduction)

if __name__ == '__main__':
    input=[[0.4,0.9]]
    input=torch.tensor(input,dtype=torch.float)
    target=[0]
    target=torch.tensor(target,dtype=torch.long)
    print(F.softmax(input, 1))
    print((1-F.softmax(input, 1)))
    print(torch.mul((1-F.softmax(input, 1)),(1-F.softmax(input, 1))))
    print(F.log_softmax(input, 1))
    print(F.cross_entropy(input,target))
    print(focal_cross_entropy(input,target))