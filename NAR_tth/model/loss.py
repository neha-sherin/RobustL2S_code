import torch
import torch.nn as nn

from torch.nn.modules.loss import CrossEntropyLoss, _Loss, MSELoss, L1Loss
import utils.tools as utils
import numpy as np

class TransformerLoss(L1Loss):   #MSELoss):   #(CrossEntropyLoss):
    def __init__(self, ignore_index=-100, reduction='mean') -> None:
        self.reduction = reduction
        self.ignore_index = ignore_index
        super().__init__(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor,trg_mask) -> torch.Tensor:
        mask = trg_mask==False
        mask = mask.unsqueeze(-1).repeat(1,1,768)
        not_masked_length = mask.to(torch.int).sum()
        _loss = super().forward(input, target)
        _loss *= mask.to(_loss.dtype)
        _loss = _loss.sum() / not_masked_length
        return _loss

    def __call__(self, input: torch.Tensor, target: torch.Tensor,trg_mask) -> torch.Tensor:
        return self.forward(input, target,trg_mask)


class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config,ignore_index=-100, reduction='mean',sigma=0.2):
        super(FastSpeech2Loss, self).__init__()
        self.transformer_loss = TransformerLoss()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, predictions,attn=None):

        (   input_lengths,
            max_input_len,
            target,
            target_lengths,
            max_target_len
        ) = inputs[2:]

        (
            preds,
            src_masks,
            trg_mask,
        ) = predictions


        _loss = self.transformer_loss(preds,target,trg_mask)

        return (_loss,torch.zeros(1),torch.zeros(1),torch.zeros(1))

