import torch
from torch import nn

def iou(pr, gt, eps=1e-7, threshold=None, activation='sigmoid'):

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = troch.nn.Softmax2d()
    else:
        raise NotImplementedError(
                "Activation implemented for sigmoid and softmax2d"
                )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()
            
    intersection = torch.sum(gt * pr)
    union = torch(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None,
            activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce
