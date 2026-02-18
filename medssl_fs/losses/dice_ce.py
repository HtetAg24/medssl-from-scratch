
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceCELoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_ce=1.0, smooth=1e-5):
        super().__init__()
        self.wd = weight_dice
        self.wc = weight_ce
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        if target.dim() == 5 and target.shape[1] == 1:
            target = target[:,0]
        ce = self.ce(logits, target.long())
        probs = torch.softmax(logits, dim=1)
        onehot = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1).long(), 1)
        dims = (0,2,3,4)
        inter = torch.sum(probs * onehot, dims)
        denom = torch.sum(probs + onehot, dims)
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return self.wd*(1-dice.mean()) + self.wc*ce
