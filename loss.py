# This is the AAM-Softmax loss
import torch, math
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from tools import *

class AAMsoftmax(nn.Module):
    def __init__(self, m, s):

        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, cosine, mask, label=None, supervised=True):

        if supervised == True:
            sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
            loss = self.ce(output, label)
            loss = loss.mean()
        else:
            sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
            loss = self.ce(output, label) * mask
            loss = loss.mean()


        prec1 = accuracy(output.detach(), label.detach(), topk=(1,))[0]

        return loss, prec1
