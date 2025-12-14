import torch
from torch import nn, autograd, optim, Tensor, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from util import *
import torch.distributed as dist

def flat(mask):
    batch_size = mask.shape[0]
    h = 28
    mask = F.interpolate(mask, size=(int(h), int(h)), mode='bilinear')
    x = mask.view(batch_size, 1, -1).permute(0, 2, 1)
    # print(x.shape)  b 28*28 1
    g = x @ x.transpose(-2, -1)  # b 28*28 28*28
    g = g.unsqueeze(1)  # b 1 28*28 28*28
    return g


def att_loss(pred, mask, p4, p5):
    g = flat(mask)
    np4 = torch.sigmoid(p4.detach())
    np5 = torch.sigmoid(p5.detach())
    p4 = flat(np4)
    p5 = flat(np5)
    w1 = torch.abs(g - p4)
    w2 = torch.abs(g - p5)
    w = (w1 + w2) * 0.5 + 1
    attbce = F.binary_cross_entropy_with_logits(pred, g, weight=w * 1.0, reduction='mean')
    return attbce


def bce_iou_loss(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask, size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def reduce_mean(tensor, nprocs):  # 用于平均所有gpu上的运行结果，比如loss
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

def structure_loss(pred, mask, epsilon=0):
    weit  = 1+5*torch.abs(F.max_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    # print(weit)
    # weit_m  = 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    pt = torch.sigmoid(pred)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='mean') + epsilon * (1 - pt)

    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))


    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-inter/(union-inter)
    # bound = (torch.exp(wbce) + torch.exp(wiou))
    return (wbce +  wiou).mean()
    # return ((((torch.exp(wbce)/bound)*wbce) + ((torch.exp(wiou)/bound)*wiou))).mean()

# def Loss(preds, target, config):
#     # preds: Dict type. Customarily, preds['final'] is the final prediction without sigmoid.
#     bce = nn.BCEWithLogitsLoss(reduction='none')
#     fnl_loss = bce(preds['final'], target.gt(0.5).float()).mean()

#     return fnl_loss

def Loss(preds, target, config):
    #loss = bce_ssim_loss(torch.sigmoid(preds['final']), target)
    loss = 0
    loss1u = structure_loss(preds['sal'][0], target)
    loss2u = structure_loss(preds['sal'][1], target)

    loss2r = structure_loss(preds['sal'][2], target)
    loss3r = structure_loss(preds['sal'][3], target)
    loss = (loss1u + loss2u) / 2 + loss2r / 2 + loss3r / 4 
    # for pred in preds['sal']:
    #     loss += bce_ssim_loss(torch.sigmoid(pred), target)

    return loss

