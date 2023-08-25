import os
import cv2
import csv
import numpy as np
from pytorch_msssim import ms_ssim, ssim

import torch
import torch.nn as nn
import torch.nn.functional as F

def list2str(seq):
    string = ''
    for i in seq:
        string += str(i)
    return string


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2.0, dtype=torch.float32)

    if mse == 0:
        return torch.tensor([100.0])

    PIXEL_MAX = 255.0

    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))

def get_imagenet_list(path):
    fns = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            fns.append(row[0])
    
    return fns

def complex_sig(shape, device):
        sig_real = torch.randn(*shape)
        sig_imag = torch.randn(*shape)
        return (torch.complex(sig_real, sig_imag)/np.sqrt(2)).to(device)

def pwr_normalize(sig):
    _, num_ele = sig.shape[0], torch.numel(sig[0])
    pwr_sig = torch.sum(torch.abs(sig)**2, dim=-1)/num_ele
    sig = sig/torch.sqrt(pwr_sig.unsqueeze(-1))

    return sig

def np_to_torch(img):
    img = np.swapaxes(img, 0, 1)  # w, h, c
    img = np.swapaxes(img, 0, 2)  # c, h, w
    return torch.from_numpy(img).float()


def to_chan_last(img):
    img = img.transpose(1, 2)
    img = img.transpose(2, 3)
    return img


def as_img_array(image):
    image = image.clamp(0, 1) * 255.0
    return torch.round(image)


def freeze_params(nets):
    for p in nets:
        p.requires_grad = False

def freeze_given_params(nets):
    # freeze all parameters except the 'gen_mask_modules'
    for p in nets.parameters():
        p.requires_grad = False

    # activate the modules in enc
    for p in nets.enc.gen_mask_module.parameters():
        p.requires_grad = True
    
    for p in nets.enc.final_layer.parameters():
        p.requires_grad = True
    
    # activate the modules in dec
    for p in nets.dec.gen_mask_module.parameters():
        p.requires_grad = True

    for p in nets.dec.linear_proj.parameters():
        p.requires_grad = True

def reactive_params(nets):
    for p in nets:
        p.requires_grad = True

def save_nets(job_name, nets, epoch):
    path = '{}/{}.pth'.format('models', job_name)

    if not os.path.exists('models'):
        print('Creating model directory: {}'.format('models'))
        os.makedirs('models')

    torch.save({
        'jscc_model': nets.state_dict(),
        'epoch': epoch
    }, path)


def load_weights(job_name, nets, device = torch.device('cuda'), path = None):
    if path == None:
        path = '{}/{}.pth'.format('models', job_name)

    cp = torch.load(path, map_location = device)
    nets.load_state_dict(cp['jscc_model'], strict=False)
    
    return cp['epoch']

def calc_loss(prediction, target, loss):
    if loss == 'l2':
        loss = F.mse_loss(prediction, target)
    elif loss == 'msssim':
        loss = 1 - ms_ssim(prediction, target, win_size=3,
                           data_range=1, size_average=True)
    elif loss == 'ssim':
        loss = 1 - ssim(prediction, target,
                        data_range=1, size_average=True)
    else:
        raise NotImplementedError()
    return loss


def calc_psnr(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = psnr(original, compare)
        metric.append(val)
    return metric


def calc_msssim(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        # val = msssim(original, compare)
        val = ms_ssim(original, compare, data_range=255,
                      win_size=3, size_average=True)
        metric.append(val)
    return metric


def calc_ssim(predictions, targets):
    metric = []
    for i, pred in enumerate(predictions):
        original = as_img_array(targets[i])
        compare = as_img_array(pred)
        val = ssim(original, compare, data_range=255,
                   size_average=True)
        metric.append(val)
    return metric



def save_frames(frame, fns, out_dir):
    if not os.path.exists(out_dir):
        print('Creating output directory: {}'.format(out_dir))
        os.makedirs(out_dir)

    for idx, pred in enumerate(frame):
        pred = as_img_array(pred.cpu().numpy())
        pred = np.squeeze(pred, axis=0)
        flag = cv2.imwrite(out_dir + '/{}.png'.format(fns[idx][0]), pred)
        assert flag




def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'