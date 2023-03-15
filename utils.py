import os
import random

import numpy as np
import torch
from torch.autograd import Variable


#### General purpose
def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def safe_mkdir(path):
    """Create a directory if there isn't one already."""
    try:
        os.makedirs(path)
    except OSError:
        pass


#### Data preprocessing
def prepare(datalist, device):
    return [data.float().to(device) for data in datalist]


def contrast_normalization(datalist, std_bias=0):
    for i in range(len(datalist)):
        m = torch.mean(datalist[i].float(), dim=(2, 3), keepdims=True)
        s = torch.std(datalist[i].float(), dim=(2, 3), keepdims=True)
        datalist[i] = (datalist[i] - m) / (s + std_bias)

    return datalist


def clone_(datalist):
    return [data.clone() for data in datalist]


def generate_easy_flow(batch_size=64, shape=128, range=1):
    # Return random shift

    flow = torch.zeros(batch_size, 2, shape, shape)
    flow = flow + torch.randint(-range, range + 1, (batch_size, 2, 1, 1))

    return flow


def roll_batch_slow(batch, shifts):
    # batch:  b, 1, h, w
    # shifts: b, 2
    for i in range(len(shifts)):
        batch[i] = torch.roll(batch[i], tuple(shifts[i].tolist()), dims=(-2, -1))
    return batch


def roll_batch(batch, shifts):
    # batch:  b, 1, h, w
    # shifts: b, 2
    b, _, h, w = batch.size()
    shift_y, shift_x = shifts[:, 0], shifts[:, 1]

    # Calculate the rolled coordinates
    idx_y = (torch.arange(h).view(1, h, 1) - shift_y.view(b, 1, 1)) % h
    idx_x = (torch.arange(w).view(1, 1, w) - shift_x.view(b, 1, 1)) % w

    # Use advanced indexing to obtain rolled_batch
    rolled_batch = batch[torch.arange(b)[:, None, None], :, idx_y, idx_x]

    # Rearrange the dimensions to match the input batch
    rolled_batch = rolled_batch.permute(0, 3, 1, 2)

    return rolled_batch


def crop_border(datalist, cropborder=1):
    for i in range(len(datalist)):
        datalist[i] = datalist[i][..., cropborder:-cropborder, cropborder:-cropborder]

    return datalist


def self_registered(im, csr):
    # bands registration
    im_r = im.clone().cpu()
    imT = contrast_normalization([im])[0]
    concat = torch.cat((imT[1:2].expand(3, -1, -1, -1), imT[[0, 2, 3]]), 1)
    flow = csr(concat).cpu().detach()  # 3, 2, h, w
    im_r[[0, 2, 3]] = warp(
        im[[0, 2, 3]].cpu(), flow.cpu(), mode="bicubic", padding_mode="reflection"
    )
    return im_r.cpu().numpy().astype(np.uint16)  # , flow.numpy()


def super_resolve(im, rec):
    sr = rec(im / 400.0)
    return (sr.detach().cpu().numpy() * 400).astype(np.uint16)


#####
#### Warping
def create_grid(flo):
    B, _, H, W = flo.shape
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = grid.to(flo.device)
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    return vgrid


def warp(x, flo, mode="bilinear", padding_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    if torch.sum(flo * flo) == 0:
        return x

    else:
        B, _, H, W = x.size()
        vgrid = create_grid(flo)
        output = torch.nn.functional.grid_sample(
            x, vgrid, align_corners=True, mode=mode, padding_mode=padding_mode
        )

        return output
