import torch
import torch.nn as nn
import torch.nn.functional as F


class TVL1(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVL1, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        count_h = self._tensor_size(x[..., 1:, :])
        count_w = self._tensor_size(x[..., :, 1:])

        h_tv = torch.abs((x[..., 1:, :] - x[..., : h_x - 1, :])).sum()
        w_tv = torch.abs((x[..., :, 1:] - x[..., :, : w_x - 1])).sum()
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[-3] * t.size()[-2] * t.size()[-1]


class WarpedLoss(nn.Module):
    def __init__(self, device, p=1):
        super(WarpedLoss, self).__init__()
        if p == 1:
            self.criterion = nn.L1Loss(reduction="mean")  # change to reduction = 'mean'
        if p == 2:
            self.criterion = nn.MSELoss(reduction="mean")
        self.device = device

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        if torch.sum(flo * flo) == 0:
            return x
        else:
            B, _, H, W = x.size()

            # mesh grid
            xx = torch.arange(0, W, device=self.device).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H, device=self.device).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            vgrid = grid + flo.to(self.device)

            # scale grid to [-1,1]
            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            vgrid = vgrid.permute(0, 2, 3, 1)
            output = F.grid_sample(
                x, vgrid, align_corners=True, mode="bicubic", padding_mode="reflection"
            )
            return output

    def doublewarp(self, x, flo1, flo2):
        """
        warp an image/tensor (im3) back to im2 then back to im1, according to the optical flow flo2 and flo1
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """

        B, _, H, W = x.size()

        # mesh grid
        xx = torch.arange(0, W, device=self.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=self.device).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        vgrid = grid + flo1
        vgrid2 = grid + flo2

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        vgrid = F.grid_sample(vgrid2, vgrid, align_corners=True, mode="bilinear")

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = F.grid_sample(
            x, vgrid, align_corners=True, mode="bicubic", padding_mode="reflection"
        )

        return output

    def compute_doublewarp_loss(self, input, target, flo1, flo2):
        # Warp input on target
        warped = self.doublewarp(target, flo1, flo2)
        border = 6
        input = input[..., border:-border]
        warped = warped[..., border:-border]
        return self.criterion(input, warped)

    def forward(self, input, target, flow):
        # Warp input on target
        warped = self.warp(target, flow)
        border = 5
        input = input[..., border:-border, border:-border]
        warped = warped[..., border:-border, border:-border]
        return self.criterion(input, warped)
