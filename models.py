import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLeaky(nn.Module):
    def __init__(self, in_dim, out_dim, slope=0.2):
        super(ConvLeaky, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.conv2 = nn.Conv2d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.slope = slope

    def forward(self, input):
        out = self.conv1(input)
        out = F.leaky_relu(out, self.slope)
        out = self.conv2(out)
        out = F.leaky_relu(out, self.slope)
        return out


class CSRBlock(nn.Module):
    def __init__(self, in_dim, out_dim, mode):
        super(CSRBlock, self).__init__()
        self.convleaky = ConvLeaky(in_dim, out_dim)
        if mode == "maxpool":
            self.final = lambda x: F.max_pool2d(x, kernel_size=2)
        elif mode == "bilinear":
            self.final = lambda x: F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
        else:
            raise Exception("mode must be maxpool or bilinear")

    def forward(self, input):
        out = self.convleaky(input)
        out = self.final(out)
        return out


class CSR_Net(nn.Module):
    def __init__(self, in_dim=2, range_=10.0, out_channels=2):
        super(CSR_Net, self).__init__()
        self.convPool1 = CSRBlock(in_dim, 32, mode="maxpool")
        self.convPool2 = CSRBlock(32, 64, mode="maxpool")
        self.convPool3 = CSRBlock(64, 128, mode="maxpool")
        self.convPool4 = CSRBlock(128, 256, mode="maxpool")

        self.convBinl1 = CSRBlock(256, 256, mode="bilinear")
        self.convBinl2 = CSRBlock(256, 256, mode="bilinear")
        self.convBinl3 = CSRBlock(256, 128, mode="bilinear")
        self.convBinl4 = CSRBlock(128, 64, mode="bilinear")

        self.conv1 = nn.Conv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )
        self.range = range_

    def forward(self, input):
        out = self.convPool1(input)
        out = self.convPool2(out)
        out = self.convPool3(out)
        out = self.convPool4(out)

        out = self.convBinl1(out)
        out = self.convBinl2(out)
        out = self.convBinl3(out)
        out = self.convBinl4(out)

        out = self.conv1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        out = torch.tanh(out) * self.range
        return out


# -------------------------------------------------------------------------------------------------
# Code from RCAN github repo: https://github.com/yulunzhang/RCAN/


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == "relu":
                    m.append(nn.ReLU(True))
                elif act == "prelu":
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == "relu":
                m.append(nn.ReLU(True))
            elif act == "prelu":
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x) + x


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.ReLU(True),
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x) + x


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, n_colors, conv=default_conv):
        super(RCAN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 2

        # define head module
        modules_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, n_resblocks=n_resblocks
            )
            for _ in range(n_resgroups)
        ]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size),
        ]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find("tail") >= 0:
                        print("Replace pre-trained upsampler to new one...")
                    else:
                        raise RuntimeError(
                            "While copying the parameter named {}, "
                            "whose dimensions in the model are {} and "
                            "whose dimensions in the checkpoint are {}.".format(
                                name, own_state[name].size(), param.size()
                            )
                        )
            elif strict:
                if name.find("tail") == -1:
                    raise KeyError('unexpected key "{}" in state_dict'.format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
