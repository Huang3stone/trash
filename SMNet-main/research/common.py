import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()
        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out


class spatial_illumination_attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        e = self.conv(x)
        f = self.conv(x)
        g = self.conv(x)
        hw = w * h
        e = e.view(b, c, hw)
        f = f.view(b, c, hw)
        # (b, h * w, h * w)
        fea_ef = torch.matmul(e.transpose(1, 2), f)
        # (b, h * w, h * w)
        fea_ef = torch.softmax(fea_ef, dim=-1)

        # （b, c, h*w）
        g = g.view(b, c, h * w)

        # (b, c, h*w)
        final = torch.matmul(g, fea_ef)
        final = final.view(b, c, h, w)

        return final + x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class global_illumination_Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(global_illumination_Attention, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.compress = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.reconstruct = ConvBlock(out_channels, out_channels, 1, 1, 0)
        # self.relu = nn.PReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        # Global Average Pooling
        y = self.avgpool(x)

        # Compression
        press_y = self.compress(y)
        # Reconstruction
        rec_y = self.reconstruct(press_y)
        # Reweighting
        wei_y = torch.sigmoid(rec_y)
        # (b, c, 1, 1)
        wei_y = wei_y.view(b, c, -1, 1)
        # (b, c, h*c, 1)
        x_scale = x.view(b, c, -1, 1)
        # (b, c, h*c , 1)
        out = torch.matmul(x_scale, wei_y)
        # (b, c, h, c)
        out = out.view(b, c, h, w)

        return out


class illumination_module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(illumination_module, self).__init__()
        self.spatial_illumination_attention = spatial_illumination_attention(in_channels)
        # self.conv = ConvBlock(in_channels, in_channels, 1, 1, 0)
        self.global_illumination_Attention = global_illumination_Attention(in_channels, out_channels)
        self.out_conv = ConvBlock(2 * out_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.spatial_illumination_attention(x)
        x2 = self.global_illumination_Attention(x)
        fusion = torch.cat([x1, x2], 1)
        out = self.out_conv(fusion)
        return out + x


class illumination_module2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(illumination_module2, self).__init__()
        self.conv = ConvBlock(in_channels, in_channels, 1, 1, 0)
        self.global_illumination_Attention = global_illumination_Attention(in_channels, out_channels)

    def forward(self, x):
        x = self.conv(x)
        out = self.global_illumination_Attention(x)
        return out + x


class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, in_channel, dim=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(in_channel * 4, in_channel * 2, 1, 1, 0)

    def forward(self, x):
        out = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        out = self.conv(out)
        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1, 0)
        self.cv2 = ConvBlock(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1, 0)
        self.cv2 = ConvBlock(c1, c_, 1, 1, 0)
        self.cv3 = ConvBlock(2 * c_, c2, 1, 1, 0)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = ConvBlock(c1, c_, 1, 1, 0)
        self.cv2 = ConvBlock(c_ * 4, c2, 1, 1, 0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Bottle2neck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, shortcut=True, baseWidth=26, scale=4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            baseWidth: basic width of conv3x3
            scale: number of scale.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = ConvBlock(inplanes, width * scale, 1, 1, 0)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        convs = []
        for i in range(self.nums):
            convs.append(ConvBlock(width, width, 3, 1, 1))
        self.convs = nn.ModuleList(convs)

        self.conv3 = ConvBlock(width * scale, planes * self.expansion, 1, 1, 0)

        self.silu = nn.SiLU(inplace=True)
        self.scale = scale
        self.width = width
        self.shortcut = shortcut

    def forward(self, x):
        if self.shortcut:
            residual = x
        out = self.conv1(x)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1:
            out = torch.cat((out, spx[self.nums]), 1)

        out = self.conv3(out)
        if self.shortcut:
            out += residual
        out = self.silu(out)
        return out


class C3_Res2Block(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottle2neck(c_, c_, shortcut) for _ in range(n)))


class CAM(nn.Module):
    def __init__(self, inc, fusion='weight'):
        super().__init__()

        assert fusion in ['weight', 'adaptive', 'concat']
        self.fusion = fusion

        self.conv1 = ConvBlock(inc, inc, 3, 1, 1)
        self.conv2 = ConvBlock(inc, inc, 3, 1, 1)
        self.conv3 = ConvBlock(inc, inc, 3, 1, 1)

        self.fusion_1 = ConvBlock(inc, inc, 1, 1, 0)
        self.fusion_2 = ConvBlock(inc, inc, 1, 1, 0)
        self.fusion_3 = ConvBlock(inc, inc, 1, 1, 0)

        if self.fusion == 'adaptive':
            self.fusion_4 = ConvBlock(inc * 3, 3, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        if self.fusion == 'weight':
            return self.fusion_1(x1) + self.fusion_2(x2) + self.fusion_3(x3)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(
                self.fusion_4(torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)), dim=1)
            x1_weight, x2_weight, x3_weight = torch.split(fusion, [1, 1, 1], dim=1)
            return x1 * x1_weight + x2 * x2_weight + x3 * x3_weight
        else:
            return torch.cat([self.fusion_1(x1), self.fusion_2(x2), self.fusion_3(x3)], dim=1)


class SAMBlock(nn.Module):
    def __init__(self, c1):
        super(SAMBlock, self).__init__()
        c2 = c1
        self.conv_f = ConvBlock(c1, c2, 1, 1, 0)
        self.conv_g = ConvBlock(c1, c2, 1, 1, 0)
        self.conv_h = ConvBlock(c1, c2, 1, 1, 0)

    def forward(self, x):
        # (b,c,h,w)
        map_f = self.conv_f(x)
        # (b,c,w,h)
        map_f = map_f.transpose(2, 3)
        map_g = self.conv_g(x)
        map_h = self.conv_h(x)
        # (b, c, w, w)
        atten_map = torch.matmul(map_f, map_g)
        # atten_map = map_f * map_g
        atten_map = torch.softmax(atten_map, dim=-1)
        # (b,c,h,w) * (b, c, w, w)
        out = torch.matmul(map_h, atten_map)
        # out = map_h * atten_map
        # (b, c, h, w)
        return out + x


from torchstat import stat

if __name__ == '__main__':
    up = nn.Upsample(scale_factor=8, mode='nearest')
    input = torch.randn(50, 64, 120, 250)
    att = SAMBlock(64, 64)
    # stat(att, (64, 320, 320))
    output = up(input)
    print(output.shape)