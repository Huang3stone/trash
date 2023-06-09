import torch
import torch.nn as nn
from antialias import Downsample as downsamp


class FusionLayer(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.fusion = ConvBlock(inchannel, inchannel, 1, 1, 0, bias=True)
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.max_pool(x).view(b, c)
        max = self.fc(max).view(b, c, 1, 1)
        fusion = self.fusion(avg + max)
        fusion = x * fusion.expand_as(x)
        fusion = fusion + x
        fusion = self.outlayer(fusion)
        return fusion


class NewEncoderBlock(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size, stride, padding):
        super(NewEncoderBlock, self).__init__()
        self.firstconv = ConvBlock(input_size=4, output_size=input_dim, kernel_size=3, stride=1, padding=1)
        self.prelu = nn.PReLU()
        codeim = out_dim // 2
        self.conv_Encoder = ConvBlock(input_dim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Offset = ConvBlock(codeim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Decoder = ConvBlock(codeim, out_dim, kernel_size, stride, padding, isuseBN=False)

    def forward(self, x):
        firstconv = self.prelu(self.firstconv(x))
        code = self.conv_Encoder(firstconv)
        offset = self.conv_Offset(code)
        code_add = code + offset
        out = self.conv_Decoder(code_add)
        return out


class ResidualDownSample(nn.Module):
    def __init__(self, in_channel, bias=False):
        super(ResidualDownSample, self).__init__()
        self.prelu = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=bias)
        self.downsamp = downsamp(channels=in_channel, filt_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channel, 2 * in_channel, 1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        out = self.prelu(self.conv1(x))
        out = self.downsamp(out)
        out = self.conv2(out)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channel, scale_factor=2, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = scale_factor
        self.residualdownsample = ResidualDownSample(in_channel)

    def forward(self, x):
        out = self.residualdownsample(x)
        return out


class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,
                                                    bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class UpSample(nn.Module):
    def __init__(self, in_channel, scale_factor=2, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
        self.residualupsample = ResidualUpSample(in_channel)

    def forward(self, x):
        out = self.residualupsample(x)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, out_dim, ):
        super(EncoderBlock, self).__init__()
        hidden = input_dim // 4  # 2021-3-30 8->4
        self.prelu = nn.PReLU()

        self.SGblock = nn.Sequential(
            ConvBlock(input_dim, input_dim, 3, 1, 1, isuseBN=False),
            nn.Conv2d(input_dim, hidden, 1, 1, 0),
            nn.Conv2d(hidden, out_dim, 1, 1, 0, ),
            ConvBlock(out_dim, out_dim, 3, 1, 1, isuseBN=False))

    def forward(self, x):
        out = self.SGblock(x)
        out = out + x
        return out


# class EnhancedNet(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(EnhancedNet, self).__init__()
#         # self.encoder = nn.Sequential(
#         #     ConvBlock(in_channels, in_channels*4, 3, 1, 1),
#         # )
#         self.conv1 = ConvBlock(in_channels, in_channels*4, 3, 1, 1)
#         self.downsample1 = ConvBlock(in_channels*4, in_channels*4, 3, 2, 1)
#         self.downsample2 = ConvBlock(in_channels*4, in_channels*4, 3, 2, 1)


#         self.upsample1 = nn.Sequential(
#                         nn.Upsample(scale_factor=2, mode='nearest'),
#                         DeconvBlock(in_channels*4, in_channels*4, 3, 1, 1),
#         )
#         self.upsample2 = nn.Sequential(
#                         nn.Upsample(scale_factor=2, mode='nearest'),
#                         DeconvBlock(in_channels*4, in_channels*4, 3, 1, 1),
#         )

#         self.deconv_0 =  DeconvBlock(in_channels*4, out_channels, 3, 1, 1)
#         self.deconv_1 =  DeconvBlock(in_channels*4, out_channels, 3, 1, 1)
#         self.deconv_2 =  DeconvBlock(in_channels*4, out_channels, 3, 1, 1)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.downsample1(x)
#         x_0 = self.downsample2(x)

#         x_1 = self.upsample1(x_0)
#         x_2 = self.upsample2(x_1)

#         x_out0 = self.deconv_0(x_0)
#         x_out1 = self.deconv_1(x_1)
#         x_out2 = self.deconv_2(x_2)

#         return [x_out2, x_out1, x_out0]


class EnhancedNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EnhancedNet, self).__init__()
        # self.encoder = nn.Sequential(
        #     ConvBlock(in_channels, in_channels*4, 3, 1, 1),
        # )
        self.conv1 = ConvBlock(in_channels, in_channels * 4, 3, 1, 1)
        self.downsample1 = ConvBlock(in_channels * 4, in_channels * 4, 3, 2, 1)
        self.downsample2 = ConvBlock(in_channels * 4, in_channels * 4, 3, 2, 1)

        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
        )
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
        )

        self.conv2 = nn.Sequential(
            ConvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
        )

        self.deconv_0 = nn.Sequential(
            DeconvBlock(in_channels * 8, in_channels * 4, 3, 1, 1),
            # SAMBlock(in_channels * 4),
            # DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
            # DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
            DeconvBlock(in_channels * 4, out_channels, 3, 1, 1),
        )
        self.deconv_1 = nn.Sequential(
            DeconvBlock(in_channels * 8, in_channels * 4, 3, 1, 1),
            # SAMBlock(in_channels * 4),
            # DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
            # DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
            DeconvBlock(in_channels * 4, out_channels, 3, 1, 1),
        )
        self.deconv_2 = nn.Sequential(
            DeconvBlock(in_channels * 4 + in_channels, in_channels * 4, 3, 1, 1),
            # SAMBlock(in_channels * 4),
            # DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
            # DeconvBlock(in_channels * 4, in_channels * 4, 3, 1, 1),
            DeconvBlock(in_channels * 4, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        ori = x
        x = self.conv1(x)
        x_down_1 = self.downsample1(x)
        x_down_2 = self.downsample2(x_down_1)

        x_up_0 = self.conv2(x_down_2)
        x_up_0 = x_down_2 + x_up_0

        x_up_1 = self.upsample1(x_up_0)
        x_up_2 = self.upsample2(x_up_1)

        x_out0 = self.deconv_0(torch.cat([x_down_2, x_up_0], dim=1))
        x_out1 = self.deconv_1(torch.cat([x_down_1, x_up_1], dim=1))
        x_out2 = self.deconv_2(torch.cat([ori, x_up_2], dim=1))

        return [x_out2, x_out1, x_out0]


class lowlightnet_ACM(nn.Module):
    def __init__(self, input_dim=3, dim=[16, 32, 64, 128]):
        super(lowlightnet_ACM, self).__init__()
        inNet_dim = input_dim + 1
        self.prelu = torch.nn.PReLU()
        self.enhanceNet = EnhancedNet(3, 3)
        self.out_conv3 = nn.Conv2d(dim[0], 4, 3, 1, 1)
        self.out_conv4 = nn.Conv2d(4, 3, 1, 1, 0)

        self.firstconv = nn.Sequential(ConvBlock(input_size=4, output_size=8, kernel_size=3, stride=1, padding=1),
                                       ConvBlock(input_size=8, output_size=dim[0], kernel_size=3, stride=1, padding=1),
                                       )

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                # DownSample(in_channel=dim[i]),
                ConvBlock(dim[i], dim[i] * 2, 3, 2, 1)
            )
            self.downsample_layers.append(stage)

        self.edecoder_level0 = ConvBlock(dim[0], dim[0], 3, 1, 1)

        self.endecoderList = nn.ModuleList()
        for i in range(1, 4):
            endecoder = nn.Sequential(
                ConvBlock(dim[i], dim[i], 1, 1, 0),
            )
            self.endecoderList.append(endecoder)

        self.attention = nn.Sequential(
            ConvBlock(dim[3], dim[3], 3, 1, 1),
        )

        self.upsample_layers = nn.ModuleList()
        for i in range(len(dim) - 1, 0, -1):
            stage = nn.Sequential(
                UpSample(in_channel=dim[i], scale_factor=2),
            )
            self.upsample_layers.append(stage)

        self.convnext_layers = nn.ModuleList()
        for i in range(len(dim) - 2, -1, -1):
            block = nn.Sequential(
                ConvBlock(input_size=dim[i], output_size=dim[i], kernel_size=1, stride=1, padding=0),
            )
            self.convnext_layers.append(block)

        self.up_conv = nn.Sequential(
            ConvBlock(dim[0], dim[0], 3, 1, 1),
        )

        self.up4 = nn.Sequential(
            # ParallelAttention(in_planes=dim[3]),
            ConvBlock(dim[3], dim[0], 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(dim[0], dim[0], 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(dim[0], dim[0], 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(dim[0], dim[0], 3, 1, 1),
            # nn.Upsample(scale_factor=8, mode='nearest'),
            # ConvBlock(dim[3], dim[0], 3, 1, 1),
        )
        self.up3 = nn.Sequential(
            # ParallelAttention(in_planes=dim[2]),
            ConvBlock(dim[2], dim[0], 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(dim[0], dim[0], 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(dim[0], dim[0], 3, 1, 1),
            # nn.Upsample(scale_factor=4, mode='nearest'),
            # ConvBlock(dim[2], dim[0], 3, 1, 1),
        )
        self.up2 = nn.Sequential(
            # ParallelAttention(in_planes=dim[1]),
            ConvBlock(dim[1], dim[0], 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(dim[0], dim[0], 3, 1, 1),
            # nn.Upsample(scale_factor=2, mode='nearest'),
            # ConvBlock(dim[1], dim[0], 3, 1, 1),
        )

        self.fusion_module = FusionLayer(dim[0] * 4, dim[0] * 4)
        self.out_conv = nn.Conv2d(dim[0] * 4, dim[0], 3, 1, 1)

    def forward(self, x_ori, tar=None):
        x = x_ori
        [x_bright_0, x_bright_1, x_bright_2] = self.enhanceNet(x_ori)
        x_bright, _ = torch.max(x_bright_0, dim=1, keepdim=True)

        x_in = torch.cat((x, x_bright), 1)

        f_endecoder = self.firstconv(x_in)

        # 第0层的同尺度输出 --->
        f_endecoder_level_out = f_endecoder
        before = f_endecoder_level_out
        for i in range(4):
            f_endecoder_level_out = self.edecoder_level0(f_endecoder_level_out) + before
            before = f_endecoder_level_out
        f_endecoder_level_out = f_endecoder_level_out + x_bright
        out_features = [f_endecoder_level_out]

        # 保留每层第一次下采样得到的feature map
        down_features = [f_endecoder]

        # 第n层的同尺度的输出
        x = f_endecoder
        for i in range(3):
            x = self.downsample_layers[i](x)
            down_features.append(x)
            if i == 0:
                x_bright_1, _ = torch.max(x_bright_1, dim=1, keepdim=True)
                x = x + x_bright_1
            elif i == 1:
                x_bright_2, _ = torch.max(x_bright_2, dim=1, keepdim=True)
                x = x + x_bright_2
            n = 3 - i
            before = x
            for j in range(n):
                res = self.endecoderList[i](x) + before
                before = res

            out_features.append(res)

        x_1 = self.attention(x)
        x = x_1 + x

        out_features_2 = [x]
        for i in range(3):
            x = self.upsample_layers[i](x)
            x = x + out_features[len(out_features) - 1 - i - 1] + down_features[len(out_features) - 1 - i - 1]
            x = self.convnext_layers[i](x)
            if i == 0:
                x_bright_2, _ = torch.max(x_bright_2, dim=1, keepdim=True)
                x = x + x_bright_2
            elif i == 1:
                x_bright_1, _ = torch.max(x_bright_1, dim=1, keepdim=True)
                x = x + x_bright_1
            out_features_2.append(x)
        fusion_out = torch.cat([self.up4(out_features_2[0]),
                                self.up3(out_features_2[1]),
                                self.up2(out_features_2[2]),
                                self.up_conv(out_features_2[3])], 1)

        fusion_out = self.fusion_module(fusion_out)
        fusion_out = self.out_conv(fusion_out)
        fusion_out = f_endecoder + fusion_out

        # real out
        out = self.prelu(fusion_out)
        out = self.out_conv3(out)
        out = out + x_bright
        out = self.out_conv4(out)
        return out


############################################################################################
# Base models
############################################################################################

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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)
        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_filter)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(num_filter)

        self.act1 = torch.nn.PReLU()
        self.act2 = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.act2(out)

        return out


if __name__ == '__main__':
    data = torch.randn(8, 3, 128, 128)
    model = lowlightnet_ACM()
    out = model(data)
    print(out)