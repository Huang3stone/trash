import torch
import torch.nn as nn
from antialias import Downsample as downsamp
from research import ACM, MetaNeXtBlock, GAM_Attention


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
        self.fusion   = ConvBlock(inchannel, inchannel, 1,1,0,bias=True)
        self.outlayer = ConvBlock(inchannel, outchannel, 1, 1, 0, bias=True)

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        avg = self.fc(avg).view(b, c, 1, 1)
        max = self.max_pool(x).view(b, c)
        max = self.fc(max).view(b, c, 1, 1)
        fusion = self.fusion(avg+max) 
        fusion = x * fusion.expand_as(x)
        fusion = fusion + x
        fusion = self.outlayer(fusion)
        return fusion


class NewEncoderBlock(nn.Module):  
    def __init__(self, input_dim, out_dim, kernel_size, stride, padding):
        super(NewEncoderBlock, self).__init__()
        self.firstconv = ConvBlock(input_size=4,output_size=input_dim,kernel_size=3,stride=1,padding=1)
        self.prelu = nn.PReLU()
        codeim = out_dim // 2
        self.conv_Encoder = ConvBlock(input_dim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Offset  = ConvBlock(codeim, codeim, kernel_size, stride, padding, isuseBN=False)
        self.conv_Decoder = ConvBlock(codeim, out_dim, kernel_size, stride, padding, isuseBN=False)

    def forward(self, x):
        firstconv = self.prelu(self.firstconv(x))
        code   = self.conv_Encoder(firstconv)
        offset = self.conv_Offset(code)
        code_add = code + offset
        out    = self.conv_Decoder(code_add)
        return out



class ResidualDownSample(nn.Module):
    def __init__(self,in_channel,bias=False):
        super(ResidualDownSample,self).__init__()
        self.prelu = nn.PReLU()
        
        self.conv1 = nn.Conv2d(in_channel,in_channel,3,1,1,bias=bias)
        self.downsamp = downsamp(channels=in_channel,filt_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channel,2*in_channel,1,stride=1,padding=0,bias=bias)
    def forward(self, x):
        out = self.prelu(self.conv1(x))
        out = self.downsamp(out)
        out = self.conv2(out)
        return out

class DownSample(nn.Module):
    def __init__(self, in_channel,scale_factor=2, stride=2,kernel_size=3):
        super(DownSample,self).__init__()
        self.scale_factor=scale_factor
        self.residualdownsample=ResidualDownSample(in_channel)

    def forward(self, x):
        out = self.residualdownsample(x)
        return out

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class UpSample(nn.Module):
    def __init__(self, in_channel, scale_factor=2,stride=2,kernel_size=3):
        super(UpSample,self).__init__()
        self.scale_factor=scale_factor
        self.residualupsample=ResidualUpSample(in_channel)

    def forward(self, x):
        out = self.residualupsample(x)
        return out

class EncoderBlock(nn.Module):  
    def __init__(self, input_dim, out_dim,):
        super(EncoderBlock, self).__init__()
        hidden = input_dim // 4  # 2021-3-30 8->4
        self.prelu = nn.PReLU()
        
        self.SGblock = nn.Sequential(
                        ConvBlock(input_dim,input_dim,3,1,1,isuseBN=False),
                        nn.Conv2d(input_dim,hidden,1,1,0),
                        nn.Conv2d(hidden,out_dim,1,1,0,),
                        ConvBlock(out_dim,out_dim,3,1,1,isuseBN=False))
    def forward(self, x):
        out = self.SGblock(x)
        out = out + x
        return out


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class lowlightnet_ACM(nn.Module):
    def __init__(self, input_dim=3, dim=[16, 32, 64, 128]):
        super(lowlightnet_ACM, self).__init__()
        inNet_dim = input_dim + 1
        self.prelu   = torch.nn.PReLU()

        self.out_conv3 = nn.Conv2d(dim[0], 4, 3, 1, 1)
        self.out_conv4 = nn.Conv2d(4,3,1,1,0)

        

        self.firstconv = nn.Sequential(ConvBlock(input_size=4,output_size=dim[0],kernel_size=3,stride=1,padding=1),
                                       EncoderBlock(input_dim=dim[0],out_dim=dim[0]))

        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            stage = nn.Sequential(
                        DownSample(in_channel=dim[i]),
                        # MetaNeXtBlock(dim[i+1]),
            )
            self.downsample_layers.append(stage)


        self.endecoderList = nn.ModuleList()
        for i in range(1,4):
            endecoder  = EncoderBlock(input_dim=dim[i],out_dim=dim[i])
            self.endecoderList.append(endecoder)


        self.attention = GAM_Attention(in_channels=dim[3])
        
        self.upsample_layers = nn.ModuleList()
        for i in range(len(dim) - 1, 0, -1):
            stage = nn.Sequential(
                        UpSample(in_channel=dim[i],scale_factor=2),
            )
            self.upsample_layers.append(stage)

        
        self.convnext_layers = nn.ModuleList()
        for i in range(len(dim) - 2, -1, -1):
            block = nn.Sequential(ConvBlock(input_size=dim[i]*2,output_size=dim[i],kernel_size=1,stride=1,padding=0),
                                  EncoderBlock(input_dim=dim[i], out_dim=dim[i]),
                                  )
            self.convnext_layers.append(block)
        
        self.upsample_dim4 = UpSample(in_channel=dim[3],scale_factor=2)
        self.upsample_dim3 = UpSample(in_channel=dim[2],scale_factor=2)
        self.upsample_dim2 = UpSample(in_channel=dim[1],scale_factor=2)
        self.upsample_dim1 = UpSample(in_channel=dim[0],scale_factor=2)

        self.out_fushion = FusionLayer(dim[0] * 3, dim[0] * 3)  
        
        self.out_conv = nn.Conv2d(dim[0] * 3, dim[0], 3, 1, 1)

    def forward(self, x_ori, tar=None):
        x = x_ori
        x_bright, _ = torch.max(x_ori, dim=1, keepdim=True)

        x_in = torch.cat((x, x_bright), 1)

        f_endecoder = self.firstconv(x_in)
        out_features = [f_endecoder]
        x = f_endecoder
        for i in range(3):
            x = self.downsample_layers[i](x)
            out_features.append(self.endecoderList[i](x))
        
        x = self.attention(x)

        out_features_2 = []
        for i in range(3):
            x = self.upsample_layers[i](x)
            x = torch.cat([x, out_features[len(out_features)-1-i-1]], 1)
            x = self.convnext_layers[i](x)
            out_features_2.append(x)
        
        fusion_out = torch.cat([self.upsample_dim2(self.upsample_dim3(out_features_2[0])), self.upsample_dim2(out_features_2[1]), out_features_2[2]],1)
        fusion_out = self.out_fushion(fusion_out)
        fusion_out = self.out_conv(fusion_out)
        fusion_out = f_endecoder + fusion_out

        # real out
        out = self.prelu(fusion_out) 
        out = self.out_conv3(out)  
        out = out  +  x_bright
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
if __name__=='__main__':
    data=torch.randn(8,3,128,128).cuda()
    model = lowlightnet_ACM().cuda()
    out = model(data)
    print(out)