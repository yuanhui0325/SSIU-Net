from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from math import sqrt
from senet.module import SELayer, ECALayer


def snake2rectangular(datas):  # data size: [B,3,32,32]
    batchSize = datas.shape[0]
    output = torch.zeros(datas.shape, dtype=torch.float32, device='cuda')
    for bs in range(batchSize):
        data = datas[bs, :, :, :]       # [3,32,32]
        temp = torch.zeros(data.shape, dtype=torch.float32, device='cuda')
        n = data.shape[1]  # 32
        pointNum = n * n
        data0 = data.permute(0, 2, 1)
        data1 = data0.reshape(3, n * n)      # [3,1024]
        input = data1.permute(1, 0)        # [1024,3]
        upb = 0  # upper bound
        lob = n - 1  # lower bound
        lfb = 0  # left  bound
        rtb = n - 1  # right bound
        j = 0
        i = -1
        r = torch.zeros((n, n), dtype=torch.float32)
        g = torch.zeros((n, n), dtype=torch.float32)
        b = torch.zeros((n, n), dtype=torch.float32)
        while pointNum > 0:
            while i < lob:
                i = i + 1
                r[i, j] = input[pointNum - 1, 0]
                g[i, j] = input[pointNum - 1, 1]
                b[i, j] = input[pointNum - 1, 2]
                pointNum = pointNum - 1
            lob = lob - 1
            while j < rtb:
                j = j + 1
                r[i, j] = input[pointNum - 1, 0]
                g[i, j] = input[pointNum - 1, 1]
                b[i, j] = input[pointNum - 1, 2]
                pointNum = pointNum - 1
            rtb = rtb - 1
            while i > upb:
                i = i - 1
                r[i, j] = input[pointNum - 1, 0]
                g[i, j] = input[pointNum - 1, 1]
                b[i, j] = input[pointNum - 1, 2]
                pointNum = pointNum - 1
            upb = upb + 1
            while j > lfb + 1:
                j = j - 1
                r[i, j] = input[pointNum - 1, 0]
                g[i, j] = input[pointNum - 1, 1]
                b[i, j] = input[pointNum - 1, 2]
                pointNum = pointNum - 1
            lfb = lfb + 1
        temp[0, :, :] = r
        temp[1, :, :] = g
        temp[2, :, :] = b
        output[bs, :, :, :] = temp
    return output

# class projection(nn.Module):
#     def __init__(self):
#         super(projection, self).__init__()
#         self.conv1=nn.Conv1d(3,32,1)
#         self.conv2=nn.Conv1d(32,64,1)
#         self.conv3=nn.Conv1d(64,128,1)
#
#         self.conv4=nn.Conv2d(128,64,1)
#         self.conv5=nn.Conv2d(64,32,1)
#         self.conv6=nn.Conv2d(32,3,1)
#
#
#     def forward(self,x):  #  x:[B,3,N]
#         B=x.size(0)            # batch_size
#         x=self.conv1(x)
#         x=self.conv2(x)
#         x=self.conv3(x)     # [B,128,N]
#         x1=x.reshape(B,-1,32,32)     #[B,128,32,32]
#         x1=self.conv4(x1)
#         x1=self.conv5(x1)
#         output=self.conv6(x1)    # [B,3,32,32]
#         return output
#
# class inv_projection(nn.Module):
#     def __init__(self):
#         super(inv_projection, self).__init__()
#         self.conv1=nn.Conv2d(3,3,1)
#
#         self.conv2=nn.Conv1d(3,3,1)
#
#
#     def forward(self, x):    # x:[B,3,32,32]
#         B=x.size(0)
#         x=self.conv1(x)
#         x=self.conv1(x)           # x:[B,3,32,32]
#         x1=x.reshape(B,3,-1)         # x1:[B,3,1024]
#         x1=self.conv2(x1)
#         output=self.conv2(x1)


class PtRestoration(nn.Module):
    def __init__(self):
        super(PtRestoration, self).__init__()
        self.preProcessing = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=32, stride=32),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(1, 1, kernel_size=1, stride=1)
            # nn.Conv2d(1, 1, kernel_size=9, stride=6),
            # nn.Conv2d(1, 1, kernel_size=9, stride=1),
            # nn.Conv2d(1, 1, kernel_size=7, stride=5)
        )
        self.base = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(16, 1, kernel_size=5, padding=2)
        )
        self.postProcessing = nn.Sequential(
            nn.Upsample(scale_factor=32, mode='nearest'),
            nn.Conv2d(1, 1, kernel_size=1)
        )
        self.layer1 = nn.AvgPool2d(kernel_size=(1024, 1), stride=1)
        self.layer2 = nn.Conv1d(1, 1, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        B = x.size(0)
        n = x.size(2)
        residual = x
        x = torch.unsqueeze(x, 1)
        x = x.expand(B, 1, n, n)                 # [B,1,1024,1024]
        x1 = self.preProcessing(x)               # [B,1,32,32]
        x2 = self.base(x1)                       # ARCNN-SKIP
        x3 = self.postProcessing(x2)              # [B,1,1024,1024]
        x4 = x + x3
        x5 = self.layer1(x4)                       # [B,1,1,1024]
        x5 = x5.squeeze(dim=1)
        # x5 = x5.unsqueeze(1)                # [B, 1, 1024]
        x6 = self.layer2(x5)
        output = residual + x6

        return output


class ARCNN_Auto(nn.Module):
    def __init__(self):
        super(ARCNN_Auto, self).__init__()
        self.upFeature=nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.PReLU(),
            nn.Conv1d(16, 32, 1),
            nn.PReLU()
            # nn.Conv1d(64, 128, 1),
        )
        self.layer21 = nn.Conv2d(32, 16, 1)
        self.layer22 = nn.Conv2d(32, 32, 1)

        self.reduceFeature=nn.Sequential(
            # nn.Conv2d(128, 64, 1),
            nn.Conv2d(48, 16, 1),
            nn.PReLU(),
            nn.Conv2d(16, 1, 1),
            nn.PReLU()
        )
        self.base = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            torch.nn.Dropout(0.5),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 1, kernel_size=5, padding=2)
        self.post_processing=nn.Sequential(
            nn.Conv1d(1, 1, 1),
            # nn.Conv1d(1,1,1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        B = x.size(0)
        x1 = self.upFeature(x)
        x1_2D = x1.reshape(B, -1, 32, 32)
        x1_2D_1 = F.relu(self.layer21(x1_2D))
        x1_2D_2 = F.relu(self.layer22(x1_2D))
        x1_2D = torch.cat((x1_2D_1, x1_2D_2), 1)
        x1_2D = self.reduceFeature(x1_2D)
        x2 = self.base(x1_2D)
        x3 = self.last(x2)
        x4_enhanced = x1_2D + x3    # 质量增强后的"二维图像“,----skip_connection
        x4_3D = x4_enhanced.reshape(B, 1, -1)      # [B,3,1024]
        x5 = self.post_processing(x4_3D)
        output = x+x5
        return output


class ARCNN(nn.Module):
    def __init__(self):
        super(ARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64, affine=True),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=7, padding=3),
            torch.nn.Dropout(0.5),
            nn.PReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            torch.nn.Dropout(0.5),
            nn.PReLU()
        )
        self.last = nn.Conv2d(16, 1, kernel_size=5, padding=2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x1 = self.base(x)
        x2 = self.last(x1)
        output = x+x2
        return output


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(1,0.2)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU(1,0.2)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out1 = self.relu(self.input(x))
        out2 = self.residual_layer(out1)
        out = self.output(out2)
        out = torch.add(out,residual)
        return out1, out2, out

class FastARCNN(nn.Module):
    def __init__(self):
        super(FastARCNN, self).__init__()
        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=2, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=7, padding=3),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=1),
            nn.PReLU()
        )
        self.last = nn.ConvTranspose2d(64, 3, kernel_size=9, stride=2, padding=4, output_padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, x):
        x = self.base(x)
        x = self.last(x)
        return x


class VRCNN(nn.Module):
    def __init__(self):
        super(VRCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, 5, 1, 2)   # input 1 channel only for y color component
        self.bn1 = nn.BatchNorm2d(64)
        self.layer21 = nn.Conv2d(64, 16, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.layer22 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.layer31 = nn.Conv2d(48, 16, 3, 1, 1)
        self.layer32 = nn.Conv2d(48, 32, 1, 1)
        self.layer4 = nn.Conv2d(48, 1, 3, 1, 1)       # output is 1 channel
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, input):
        residual = input
        x = F.relu(self.bn1(self.layer1(input)))    # [16,64,28,28]
        x21 = F.relu(self.bn2(self.layer21(x)))     # [16,16,28,28]
        x22 = F.relu(self.bn3(self.layer22(x)))      # [16,32,26,26]
        # 连接两个特征图
        x = torch.cat((x21, x22), 1)
        x31 = F.relu(self.layer31(x))
        x32 = F.relu(self.layer32(x))
        # 连接两个特征图
        x = torch.cat((x31, x32), 1)
        x = self.layer4(x)
        x = residual + x
        return x


class SE_VRCNN(nn.Module):
    def __init__(self):
        super(SE_VRCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, 5, 1, 2)   # input 1 channel only for y color component
        self.bn1 = nn.BatchNorm2d(64)
        self.layer21 = nn.Conv2d(64, 16, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.layer22 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.layer31 = nn.Conv2d(48, 16, 3, 1, 1)
        self.layer32 = nn.Conv2d(48, 32, 1, 1)
        self.se = SELayer(48, 3)
        self.layer4 = nn.Conv2d(48, 1, 3, 1, 1)       # output is 1 channel
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.00)

    def forward(self, input):
        residual = input
        x = self.layer1(input)   # [16,64,28,28]  torch.nn.Dropout(0.5),
        x = F.relu(self.bn1(x))
        x = nn.Dropout(0.5)(x)
        x21 = self.bn2(self.layer21(x))     # [16,16,28,28]
        x21 = F.relu(x21)
        # nn.Dropout(0.5)(x21)
        x22 = self.bn3(self.layer22(x))      # [16,32,26,26]
        x22 = F.relu(x22)       # 连接两个特征图
        # x22 = nn.Dropout(0.5)(x22)
        x = torch.cat((x21, x22), 1)
        x = nn.Dropout(0.5)(x)
        # x = self.bn2(x)
        # x = self.se(x)
        x31 = self.bn2(self.layer31(x))
        x31 = F.relu(x31)
        # nn.Dropout(0.5)(x31)
        x32 = self.bn3(self.layer32(x))
        x32 = F.relu(x32)
        # nn.Dropout(0.5)(x32)
        # 连接两个特征图
        x = torch.cat((x31, x32), 1)
        x = nn.Dropout(0.5)(x)
        x = self.se(x)
        x = self.layer4(x)
        x = residual + x
        x = F.relu(x)
        return x


class ECA_VRCNN(nn.Module):
    def __init__(self):
        super(ECA_VRCNN, self).__init__()
        self.layer1 = nn.Conv2d(1, 64, 5, 1, 2)   # input 1 channel only for y color component
        self.bn1 = nn.BatchNorm2d(64)
        self.layer21 = nn.Conv2d(64, 16, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.layer22 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.layer31 = nn.Conv2d(48, 16, 3, 1, 1)
        self.layer32 = nn.Conv2d(48, 32, 1, 1)
        self.eca = ECALayer(48, 3)
        self.layer4 = nn.Conv2d(48, 1, 3, 1, 1)       # output is 1 channel
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)

    def forward(self, input):
        residual = input
        x = self.layer1(input)   # [16,64,28,28]  torch.nn.Dropout(0.5),
        x = F.relu(self.bn1(x))
        x21 = self.bn2(self.layer21(x))     # [16,16,28,28]
        x21 = F.relu(x21)
        x22 = self.bn3(self.layer22(x))      # [16,32,26,26]
        x22 = F.relu(x22)       # 连接两个特征图
        x = torch.cat((x21, x22), 1)
        x31 = self.layer31(x)
        x31 = F.relu(x31)
        x32 = self.layer32(x)
        x32 = F.relu(x32)
        # 连接两个特征图
        x = torch.cat((x31, x32), 1)
        x = self.eca(x)
        x = self.layer4(x)
        x = residual + x
        x = F.relu(x)
        return x


"""
*******U-Net******
"""

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(1,0.2),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(1,0.2)
        )


    def forward(self,x):
        x = self.conv(x)
        return x




class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.PReLU(1,0.2)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class pixelshuffle(nn.Module):
    def __init__(self, ch_in):
        super(pixelshuffle, self).__init__()
        self.ps = nn.PixelShuffle(2)
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,2*ch_in,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(2*ch_in),
			nn.PReLU(1,0.2)
        )

    def forward(self,x):
        x = self.up(x)
        x = self.ps(x)
        return x


class convh1_1(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(convh1_1,self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.PReLU(1,0.2),
            nn.Conv2d(ch_out, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(1,0.2)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class Sepconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Sepconv, self).__init__()
        self.sep = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(1,0.2),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=2, dilation=2,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.PReLU(1,0.2)
        )

    def forward(self, x):
        x = self.sep(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, ch_in):
        super(Attention_block, self).__init__()
        self.Conv1 = conv_block(ch_in, ch_in)
        self.Conv2 = conv_block(ch_in, ch_in)
        self.Conv3 = nn.Sequential(
            nn.Conv2d(2*ch_in, 2*ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(2*ch_in),
            nn.PReLU(1, 0.2),
            nn.Conv2d(2*ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.Conv4 = conv_block(ch_in=ch_in, ch_out=ch_in)

    def forward(self, x, y):
        mixs = torch.cat([x, y], dim=1)
        mixs = self.Conv3(mixs)

        b = torch.sigmoid(mixs)
        x = torch.mul(x, b)
        x = self.Conv2(x)

        return [x, y]


class U_Net(nn.Module):
    def __init__(self,img_ch=4,output_ch=1):
        super(U_Net,self).__init__()
        self.rl = nn.ReLU(True)
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Avgpool = nn.AvgPool2d(2, 2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.cf0 = conv_block(ch_in=512, ch_out=512)
        # self.cf1 = conv_block(ch_in=512, ch_out=512)
        # self.cf2 = conv_block(ch_in=256, ch_out=256)
        # self.cf3 = conv_block(ch_in=256, ch_out=256)

        self.cf0 = conv_block(ch_in=256, ch_out=256)
        self.cf1 = Sepconv(ch_in=512, ch_out=512)
        self.cf2 = conv_block(ch_in=256, ch_out=256)
        self.cf3 = Sepconv(ch_in=512, ch_out=512)
        # self.cf4 = Sepconv(ch_in=512, ch_out=256)
        # self.cf5 = conv_block(ch_in=512, ch_out=512)

        self.atten1 = Attention_block(64)
        self.atten2 = Attention_block(128)
        self.atten3 = Attention_block(256)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.ps4 = pixelshuffle(ch_in=512)
        self.ps3 = pixelshuffle(ch_in=256)
        self.ps2 = pixelshuffle(ch_in=128)

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.001)

    def forward(self, x, types=1):
        # encoding path
        t1 = torch.flip(x, [3])
        t2 = torch.flip(x, [2])
        t3 = torch.flip(x, [2, 3])
        xx = torch.cat([x, t1, t2, t3], dim=1)

        x1 = self.Conv1(xx)
        x2 = self.Maxpool(x1)    # [batch, 64, 16, 16]
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)    # [batch, 128, 8, 8]
        x3 = self.Conv3(x3)

        if types > 0:
            x3 = self.cf0(x3)
            x3 = self.cf2(x3)
            d3 = self.ps3(x3)  # 256->128
        else:
            x4 = self.Maxpool(x3)
            x4 = self.Conv4(x4)
            x4 = self.cf1(x4)
            x4 = self.cf3(x4)

            d4 = self.ps4(x4)
            d4 = torch.cat(self.atten3(x3, d4), dim=1)
            d3 = self.Up_conv4(d4)
            d3 = self.ps3(d3)  # 256->128

        # x4 = self.Avgpool(x3)
        # x4 = self.Conv4(x4)



        # x5 = self.cf0(x4)
        # x51 = self.cf1(x5)
        # x52 = self.cf2(x5)
        # x5 = torch.cat((x51, x52), dim=1)
        
        # d4 = self.ps4(x4)
        # d4 = torch.cat(self.atten3(x3, d4), dim=1)
        # d4 = self.Up_conv4(d4)


        d3 = torch.cat(self.atten2(x2, d3),dim=1)
        d3 = self.Up_conv3(d3)   #256->128
        
        d2 = self.ps2(d3)     #128->64
        d2 = torch.cat(self.atten1(x1, d2),dim=1)
        d2 = self.Up_conv2(d2)   #128->64
        
        d1 = self.Conv_1x1(d2)

        d1 = d1+x


        return d1


if __name__ == '__main__':
    digital = torch.rand(16, 1, 32, 32)
    a = Attention_block(1)
    layer1 = nn.Conv2d(1, 1, kernel_size=32, stride=32)
    # layer1=nn.Conv2d(1, 1, kernel_size=9, stride=6)
    # layer2=nn.Conv2d(1, 1, kernel_size=9, stride=1)
    # layer3=nn.Conv2d(1, 1, kernel_size=7, stride=5)
    dig1 = layer1(digital)
    # dig2=layer2(dig1)
    # dig3=layer3(dig2)
    print(dig1.size())
