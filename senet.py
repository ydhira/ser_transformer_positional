import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np
from scipy.fftpack import dct, idct
from scipy import linalg as la
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
#import matplotlib.pyplot as plt 

# PyTorch implementation of Squeeze-Excitation Network
# source: https://github.com/moskomule/senet.pytorch
CUDA=True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn = nn.BatchNorm1d(128, affine=False)
        # self.fc1 = nn.utils.weight_norm(nn.Linear(128, class_n, bias=False), name='weight')

        ## residual block
        self.conv3_res1a = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn3_res1a = nn.BatchNorm2d(64)
        self.conv3_res1b = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn3_res1b = nn.BatchNorm2d(64)
        
    def forward(self, x):
        # print("1: ", x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        # print("2: ", x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        # print("3: ", x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # uncomment this line if you want to use deeper network
        x = x + self.bn3_res1b(self.conv3_res1b(F.relu(self.bn3_res1a(self.conv3_res1a(x)))))
        # print("4: ", x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print("5: ", x.shape)
        x = F.avg_pool2d(x, [x.size()[2], x.size()[3]], stride=1)
        # print("6: ", x.shape)
        x = x.view(-1, 128)
        # print("7: ", x.shape)
        x = self.bn(x)
        # print("8: ", x.shape)
        return x

class NetR(nn.Module):
    def __init__(self):
        super(NetR, self).__init__()
        self.conv1 = nn.ConvTranspose2d(16, 1, 2, 2)
        self.conv2 = nn.ConvTranspose2d(32, 16, 3, 2, 1)
        # self.conv3_1 = nn.ConvTranspose2d(32, 32, 1, 1)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.conv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn = nn.BatchNorm1d(128, affine=False)
        # self.fc1 = nn.utils.weight_norm(nn.Linear(128, class_n, bias=False), name='weight')

        ## residual block
        self.conv3_res1a = nn.ConvTranspose2d(32, 32, 3, 1, 1, bias=False)
        self.bn3_res1a = nn.BatchNorm2d(32)
        self.conv3_res1b = nn.ConvTranspose2d(32, 32, 3, 1, 1, bias=False)
        self.bn3_res1b = nn.BatchNorm2d(32)

        self.upsample = nn.Upsample(scale_factor=(5,50), mode='nearest')

        self.final_layer = nn.Conv2d(1, out_channels= 1,  kernel_size= 1, padding= 1)
        
    def forward(self, x):
        # print("1: ", x.shape)
        x = self.upsample(x)
        # print("1.1: ", x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        # print("2: ", x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        # print("3: ", x.shape)
        # x = F.relu(self.bn3(self.conv3_1(x)))
        # print("3_2: ", x.shape)
        x = x + self.bn3_res1b(self.conv3_res1b(F.relu(self.bn3_res1a(self.conv3_res1a(x)))))
        x = F.relu(self.bn2(self.conv2(x)))
        # uncomment this line if you want to use deeper network
        # print("4: ", x.shape)
        
        x = F.relu(self.bn1(self.conv1(x)))
        # print("5: ", x.shape)
        # x = F.avg_pool2d(x, [x.size()[2], x.size()[3]], stride=1)
        # 
        # x = x.view(-1, 128)
        # print("7: ", x.shape)
        # x = self.bn(x)
        # print("8: ", x.shape)
        x = self.final_layer(x)
        # print("6: ", x.shape)
        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Net()
        self.decoder = NetR()

    def forward(self, x):
        # print("Input: ", x.shape)
        emb = self.encoder(x)
        # print("Emb: ", emb.shape)
        emb = emb.unsqueeze(2).unsqueeze(2)
        out = self.decoder(emb)
        return out 

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # F_squeeze 
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * self.expansion, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ basic ResNet class: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py """
    def __init__(self, block, layers, num_classes, focal_loss=False):
        
        self.inplanes = 16
        self.focal_loss = focal_loss 

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.classifier = nn.Linear(128 * block.expansion, num_classes)
        # print("*****************", 128 * block.expansion)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        # print("1: ", x[0].shape)
        x = [torch.Tensor(f).cuda() for f in x]
        x = rnn.pad_sequence(x, batch_first=True).unsqueeze(1)
        # print("2: ", x.shape, x[0].shape)
        # b, t, f = x.shape[0], x.shape[1], x.shape[2]
        x = self.conv1(x)
        # print(x.size())
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size())

        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        # print(x.size())

        x = self.avgpool(x).view(x.size()[0], -1)
        # print("-l", x.shape) # feature_size is 256 
        # x_with_target = torch.cat((x, target), dim=1)
        out = self.classifier(x)
        # print(out.shape)

        if self.focal_loss: return out 
        else: return F.log_softmax(out, dim=-1)

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def se_resnet18(**kwargs):
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet34(**kwargs):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def se_resnet50(**kwargs):
    model = ResNet(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model

def se_resnet50attn(**kwargs):
    model = ResNet_with_attention(SEBottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def se_resnet101(**kwargs):
    model = ResNet(SEBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


