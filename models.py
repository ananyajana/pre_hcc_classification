#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:04:36 2020

@author: aj611
"""

import torch
from torch import nn
from torchvision import models as torch_models
import torch.nn.functional as F
from collections import OrderedDict

class sa2D(nn.Module):
    def __init__(self, inChans, n_labels=3):

        super(sa2D, self).__init__()
        #self.avg_pool = nn.AvgPool3d((8, 8, 4), stride=(8, 8, 4)) #GAP layer
        # alternatively, we could also put nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc1 = nn.Linear(512, 256)
        self.fc1 = nn.Linear(inChans, 256)
        self.fc2 = nn.Linear(256, n_labels)
        # although not mentioned in SAR paper we need the sigmoid to squeeze the result in the range(0, 1)
        self.sigmoid = nn.Sigmoid()

    #def forward(self, x=None):
    def forward(self, x):
        #print('input size: ', x.size())
        #out = self.avg_pool(x)
        #print('after avg_pool size: ', out.size())
        #out = torch.flatten(out, 1)
        #print('after flatten size: ', out.size())
        #out = self.fc1(out)
        out = self.fc1(x)
        #print('after fc1 size: ', out.size())
        out = self.fc2(out)
        #print('after fc2 size: ', out.size())
        out = self.sigmoid(out)
        #print('after sigmoid size: ', out.size())
        return out

class Classifier(nn.Module):
    def __init__(self, num_features=10, num_classes=3):
        super(Classifier, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.num_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.num_classes)

    def forward(self, x, feat=False):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.mean(dim=0, keepdim=True)
        x = self.fc3(x)

        return x

class BaselineNet7(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(BaselineNet7, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.resnet = ResNet_extractor2(layers, train_res4, 2, False)
        self.classifier = Classifier(dim_dict[layers], num_classes)
        self.scale_classifier = sa2D(dim_dict[layers], 3)

    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.resnet.load_state_dict(new_chkpt)

    def forward(self, x):
        x = self.resnet(x)
        out = self.classifier(x)
        scales = self.scale_classifier(x)
        return out, scales

class BaselineNet2(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(BaselineNet2, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.resnet = ResNet_extractor2(layers, train_res4, 2, False)
        self.classifier = Classifier(dim_dict[layers], num_classes)

    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.resnet.load_state_dict(new_chkpt)

    def forward(self, x):
        x = self.resnet(x)
        out = self.classifier(x)
        return out


class ResNet_extractor2(nn.Module):
    def __init__(self, layers=18, train_res4=True, num_classes=2, pre_train=False):
        super().__init__()
        self.num_classes = num_classes
        self.pre_train = pre_train
        
        if layers == 18:
            print('pretrained')
            self.resnet = torch_models.resnet18(pretrained=True)
            #self.resnet = torch_models.resnet18(pretrained=False)
        else:
            raise(ValueError('Layers must be 18, 34, 50 or 101.'))
        for param in self.resnet.parameters():
            #param.requires_grad = False
            param.requires_grad = True

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        self.fc = nn.Linear(512, self.num_classes)
    
    def get_resnet(self):
        return self.resnet

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        #print('x size before avg_pool: ', x.size())
        x = self.resnet.avgpool(x)
        #print('x size: ', x.size())

        x = torch.flatten(x, 1)
        if self.pre_train is False:
            #print('x size: ', x.size())
            return x
        else:
            return self.fc(x)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        self.cnn1 = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(1, 4, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(4),
                
                nn.ReflectionPad2d(1),
                nn.Conv2d(4, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),
                
                nn.ReflectionPad2d(1),
                nn.Conv2d(8, 8, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(8),
        )
        
        self.fc1 = nn.Sequential(
                nn.Linear(8*100*100, 500),
                nn.ReLU(inplace=True),
                
                nn.Linear(500, 500),
                nn.ReLU(inplace=True),
                
                nn.Linear(500, 5))
    
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

class BaselineNet(nn.Module):
    def __init__(self, in_features=1, num_classes=3, pre_train=False):
        super(BaselineNet, self).__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        #self.num_features = 128
        self.num_features = 256
        #self.pre_train_num_classes = 4
        self.pre_train_num_classes = 2
        self.pre_train = pre_train
        self.fea_extractor = FeaExtractorContextRestore2(self.in_features, self.num_features, self.pre_train)
        self.classifier = Classifier(self.num_features, self.num_classes)

        '''
        for param in self.fea_extractor.parameters():
            param.requires_grad = False

        for param in self.fea_extractor.conv2.parameters():
            param.requires_grad = True
        for param in self.fea_extractor.conv3.parameters():
            param.requires_grad = True
        for param in self.fea_extractor.conv4.parameters():
            param.requires_grad = True
        for param in self.fea_extractor.conv5.parameters():
            param.requires_grad = True
        for param in self.fea_extractor.conv10.parameters():
            param.requires_grad = True
        for param in self.fea_extractor.conv11.parameters():
            param.requires_grad = True
        '''

    '''
    def set_fea_extractor(self, chkpt_path):
        chkpt = torch.load(chkpt_path)
        self.fea_extractor.load_state_dict(chkpt)
    '''
    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.fea_extractor.load_state_dict(new_chkpt)

    def forward(self, x):
        x = self.fea_extractor(x)
        out = self.classifier(x)

        return out


def weight_init(net):
    for name, m in net.named_children():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

def weight_init3(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()    

def weight_init2(net):
    for m in net.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

# feature extractor class
# this class can also be trained for self-supervision


class FeaExtractorContextRestore2(nn.Module):
    def __init__(self, in_features=1, num_features=10, pre_train=False):
        super(FeaExtractorContextRestore2, self).__init__()

        self.in_features = in_features
        self.num_features = num_features
        self.pre_train = pre_train
        
        self.conv1 = nn.Conv2d(self.in_features, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        #self.conv3 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        #self.conv6 = nn.Conv2d(32, 32, 3, 1, 1)
        #self.conv7 = nn.Conv2d(32, 64, 3, 1, 1)
        #self.conv8 = nn.Conv2d(64, 64, 3, 1, 1)
        #self.conv9 = nn.Conv2d(64, 64, 3, 1, 1)
        #self.conv10 = nn.Conv2d(64, self.num_features, 3, 1, 1)
        self.conv10 = nn.Conv2d(64, self.num_features, 3, 1, 1)
        self.conv11 = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
        #self.conv12 = nn.Conv2d(self.num_features, self.num_features, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.up1 = nn.Conv2d(self.num_features, 64, 3, 1, 1)
        self.up2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.up3 = nn.Conv2d(32, 16, 3, 1, 1)
        self.up4 = nn.Conv2d(16, self.in_features, 3, 1, 1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
    def forward(self, x):        
        x = self.pool(self.conv2(self.conv1(x)))
        x = self.pool(self.conv5(self.conv4(x)))
        x = self.pool(self.conv11(self.conv10(x)))
        if self.pre_train is False:
            x = self.adaptive_avg_pool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            return self.up4(self.upsample(self.up3(self.upsample(self.up2(self.upsample(self.up1(x)))))))

