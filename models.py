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
        self.fc1 = nn.Linear(inChans, 256)
        self.fc2 = nn.Linear(256, n_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.sigmoid(out)
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


