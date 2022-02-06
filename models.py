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


########################## AttnMIL ######################
class BaselineNet3(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(BaselineNet3, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.L = 512
        self.D = 128
        self.K = 1

        self.resnet = ResNet_extractor2(layers, train_res4, 2, False)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        ) 
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, num_classes),
            nn.Sigmoid()
        )

    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.resnet.load_state_dict(new_chkpt)

    def forward(self, x):
        H = self.resnet(x) # NxL
        
        #print('H after feature extraction: ', H.size())
        A = self.attention(H) # NxK
        #print('A after attention tion: ', A.size())
        A = torch.transpose(A, 1, 0) # KxN
        #print('A after transpose: ', A.size())
        A = F.softmax(A, dim=1) # softmax over N
        #print('A after softmax : ', A.size())
        M = torch.mm(A, H)  #KxL
        #print('M after torch.mm : ', A.size())
        Y_prob = self.classifier(M)
        #print('Y_prob size: ', Y_prob.size())
        Y_hat = torch.ge(Y_prob, 0.5).float()
        #return Y_prob, Y_hat, A
        return Y_prob

################################ Gated AttnMIL ############################
class BaselineNet4(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(BaselineNet4, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.L = 512
        self.D = 128
        self.K = 1

        self.resnet = ResNet_extractor2(layers, train_res4, 2, False)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        ) 
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, num_classes),
            nn.Sigmoid()
        )

    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.resnet.load_state_dict(new_chkpt)

    def forward(self, x):
        H = self.resnet(x) # NxL
        
        #print('H after feature extraction: ', H.size())
        A_V = self.attention_V(H) # NxD
        A_U = self.attention_U(H) # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        #print('A after attention tion: ', A.size())
        A = torch.transpose(A, 1, 0) # KxN
        #print('A after transpose: ', A.size())
        A = F.softmax(A, dim=1) # softmax over N
        #print('A after softmax : ', A.size())
        M = torch.mm(A, H)  #KxL
        #print('M after torch.mm : ', A.size())
        Y_prob = self.classifier(M)
        #print('Y_prob size: ', Y_prob.size())
        Y_hat = torch.ge(Y_prob, 0.5).float()
        #return Y_prob, Y_hat, A
        return Y_prob, Y_hat, A

     # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data[0]

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


##################33 Loss AttnMIL #######################
class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1, flag):
        if flag==1:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            

            alpha01 = features.size(0)*alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else:
            context = features
            alpha = torch.zeros(features.size(0),1)
                
        return context, out_c, torch.squeeze(alpha)
# Loss attention method
class BaselineNet5(nn.Module):
    def __init__(self, num_classes, layers=18, train_res4=True):
        super(BaselineNet5, self).__init__()
        dim_dict = {18: 512, 34: 512, 50: 2048, 101: 2048}

        self.L = 512
        self.D = 128
        self.K = 1

        self.resnet = ResNet_extractor2(layers, train_res4, 2, False)

        self.att_layer = AttentionLayer(self.L)
        #self.linear = nn.Linear(self.L * self.K, 4)
        self.linear = nn.Linear(self.L * self.K, num_classes)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, num_classes),
            nn.Sigmoid()
        )

    def set_fea_extractor(self, chkpt_path):
        print('loading checkpoint model')
        chkpt = torch.load(chkpt_path)
        new_chkpt=OrderedDict()
        for k, v in chkpt.items():
            name = k[7:] # remove module
            new_chkpt[name] = v
        self.resnet.load_state_dict(new_chkpt)

    def forward(self, x, flag=1):
        H = self.resnet(x) # NxL

        out, out_c, alpha = self.att_layer(H, self.linear.weight, self.linear.bias, flag)
        out = out.mean(0, keepdim=True)

        y = self.linear(out)
        return y, out_c, alpha
        
