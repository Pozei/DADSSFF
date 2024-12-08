import torch
import numpy as np
import torch.nn as nn

''' Reference:
            Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,
            and Hanqing Lu, ”Dual attention network for scene segmentation,” in
            Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.(CVPR), 2019, pp:3146-315
    Code:        
            https://github.com/junfu1115/DANet/
'''

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()

        self.gamma   = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x  : input feature maps(B X C X H X W)
            returns:
                out: attention value + input feature
                attention: B X C X C
        """
        batchsize, Channel, height, width = x.shape

        CAM_query = x.view(batchsize, Channel, -1)
        CAM_key   = x.view(batchsize, Channel, -1).permute(0, 2, 1)
        CAM_value = x.view(batchsize, Channel, -1)

        CAM_energy = torch.bmm(CAM_query, CAM_key)  ## matrix product 
        CAM_energy = torch.max(CAM_energy, -1, keepdim=True)[0].expand_as(CAM_energy) - CAM_energy #其中通过减去最大值可以避免指数运算溢出，并且有助于捕捉到更多的细微差异。
        CAM_attention  = self.softmax(CAM_energy) ## Normalize
        
        out = torch.bmm(CAM_attention , CAM_value)
        out = self.softmax(out) 
        out = out.view(batchsize, Channel, height, width)

        out = x + self.gamma*out
        
        return out
    
class PAM_Module(nn.Module):
    """ Position attention module"""
    def __init__(self, in_dim):

        super(PAM_Module, self).__init__()


        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv   = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim,    kernel_size=1)

        self.gamma      = nn.Parameter(torch.zeros(1))
        self.softmax    = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        batchsize, Channel, height, width = x.size()

        query_out = self.query_conv(x)
        query_out = query_out.view(batchsize, -1, height*width)
        PAM_query = query_out.permute(0, 2, 1)

        key_out = self.key_conv(x)
        PAM_key = key_out.view(batchsize, -1, height*width)

        PAM_energy = torch.bmm(PAM_query, PAM_key)
        PAM_energy = PAM_energy.permute(0, 2, 1)
        PAM_attention = torch.max(PAM_energy, -1, keepdim=True)[0].expand_as(PAM_energy)-PAM_energy
        PAM_attention = self.softmax(PAM_attention)

        value_out = self.value_conv(x)
        PAM_value = value_out.view(batchsize, -1, height*width)
        PAM_value = self.softmax(PAM_value)

        out = torch.bmm(PAM_value, PAM_attention)
        out = self.softmax(out)
        out = out.view(batchsize, Channel, height, width)

        out = x + self.gamma*out

        return out

class CoAM_Module(nn.Module):
    """ Correlation attention module"""
    def __init__(self,):
        super(CoAM_Module, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward (self, x1, x2):
        """
            inputs:
                x1 : input feature maps1( B X C X H X W)
                x2 : input feature maps2( B X C X H X W)
            returns:
                out: 
                attention: B X C X C
        """

        batchsize, Channel, height, width = x1.size()

        CoAM_query = x1.view(batchsize, Channel, -1)
        CoAM_key   = x1.view(batchsize, Channel, -1).permute(0, 2, 1)
        
        CoAM_energy= torch.bmm(CoAM_query, CoAM_key)
        CoAM_attention = torch.max(CoAM_energy, -1, keepdim=True)[0].expand_as(CoAM_energy)-CoAM_energy

        CoAM_value = x2.view(batchsize, Channel, -1)
        out = torch.bmm(CoAM_attention, CoAM_value)
        out = self.softmax(out)
        out = out.view(batchsize, Channel, height, width)

        out = out + self.gamma*x2

        return out