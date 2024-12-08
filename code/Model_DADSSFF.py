import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from Attention_Module import CAM_Module, PAM_Module

class DADSSFF(nn.Module):
    def __init__(self, bands, num_classes):
        super(DADSSFF, self).__init__()

        self.num_classes = num_classes

        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=bands, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),        
            nn.AvgPool2d(kernel_size=3,  stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64,    out_channels=128, kernel_size=3, stride=1, padding=1, bias=False), 
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        self.sequential1=nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512,  kernel_size=3, stride=1, padding=0, bias=False),           
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=256,  kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=128,  kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        self.sequential2 = nn.Sequential(
            nn.Conv2d(in_channels=bands, out_channels=64,  kernel_size=3, stride=1, padding=0, bias=False),        
            nn.AvgPool2d(kernel_size=3,  stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64,    out_channels=128, kernel_size=2, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))

        self.CAM  = CAM_Module()
        self.PAM  = PAM_Module(in_dim=bands*2)     

        self.sequential3 = nn.Sequential(
            nn.Conv2d(in_channels=bands*2, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False),        
            nn.AvgPool2d(kernel_size=3,  stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,     out_channels=256,  kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,     out_channels=128,  kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5))
        
        self.fc=nn.Sequential(
            nn.Linear(512*3, 16),
            nn.ReLU(inplace=True), 
            nn.Linear(16, num_classes))

    def forward(self, Time1, Time2):
        '''
            The shape here is an example of the China dataset.
        '''
        
        ## 1. Domain Alignment
        Time1_fea = self.CNN(Time1)                                # torch.Size([64, 128, 5, 5])
        Time2_fea = self.CNN(Time2)
        loss_Mean = (Time1_fea.mean()-Time2_fea.mean())**2*self.num_classes
        loss_CORAL= self.CORAL(Time1_fea, Time2_fea) 
        loss_DA   = loss_Mean + loss_CORAL

        dif_fea = (Time1_fea - Time2_fea)**2*self.num_classes
        out_dif_fea = self.sequential1(dif_fea)                    # torch.Size([64, 128, 2, 2])
        out_dif_fea = out_dif_fea.view(out_dif_fea.shape[0], -1)   # torch.Size([64, 512]) 

        ## 2. Spectral Features
        dif_img = (Time1 - Time2)**2*self.num_classes              # torch.Size([64, 154, 7, 7])
        dif_img = self.CAM(dif_img)
        out_dif_img  = self.sequential2(dif_img)                   # torch.Size([64, 128, 2, 2])
        out_dif_img  = out_dif_img.view(out_dif_img.shape[0], -1)  # torch.Size([64, 512])
        loss_fea_img = self.KL(out_dif_fea,  out_dif_img)

        ## 3. Spatial Features
        con_fea = torch.cat((Time1, Time2), dim=1)                 # torch.Size([64, 308, 7, 7])
        out_con_fea  = self.PAM(con_fea)                           # torch.Size([64, 308, 7, 7])
        out_con_fea  = self.sequential3(out_con_fea)               # torch.Size([64, 128, 2, 2])
        out_con_fea  = out_con_fea.view(out_con_fea.shape[0], -1)  # torch.Size([64, 512])
        loss_con_fea = self.KL(out_dif_fea,  out_con_fea)

        ## 4. Features Fusion
        cosine_sim1 = F.cosine_similarity(out_dif_fea, out_dif_img, dim=1).mean()
        cosine_sim2 = F.cosine_similarity(out_dif_fea, out_con_fea, dim=1).mean()
        out = torch.cat((out_dif_fea, cosine_sim1*out_dif_img, cosine_sim2*out_con_fea), dim=1) # torch.Size([64, 1536])
        out = self.fc(out) 

        ## 5. KL divergence
        loss_KLD = cosine_sim1*loss_fea_img + cosine_sim2*loss_con_fea
        

        return out, loss_DA, loss_KLD


    def CORAL(self, Time1, Time2):
        ''' Reference:
                Baochen Sun, and  Saenko Kate, "Deep CORAL: Correlation alignment for deep domain adaptation," 
                In Computer Vision-ECCV 2016 Workshops: Amsterdam, The Netherlands, Proceedings,
                Part III 14. Springer International Publishing., October 8-10 and 15-16, 2016, pp: 443-450.
            Code:
                https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA
        '''
        Time1 = Time1.view(Time1.size(0), -1)
        Time2 = Time2.view(Time2.size(0), -1)

        n, dim=Time1.data.shape
        m, _  =Time2.data.shape

        # Time1 covariance
        N1=torch.mean(Time1, 0, keepdim=True) - Time1
        M1=N1.t() @ N1/(n-1)

        # Time2 covariance
        N2=torch.mean(Time2, 0, keepdim=True) - Time2
        M2=N2.t() @ N2/(m-1)

        loss       = torch.mul((M1-M2), (M1-M2))
        loss_coral = torch.sum(loss)/(4*dim*dim)
        
        return loss_coral
    

    def KL(self, Time1, Time2):
        '''
            here using PyTorch's built-in kl_div function
        '''
        Time1 = Time1.view(Time1.size(0), -1)
        Time2 = Time2.view(Time2.size(0), -1)

        _, dim=Time1.data.shape

        Time1_out_prob = F.softmax(Time1, dim=1)
        Time2_out_prob = F.softmax(Time2, dim=1)

        ## Calculate the KL divergence
        kl_divergence = F.kl_div(Time1_out_prob.log(), Time2_out_prob, reduction='batchmean')/(dim)

        return kl_divergence 