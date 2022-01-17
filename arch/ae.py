# Code for Overcpmplete SAR despeckling
# Author: Malsha Perera
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import math



class OUSAR(nn.Module):
    
    def __init__(self):
        super(OUSAR, self).__init__()
        
        self.encoder1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.bne1 = nn.InstanceNorm2d(32)
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.bne2 = nn.InstanceNorm2d(64)
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bne3 = nn.InstanceNorm2d(128)
        self.encoder4=   nn.Conv2d(128, 512, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(512, 1024, 3, stride=1, padding=1)



        self.decoder1 =   nn.Conv2d(1024, 512, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.bnd1 = nn.InstanceNorm2d(64)
        self.decoder2 =   nn.Conv2d(512, 128, 3, stride=1, padding=1)
        self.bnd2 = nn.InstanceNorm2d(32)
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bnd3 = nn.InstanceNorm2d(16)
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)

        # self.decoderf1 =   nn.Conv2d(16, 128, 2, stride=1, padding=1)
        self.decoderf1 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bndf1 = nn.InstanceNorm2d(64)
        self.decoderf2=   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.bndf2 = nn.InstanceNorm2d(32)
        self.decoderf3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.bndf3 = nn.InstanceNorm2d(16)
        self.decoderf5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)
        # self.decoderf5 =   nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.encoderf1 =   nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.bnef1 = nn.InstanceNorm2d(32)
        self.encoderf2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bnef2 = nn.InstanceNorm2d(64)
        self.encoderf3 =   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bnef3 = nn.InstanceNorm2d(128)
        self.encoderf4 =   nn.Conv2d(128, 16, 3, stride=1, padding=1)
        # self.encoderf5 =   nn.Conv2d(128, 128, 3, stride=1, padding=1)
        
        self.final = nn.Conv2d(16,1,1,stride=1,padding=0)
        self.bnf = nn.InstanceNorm2d(3)

        self.tmp1 = nn.Conv2d(16,32,1,stride=1,padding=0)
        self.bnt1 = nn.InstanceNorm2d(32)
        self.tmp2 = nn.Conv2d(32,32,1,stride=1,padding=0)
        # self.bnt2 = nn.BatchNorm2d(32)
        self.tmp3 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.tmp4 = nn.Conv2d(16,32,1,stride=1,padding=0)


        self.tmpf3 = nn.Conv2d(128,32,1,stride=1,padding=0)
        self.tmpf2 = nn.Conv2d(64,32,1,stride=1,padding=0)
        self.tmpf1 = nn.Conv2d(32,32,1,stride=1,padding=0)
        self.tan = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out1 = F.relu(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))
        t1 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        o1 = out1
        out1 = F.relu(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear'))
        
        t2 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')
        o2 = out1        
        
        out1 = F.relu(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear')) 
        t3 = F.interpolate(out1,scale_factor=(0.0625,0.0625),mode ='bilinear')
        
        # U-NET encoder start
        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        #Fusing all feature maps from K-NET

        out = torch.add(out,torch.add(self.tmpf3(t3),torch.add(t1,self.tmpf2(t2))))
        
        u1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        u2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        u3=out
        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        u4=out
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))

        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u4)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u3)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,u1)
        
        # Start K-Net decoder

        out1 = F.relu(F.max_pool2d(self.decoderf1(out1),2,2))
        out1 = torch.add(out1,o2)
        
        t3 = F.interpolate(out1,scale_factor=(0.125,0.125),mode ='bilinear')

        out1 = F.relu(F.max_pool2d(self.decoderf2(out1),2,2))
        out1 = torch.add(out1,o1)

        t2 = F.interpolate(out1,scale_factor=(0.25,0.25),mode ='bilinear')
        
        out1 = F.relu(F.max_pool2d(self.decoderf3(out1),2,2))

        t1 = F.interpolate(out1,scale_factor=(0.5,0.5),mode ='bilinear')
        # Fusing all layers at the last layer of decoder
        # print(t1.shape,t2.shape,t3.shape,out.shape)
        out = torch.add(out,torch.add(self.tmp3(t3),torch.add(self.tmp1(t1),self.tmp2(t2))))

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,out1)

        out = F.relu(self.final(out))

        # out = torch.div(x,out + 1e-9)
        # out = F.tanh(out)

        return self.tan(out)


