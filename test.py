# Code for testing real SAR images
# Author: Malsha Perera
import argparse
import torch
import torchvision
from torch import nn
import math
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
from torchvision.transforms import functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import torch.nn.init as init
from arch.ae import OUSAR
import cv2
from functools import partial
from random import randint
from scipy.io import loadmat, savemat




parser = argparse.ArgumentParser(description='KiU-Net')

parser.add_argument('--cuda', default="on", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save_path', required=True , type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', type=str,
                    help='model name')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--loadmodel', default='load', type=str)
parser.add_argument('--full_image', default='on',type=str)


args = parser.parse_args()


modelname = args.model
loaddirec = args.loadmodel
save_path = args.save_path
full_image = args.full_image
print(full_image)




device = torch.device("cuda")

model = OUSAR()

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model,device_ids=[0,1]).cuda()
model.to(device)


model.load_state_dict(torch.load(loaddirec))
model.eval()



if not os.path.isdir(save_path):
                
    os.makedirs(save_path)





    

im_file = './test_images/test_01.png'

img = cv2.imread(im_file,0)

noisy_im = (np.float32(img)+1.0)/256.0



x = np.float32(noisy_im)
x = F.to_tensor(x)
x = x.unsqueeze(0)


pred_im = model(x)
tmp = pred_im.detach().cpu().numpy()

tmp = tmp.squeeze()
tmp = tmp*256 -1

filename_out = 'test_01_results.png'
filepath_out = save_path + filename_out

cv2.imwrite(filepath_out,tmp)


print('done')