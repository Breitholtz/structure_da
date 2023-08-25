import numpy as np
import torch
from torch import nn
from network import *


#====================================
from torch.utils.data import Dataset
class dataset(Dataset):
    def __init__(self,data,labels,structure_data):
        self.data=data
        self.label=labels
        self.A=structure_data
# something for later..
# class cub_dataset(Dataset):
#     def __init__(self,data_file,structure_data):
#         #self.data=data_file[]
#         #self.label
#====================================


class BaseModel(nn.Module):
    
    def __init__(self,opt):
        super(BaseModel, self).__init__()
        self.opt=opt
        
        
    def reconstruction_loss(self):
        # here we should do the reconstruction loss from the structure information
        print("hi")
    def classification_loss(self):
        ## classification loss
        print("hi")
    def train(self,dataloader):
        print("hi")
        #for data in dataloader
