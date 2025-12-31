import argparse
import logging
import os
import random
import shutil
import sys
import time
import torch.nn.functional as F
import copy

import numpy as np
import torch
from torch import Tensor
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import math
from dataloaders import utils
from typing import Sequence
from functools import reduce
from dataloaders.dataset import BaseDataSets, RandomGenerator,TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_idea', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--fully_batch_size', type=int, default=24,
                    help='fully supervised client batchsize')
parser.add_argument('--labeled_bs', type=int, default=6,
                    help='labeled_batch_size per gpu')
parser.add_argument('--semi_batch_size', type=int, default=12,
                    help='the number of semi-supervised client local whole batchsize，including unlabel and label')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument('--fully_num_of_clients', type=int, default=2,
                    help='number of fully supervised clients')
parser.add_argument('--semi_num_of_clients', type=int, default=2,
                    help='number of fully supervised clients')
parser.add_argument('--fully_cfraction', type=float, default=1,
                    help='the baifenbi of all clients')
parser.add_argument('--semi_cfraction', type=float, default=0.3,
                    help='the baifenbi of all clients')
parser.add_argument('--pre_num_comm', type=int, default=1,
                    help='the number of pre community')
parser.add_argument('--num_comm', type=int, default=200,
                    help='the number of community')
parser.add_argument('--semi_local_epoch', type=int, default=10,
                    help='the number of semi local epoch')
parser.add_argument('--fully_local_epoch', type=int, default=10,
                    help='number of fully loacl supervised epoch')
parser.add_argument('--batchsize', type=int, default=12,
                    help='the number of local batchsize')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--lower',type=bool,default=True,help='Is every client own same dataset？')
args = parser.parse_args()

class Cancer(Dataset):
    def __init__(self, im_path, mask_path, train=False, \
                IMAGE_SIZE=(384,384), CROP_SIZE=(224,224), 
                noisy=True):
        self.data = im_path
        self.label = mask_path
        self.train = train
        self.IMAGE_SIZE = IMAGE_SIZE
        self.CROP_SIZE = CROP_SIZE
        self.noisy = noisy

    def normalize_image(image):
        image = image.astype(np.float32)
        normalized_image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)

        return normalized_image
    
    def transform(self, image, mask, train):
        resize_image = Resize(self.IMAGE_SIZE)
        resize_label = Resize(self.IMAGE_SIZE,interpolation=Image.NEAREST)
        image = resize_image(image)
        mask = resize_label(mask)
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        mask = Image.open(self.label[idx]).convert('L')
        x, y = self.transform(image, mask, self.train)
        return x, y


IMAGE_SIZE = (256, 256)
CROP_SIZE = (224, 224)

DIR1 ='/mnt/sdd/dataset/data/ROSENDAHL/train/'
DIR2 ='/mnt/sdd/dataset/data/VIDIR_MODERN/train/'
DIR3 ='/mnt/sdd/dataset/data/VIENNA_DIAS/train/'
DIR4 ='/mnt/sdd/dataset/data/VIDIR_MOLEMAX/train/'
DIR = [DIR1,DIR2,DIR3,DIR4]
DIR_DATA1 = DIR1 +'image/'
DIR_GT1 = DIR1 +'mask/'
DIR_DATA2 = DIR2 +'image/'
DIR_GT2 = DIR2 +'mask/'
DIR_DATA3 = DIR3 +'image/'
DIR_GT3 = DIR3 +'mask/'
DIR_DATA4 = DIR4 +'image/'
DIR_GT4 = DIR4 +'mask/'
data_files1,mask_files1 = os.listdir(DIR_DATA1),os.listdir(DIR_GT1)
data_files2,mask_files2 = os.listdir(DIR_DATA2),os.listdir(DIR_GT2)
data_files3,mask_files3 = os.listdir(DIR_DATA3),os.listdir(DIR_GT3)
data_files4,mask_files4 = os.listdir(DIR_DATA4),os.listdir(DIR_GT4)
data_files = [data_files1, data_files2, data_files3, data_files4]
mask_files = [mask_files1, mask_files2, mask_files3, mask_files4]

CLIENTS = ['1', '2', '3', '4']

split_dataset=dict()

for order, client in enumerate(CLIENTS):
    x_, y_ = [DIR[order]+'image/'+f for f in data_files[order]], [DIR[order]+'mask/'+f for f in mask_files[order]]
    x_train, x_val, y_train, y_val = train_test_split( 
    x_, y_, test_size=1-TRAIN_RATIO, random_state=RS)

    split_dataset[client+'_train']=Cancer(x_train, y_train, train=True,\
                                          IMAGE_SIZE=IMAGE_SIZE\
                                           , CROP_SIZE=CROP_SIZE)
    
    split_dataset[client+'_val'] =Cancer(x_val, y_val, train=False,\
                                          IMAGE_SIZE=IMAGE_SIZE\
                                           , CROP_SIZE=CROP_SIZE)

def model_dist(w_1, w_2):
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1:
        dist = torch.norm(w_1[key].cuda().float() - w_2[key].cuda().float()) 
        dist_total += dist.cpu()
    return dist_total.cpu().item()



class client(object):
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None
    
    def Fedidea_selfUpdate(self, localEpoch, localBatchSize, model,teach_model, criterion,dice_loss,bce_loss, optimizer, global_parameters,prev_w_self_dict,supervised_batch_pro,label_model_para,len_lab_ListClinet,number,mlp): 
        model.load_state_dict(global_parameters, strict=True)
        self.train_dl=DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        epoch_loss=[]
        global_model = copy.deepcopy(model)
        prev_local_model = copy.deepcopy(model)
        prev_local_model.load_state_dict(copy.deepcopy(prev_w_self_dict))
        inter_layers=['up4']
        temperature=0.5
        mu=5
        batch_pro=[]

        def project_feature_patch(feat, mlp, patch_size=16):
            B, C, H, W = feat.shape
            assert H % patch_size == 0 and W % patch_size == 0

            pooled = F.avg_pool2d(feat, kernel_size=patch_size, stride=patch_size)  

            pooled = pooled.flatten(2).contiguous()   
            pooled_reshaped = pooled.reshape(B*C, -1)

            phi = mlp(pooled_reshaped)             
            phi = phi.reshape(B, C, -1).contiguous()      

            return phi

        def inter_layer_contrastive_loss(local_out, global_out, prev_out, mlp, temperature=0.5):

            phi_k      = project_feature_patch(local_out, mlp)   
            phi_k_l    = project_feature_patch(global_out, mlp)   
            phi_k_old  = project_feature_patch(prev_out, mlp)    


            pos = F.cosine_similarity(phi_k, phi_k_l, dim=-1) / temperature  
            neg = F.cosine_similarity(phi_k, phi_k_old, dim=-1) / temperature 


            l_k = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg))) 
            l_k = l_k.mean(dim=0)  


            sim_k = pos.mean(dim=0)       
            v_k = F.softmax(sim_k, dim=0) 

            loss_ir = torch.sum(v_k * l_k) 

            return loss_ir
        def strong_augmentation_per_sample(volume_batch):
            x = volume_batch.clone()
            flip_dims_list = []  
            for i in range(x.shape[0]):
                flip_dims = []
                if random.random() < 0.8:
                    noise = torch.randn_like(x[i]) * 0.3   
                    x[i] = x[i] + noise
                if random.random() < 0.8:
                    gamma = random.uniform(0.5, 1.5)
                    x[i] = torch.clamp(x[i], 0, 1)
                    x[i] = x[i] ** gamma
                if random.random() < 0.5:
                    x[i] = torch.flip(x[i], dims=[-1])
                    flip_dims.append(-1)
                if random.random() < 0.5:
                    x[i] = torch.flip(x[i], dims=[-2])
                    flip_dims.append(-2)
                flip_dims_list.append(flip_dims)
            return x, flip_dims_list
        for Unlabel_para in label_model_para:
            teach_model.load_state_dict(Unlabel_para)
            for epoch in range(localEpoch):
                batch_loss=[]
                for i_batch,(img, _) in enumerate(self.train_dl):
                    volume_batch = img
                    volume_batch = volume_batch.cuda()
                    device = volume_batch.device  
                    mlp = mlp.to(device)     
                    mlp.train()
                    noise = torch.randn_like(volume_batch) * 0.1    
                    volume_batch_noise=volume_batch + noise
                    volume_batch_noise=volume_batch_noise.cuda()
                    volume_strong, flip_dims_list = strong_augmentation_per_sample(volume_batch)
                    model.train()

                    outputs_non,local_out_non,local_fea_non=model(volume_batch)
                    local_feat = local_out_non['up4']  
                    outputs,local_out,local_fea=model(volume_batch_noise)

                    PreUnLabel,PreUnLabel_out,PreUnLabel_fea_low=teach_model(volume_batch_noise)
                    PreUnLabel_non,PreUnLabel_out_non,PreUnLabel_fea_low_non=teach_model(volume_batch)
                    PreUnLabel_out_feat = PreUnLabel_out_non['up4']

                    probs = torch.softmax(PreUnLabel, dim=1)  

                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  
                    lambda_uncertainty = 0.20  
                    mask_uncertainty = (entropy < lambda_uncertainty).float()  

                    mask = mask_uncertainty.unsqueeze(1).repeat(1, outputs.shape[1], 1, 1)

                    pseudo_label = (probs >= 0.75).float()  
                
                    pseudo_label_denoised = pseudo_label * mask  

                    for i in range(pseudo_label_denoised.shape[0]):
                        flip_dims = flip_dims_list[i]
                        for dim in flip_dims:
                            pseudo_label_denoised[i] = torch.flip(pseudo_label_denoised[i], dims=[dim])


                    outputs_prev_local,prev_local_out,prev_local_fea=prev_local_model(volume_batch)
                    prev_local_out_feat = prev_local_out['up4']
                    noise_outputs,noise_local_out,noise_local_fea=model(volume_strong)
   
                    pseudo_label_argmax = torch.argmax(pseudo_label_denoised, dim=1, keepdim=True) 
                    d_loss = dice_loss(noise_outputs, pseudo_label_argmax,softmax=True)
                    torch.cuda.empty_cache()
                    loss_ir = inter_layer_contrastive_loss(
                            local_out=local_feat,
                            global_out=PreUnLabel_out_feat,
                            prev_out=prev_local_out_feat,
                            mlp=mlp,
                            temperature=temperature
                        )

                    loss= mu * loss_ir+d_loss
                    batch_loss.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                epoch_loss.append(np.array(batch_loss).mean())
                torch.cuda.empty_cache()
        with torch.no_grad():
            model.eval()
            teach_model.eval()
            per_image_consistency = []  
            for img, _ in self.train_dl:
                volume_batch = img.cuda()

                pred_local, _, _ = model(volume_batch)
                prob_local = (torch.softmax(pred_local, dim=1) > 0.75).float()  

                pred_label, _, _ = teach_model(volume_batch)
                pred_label = (torch.softmax(pred_label, dim=1) > 0.75).float()  


                pred_local_label = torch.argmax(prob_local, dim=1)   
                pred_teacher_label = torch.argmax(pred_label, dim=1) 

                B = pred_local_label.shape[0]
                for b in range(B):
                    fg_local   = (pred_local_label[b] == 1) | (pred_local_label[b] == 2)
                    fg_teacher = (pred_teacher_label[b] == 1) | (pred_teacher_label[b] == 2)

                    valid = fg_local | fg_teacher
                    valid_pix = valid.sum().item()

                    if valid_pix == 0:
                        continue

                    same = (fg_local == fg_teacher) & valid
                    same_pix = same.sum().item()

                    per_image_consistency.append(same_pix / valid_pix)

            if len(per_image_consistency) > 0:
                L_u = sum(per_image_consistency) / len(per_image_consistency)
            else:
                L_u = 0.0
        return model.state_dict(),sum(epoch_loss) / len(epoch_loss) ,L_u   
    
    def Fedidea_crossUpdate(self,localEpoch,localBatchSize,model,teach_model,ce_loss,dice_loss,optimizer,global_parameters,prev_w_lab_dict,weight):
        model.load_state_dict(global_parameters,strict=True)
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        epoch_loss=[]
        cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        global_model = copy.deepcopy(model)
        prev_local_model = copy.deepcopy(model)
        prev_local_model.load_state_dict(copy.deepcopy(prev_w_lab_dict))
        batch_pro=[]
        for epoch in range(localEpoch):
            batch_loss=[]
            for i_batch, (img, mask) in enumerate(self.train_dl):
                    volume_batch, label_batch = img, mask
                    volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                    model.train()

                    PredFull,local_out,local_fea=model(volume_batch)
                    full_outputs_soft = torch.softmax(PredFull, dim=1)

                    ans=label_batch[:].long()
                    ans=torch.squeeze(ans, dim=1)
                    loss_dice = dice_loss(full_outputs_soft, label_batch)
                    superviseloss= loss_dice
                    loss_total= superviseloss
                    optimizer.zero_grad()
                    loss_total.backward()
                    optimizer.step()
            epoch_loss.append(np.array(batch_loss).mean())
        with torch.no_grad():    
            model.eval()
            per_image_acc = []   
            with torch.no_grad():
                for img, mask in self.train_dl:
                    img = img.cuda()
                    mask = mask.cuda()  

                    pred, _, _ = model(img)
                    pred_soft = (torch.softmax(pred, dim=1) > 0.75).float()  
                    pred_label = torch.argmax(pred_soft, dim=1)  

                    mask_label = mask.squeeze(1).long()  
                    B = pred_label.shape[0]
                    for b in range(B):
                        fg_mask = (mask_label[b] == 1) | (mask_label[b] == 2)
                        total = fg_mask.sum().item()

                        if total == 0:   
                            continue

                        correct = ((pred_label[b] == mask_label[b]) & fg_mask).sum().item()
                        per_image_acc.append(correct / total)

                if len(per_image_acc) > 0:
                    L_q = sum(per_image_acc) / len(per_image_acc)
                else:
                    L_q = 0.0

        return model.state_dict(),sum(epoch_loss) / len(epoch_loss),batch_pro,L_q
    
    def local_val(self):
        pass

class ClientsGroup(object):
    def __init__(self, dataSetName, dev,dataset):
        self.data_set_name = dataSetName
        self.dataset=dataset
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        for number,client_data in enumerate(self.dataset.values()):
            someone=client(client_data,self.dev)
            self.clients_set['client{}'.format(number)] = someone

class ClientsGroupDomain(object):
    def __init__(self, dataSetName, dev,dataset):
        self.data_set_name = dataSetName
        self.dataset=dataset
        self.dev = dev
        self.clients_set = {}
        self.test_data_loader = None
        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        for name,client_data in enumerate(self.dataset.items()):
            someone=client(client_data,self.dev)
            self.clients_set['client{}'.format(name)] = someone     

