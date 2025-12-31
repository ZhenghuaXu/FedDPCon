import argparse
import logging
import os
import random
import shutil
import sys
import time
import copy

import numpy as np
import torch
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

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator,TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import val_HAM_2D
from SemiClientSetting import ClientsGroup
from sklearn.model_selection import StratifiedShuffleSplit
from SemiClientSetting import Cancer,model_dist
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_idea', help='model_name')
parser.add_argument('--num_classes', type=int,  default=3,
                    help='output channel of network')
parser.add_argument('--fully_batch_size', type=int, default=12,
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
parser.add_argument('--fully_num_of_clients', type=int, default=1,
                    help='number of fully supervised clients')
parser.add_argument('--semi_num_of_clients', type=int, default=3,
                    help='number of fully supervised clients')
parser.add_argument('--fully_cfraction', type=float, default=1,
                    help='the baifenbi of all clients')
parser.add_argument('--semi_cfraction', type=float, default=1,
                    help='the baifenbi of all clients')
parser.add_argument('--pre_num_comm', type=int, default=48,
                    help='the number of pre community')
parser.add_argument('--num_comm', type=int, default=300,
                    help='the number of community')
parser.add_argument('--semi_local_epoch', type=int, default=10,
                    help='the number of semi local epoch')
parser.add_argument('--fully_local_epoch', type=int, default=10,
                    help='number of fully loacl supervised epoch')
parser.add_argument('--batchsize', type=int, default=12,
                    help='the number of local batchsize')
parser.add_argument('--Fed_batch_size', type=int, default=12,
                    help='the number of local batchsize')
parser.add_argument('--step_three_num_comm', type=int, default=90,
                    help='the number of step3 community')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--lower',type=bool,default=True,help='Is every client own same dataset？')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=300.0, help='consistency_rampup')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'


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

CLIENTS = ['ROSENDAHL', 'VIDIR_MODERN', 'VIENNA_DIAS', 'VIDIR_MOLEMAX']

DIR1_VAL ='/mnt/sdd/dataset/data/ROSENDAHL/val/'
DIR2_VAL ='/mnt/sdd/dataset/data/VIDIR_MODERN/val/'
DIR3_VAL ='/mnt/sdd/dataset/data/VIENNA_DIAS/val/'
DIR4_VAL ='/mnt/sdd/dataset/data/VIDIR_MOLEMAX/val/'
DIR_VAL = [DIR1_VAL,DIR2_VAL,DIR3_VAL,DIR4_VAL]
DIR_DATA1_VAL = DIR1_VAL +'image/'
DIR_GT1_VAL = DIR1_VAL +'mask/'
DIR_DATA2_VAL = DIR2_VAL +'image/'
DIR_GT2_VAL = DIR2_VAL +'mask/'
DIR_DATA3_VAL = DIR3_VAL +'image/'
DIR_GT3_VAL = DIR3_VAL +'mask/'
DIR_DATA4_VAL = DIR4_VAL +'image/'
DIR_GT4_VAL = DIR4_VAL +'mask/'

data_files1_VAL,mask_files1_VAL = os.listdir(DIR_DATA1_VAL),os.listdir(DIR_GT1_VAL)
data_files2_VAL,mask_files2_VAL = os.listdir(DIR_DATA2_VAL),os.listdir(DIR_GT2_VAL)
data_files3_VAL,mask_files3_VAL = os.listdir(DIR_DATA3_VAL),os.listdir(DIR_GT3_VAL)
data_files4_VAL,mask_files4_VAL = os.listdir(DIR_DATA4_VAL),os.listdir(DIR_GT4_VAL)
data_files_VAL = [data_files1_VAL, data_files2_VAL, data_files3_VAL, data_files4_VAL]
mask_files_VAL = [mask_files1_VAL, mask_files2_VAL, mask_files3_VAL, mask_files4_VAL]

split_dataset=dict()

training_clients, val_clients = dict(), dict()
def extract_number(filename):
    return int(filename.split('_')[-1].split('.')[0])

for order, client in enumerate(CLIENTS):
    train_image_files = sorted(os.listdir(DIR[order]+'image/'), key=extract_number)
    train_mask_files = sorted(os.listdir(DIR[order]+'mask/'), key=extract_number)
    val_image_files = sorted(os.listdir(DIR_VAL[order]+'image/'), key=extract_number)
    val_mask_files = sorted(os.listdir(DIR_VAL[order]+'mask/'), key=extract_number)

    x_train, y_train = [DIR[order]+'image/'+f for f in train_image_files], [DIR[order]+'mask/'+f for f in train_mask_files]

    x_val,y_val=[DIR_VAL[order]+'image/'+f for f in val_image_files], [DIR_VAL[order]+'mask/'+f for f in val_mask_files]
    
    split_dataset[client+'_train']=Cancer(x_train, y_train, train=True, CROP_SIZE=CROP_SIZE)

    split_dataset[client+'_val'] =Cancer(x_val, y_val, train=False,CROP_SIZE=CROP_SIZE)


train_dict = {key: value for key, value in split_dataset.items() if 'train' in key}
val_dict = {key: value for key, value in split_dataset.items() if 'val' in key}


def get_current_consistency_weight(epoch):
        return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup) 
val_list={}
db_val={}
for number,val_data in enumerate(val_dict.values()):
    db_val['client'+str(number)] = val_data
    val_list['client'+str(number)] = DataLoader(val_data, batch_size=1, shuffle=True)

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    
    def create_model(ema=False):
        model = net_factory(net_type=args.model, in_chns=3,
                            class_num=2)
        return model

    model = create_model()
    model_un=create_model()
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7]).cuda()
    model_un=torch.nn.DataParallel(model_un, device_ids=[0,1,2,3,4,5,6,7]).cuda()
    mlp = torch.nn.Sequential(
        torch.nn.Linear(576, 128),  
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(128, 64)    
    ).cuda()
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(2)
    MSE_loss=nn.MSELoss()
    bce_loss=nn.BCELoss()
    myClients = ClientsGroup('HAM', dev, train_dict)
    global_parameters = {}
    prev_w_self_dict = {}
    prev_w_lab_dict = {}
    for key, var in model.state_dict().items():
        global_parameters[key] = var.clone()

#-------------------------------------------------------------------------------Step1-----------------------------------------------------------------------------------------
    un_ListClinet=[0,1,3]
    lab_ListClinet=[2]
    if (len(lab_ListClinet)==1):
        model.load_state_dict(torch.load('/mnt/sdd/code4.0/HAM2Weight.pth'))
    elif (len(lab_ListClinet)==2):
        model.load_state_dict(torch.load('/mnt/sdd/code4.0/HAM2-3Weight.pth'))

    L_q_dict = {}
    client_optimizers = {}   
    client_optim_states = {}  
    opt_lr = base_lr / 10
    opt_momentum = 0.9
    opt_wd = 0.0001

    all_client_ids = ['client{}'.format(j) for j in (un_ListClinet + lab_ListClinet)]

    for cid in all_client_ids:
        if cid in ['client{}'.format(j) for j in lab_ListClinet]:
            client_optimizers[cid] = optim.SGD(
                list(model.parameters()),
                lr=opt_lr, momentum=opt_momentum, weight_decay=opt_wd
            )
        else:
            client_optimizers[cid] = optim.SGD(
                list(model.parameters()) + list(mlp.parameters()),
                lr=opt_lr, momentum=opt_momentum, weight_decay=opt_wd
            )
#------------------------------------------------------------------------------Step2-----------------------------------------------------------------------------------------
    global_parameters=model.state_dict()
    for client_idx in range(len(un_ListClinet)):
        prev_w_self_dict['client{}'.format(un_ListClinet[client_idx])] = copy.deepcopy(global_parameters)

    for client_idx in range(len(lab_ListClinet)):
        prev_w_lab_dict['client{}'.format(lab_ListClinet[client_idx])] = copy.deepcopy(global_parameters)
    
    for cur_comm in range(args.num_comm):
        un_clients_in_comm = ['client{}'.format(j) for j in un_ListClinet[0:len(un_ListClinet)]]
        lab_clients_in_comm = ['client{}'.format(j) for j in lab_ListClinet[0:len(lab_ListClinet)]]

        label_model_para=[]
        for client in lab_clients_in_comm:
            print("mix cross-supervised {} communicate round {}".format(client,cur_comm+1))
            weight=get_current_consistency_weight(cur_comm)
            if client in client_optim_states:
                client_optimizers[client].load_state_dict(client_optim_states[client])                                              
            local_parameters_fully,epoch_loss,supervised_batch_pro, L_q = myClients.clients_set[client].Fedidea_crossUpdate(args.fully_local_epoch, args.fully_batch_size, model,
                                                                         model_un,ce_loss,dice_loss, client_optimizers[client], global_parameters,prev_w_lab_dict[client],weight)
            prev_w_lab_dict[client]=copy.deepcopy(local_parameters_fully)
            client_optim_states[client] = copy.deepcopy(client_optimizers[client].state_dict())
            label_model_para.append(local_parameters_fully)
            L_q_dict[client] = L_q 

        if len(label_model_para) == 1:
            local_label_agg = label_model_para

        else:
            agg_label_para = {}
            lab_sum = sum([L_q_dict[c] for c in lab_clients_in_comm]) + 1e-12
            for k in global_parameters.keys():
                agg_label_para[k] = sum([
                    (L_q_dict[c] / lab_sum) * prev_w_lab_dict[c][k]
                    for c in lab_clients_in_comm
                ])
            local_label_agg = [agg_label_para]

        for client in un_clients_in_comm:
            print("mix self-supervised {} self-supervised communicate round {}".format(client,cur_comm+1))
            if client in client_optim_states:
                client_optimizers[client].load_state_dict(client_optim_states[client])
            local_parameters_Fed,epoch_loss, L_q  = myClients.clients_set[client].Fedidea_selfUpdate(args.fully_local_epoch,args.Fed_batch_size, model,model_un,
                                                                          MSE_loss,dice_loss,bce_loss, client_optimizers[client], global_parameters,prev_w_self_dict[client],supervised_batch_pro,local_label_agg,len(lab_ListClinet),number=cur_comm,mlp=mlp)
            prev_w_self_dict[client]=copy.deepcopy(local_parameters_Fed)
            client_optim_states[client] = copy.deepcopy(client_optimizers[client].state_dict())
            L_q_dict[client] = L_q  

        for k in global_parameters:
            weight=get_current_consistency_weight(cur_comm)
            lab_sum = sum([L_q_dict[c] for c in lab_clients_in_comm]) if len(lab_clients_in_comm) > 0 else 1.0
            un_sum  = sum([L_q_dict[c] for c in un_clients_in_comm])  if len(un_clients_in_comm) > 0  else 1.0

            global_parameters[k] = (
                sum([
                    (1.0 - weight) * (L_q_dict[c] / (lab_sum + 1e-12)) * prev_w_lab_dict[c][k]
                    for c in lab_clients_in_comm
                ]) +
                sum([
                    weight * (L_q_dict[c] / (un_sum + 1e-12)) * prev_w_self_dict[c][k]
                    for c in un_clients_in_comm
                ])
            )
        with torch.no_grad():
            if (cur_comm + 1) % (1) == 0:
                mean_dice=0.0
                model.load_state_dict(global_parameters, strict=True)
                metric_list = 0.0
                best_performance = 0.0
                for number,(val_client, db) in enumerate(zip(val_list.values(),db_val.values())):
                    for i_batch, (img, mask) in enumerate(val_client):
                        metric_i = val_HAM_2D(
                            i_batch,img, mask, model, classes=3)
                        metric_list += np.array(metric_i)
                    metric_list = metric_list / len(db)
                    for i,inner_list in enumerate(metric_list):
                        dice_performance=inner_list[0]
                        hd95=inner_list[1]
                        mean_dice += dice_performance
                        if i==0:
                            print('number %d : iteration %d : dice : %f hd95 : %f ' % (number,cur_comm, dice_performance, hd95))
                        if number==3:
                            print('mean dice = ', mean_dice/4)
    print( "Training Finished!")

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, 2, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
