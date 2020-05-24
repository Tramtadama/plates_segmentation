import json
import math
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from aug_pool import aug_pool
from utils import create_img_names
from datafactory import Znacky_set
from losses import iou, BCEDiceLoss
import argparse

import os
for_inf_test = os.listdir('data/images')
data_folder = 'data/'
all_names, img_names_test, img_names_train, img_names_val = create_img_names(
        data_folder+'full_data.json', 
        data_folder+'full_test_proper.csv',
        data_folder+'full_train.csv', data_folder+'full_val_proper.csv')


parser = argparse.ArgumentParser(description='plates train')
parser.add_argument('--epochs', default=1, type=int)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--cuda', default=False, type=bool)
parser.add_argument('--train_bs', default=1, type=int)
parser.add_argument('--inf_bs', default=1, type=int)
parser.add_argument('--test_mode', default=False, type=bool)
args = parser.parse_args()

with open(args.data + 'full_data.json') as json_file:
    data_dict = json.load(json_file)

if args.test_mode:
    train_cut = []
    for name in img_names_train:
        if name in for_inf_test:
            train_cut.append(name)

    val_cut = []
    for name in img_names_val:
        if name in for_inf_test:
            val_cut.append(name)

    train_names = train_cut[:8]
    val_names = val_cut[:1]
else:
    train_names = img_names_train
    val_names = img_names_val

shape = (224, 224)
train_batch_size = 8
inf_batch_size = 1

loss = BCEDiceLoss()

train_dset = Znacky_set(shape, train_names, y=data_dict, DIR=args.data
        +'images/', aug_pool=aug_pool)
val_dset = Znacky_set(shape, val_names, y=data_dict, DIR=args.data +
        'images/')
test_dset = Znacky_set(shape, img_names_test, y=data_dict, DIR=args.data +
        'images/')

train_loader = DataLoader(train_dset, batch_size=args.train_bs, shuffle=True)
val_loader = DataLoader(val_dset, batch_size=args.inf_bs, shuffle=False)
test_loader = DataLoader(test_dset, batch_size=args.inf_bs, shuffle=False)

model = smp.Unet('efficientnet-b7', encoder_weights='imagenet', activation=None)

epochs = args.epochs
no_batches_val = math.ceil(len(img_names_val)/args.inf_bs)
no_batches_train = math.ceil(len(img_names_train)/args.train_bs)

lr = 0.03
optim = torch.optim.Adam(model.parameters(), lr=lr)

last_epoch = 0
init_epoch = 0
train_loss_list = []
lrs = []
val_loss_history = []
train_loss_history = []
val_iou_history = []
train_iou_history = [] 
Tmax = 80
Tres = 80
Tmult = 2
scheduler = CosineAnnealingLR(optim, Tmax, eta_min=0.003, last_epoch=-1)

for epoch in tqdm(range(1+init_epoch, epochs + init_epoch + 1)):
    
    lrs.append(optim.param_groups[0]['lr'])
    
    if args.cuda:
        model = model.train().cuda()
    else:
        model = model.train()

    batch_train_loss_history = []
    batch_train_iou = []
    train_loss = 0
    train_iou = 0

    for (imgs, labels, _) in tqdm(train_loader):
        
        if args.cuda:
            imgs, labels = imgs.cuda(), labels.cuda()
        optim.zero_grad()
        out = model(imgs)
        
        batch_train_loss = loss(out, labels)

        if args.cuda:
            out = out.cpu()
            labels = labels.cpu()
            batch_train_loss = batch_train_loss.cpu()

        batch_train_iou = iou(out, labels, 
                threshold=0.5, activation="sigmoid")

        train_loss += batch_train_loss.item()
        train_iou += batch_train_iou.item()

        batch_train_loss.backward()
        optim.step()

    scheduler.step()

    if epoch % Tres == 0:
        Tres = Tmax*Tmult + Tres
        Tmax = Tmax*Tmult
        optim.param_groups[0]['lr'] = lr
        scheduler = CosineAnnealingLR(optim, Tmax, eta_min=0.003)

    train_loss_history.append(train_loss/no_batches_train)
    train_iou_history.append(train_iou/no_batches_train)

    print('train_loss', train_loss/no_batches_train, 'train_iou',
            train_iou/no_batches_train)

    val_loss = 0
    val_iou = 0
    
    if args.cuda:
        model.cpu().eval()
    else:
        model.eval()

    with torch.no_grad():
        for imgs, labels, img_pth in tqdm(val_loader):

            optim.zero_grad()

            pred = model(imgs)

            batch_val_loss = loss(pred, labels)
            batch_val_iou = iou(pred, labels, threshold=0.5,
                    activation="sigmoid")

            val_loss += batch_val_loss.item()
            val_iou += batch_val_iou.item()

        val_loss_history.append(val_loss/no_batches_val)
        val_iou_history.append(val_iou/no_batches_val)

        print('eval_loss', val_loss/no_batches_val,
                'eval_iou', val_iou/no_batches_val)

        torch.save({'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'lrs':lrs,
                    'val_loss_history':val_loss_history,
                    'train_loss_history':train_loss_history,
                    'val_iou_history':val_iou_history,
                    'train_iou_history':train_iou_history}
                    ,'models/efbn7_full_data' + str(epoch) + '.tar')
