import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_data_functions import TrainData
from val_data_functions import ValData
from utils import to_psnr, print_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
import os
import numpy as np
import random
import wandb 

from .transweather_model import Transweather,Transweather_base

plt.switch_backend('agg')

wandb.init(project="transweather", resume="allow", id="lmao")

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=2e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=18, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=200, type=int)

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
start_epoch = 0

#set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed) 
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nlambda_loss: {}'.format(learning_rate, crop_size,
      train_batch_size, val_batch_size, lambda_loss))


train_data_dir = './data/train/'
val_data_dir = './data/test/'

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# --- Define the network --- #
net = Transweather_base()


# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# --- Multi-GPU --- #
net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
# vgg_model = nn.DataParallel(vgg_model, device_ids=device_ids)
for param in vgg_model.parameters():
    param.requires_grad = False

checkpoint_path = '/content/drive/MyDrive/ml-data/TransWeather'

# --- Load the network weight --- #
if os.path.exists('{}/{}/'.format(checkpoint_path, exp_name))==False:     
    os.mkdir('{}/{}/'.format(checkpoint_path, exp_name))  
try:
    state = torch.load('{}/{}/latest'.format(checkpoint_path, exp_name))
    start_epoch = state['epoch']
    net.load_state_dict(state['model'])
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')


loss_network = LossNetwork(vgg_model)
loss_network.eval()

# --- Load training data and validation/test data --- #

### The following file should be placed inside the directory "./data/train/"

labeled_name = 'allweather.txt' # Change this based on the dataset you choose to train on

### The following files should be placed inside the directory "./data/test/"

val_filename1 = 'allweather.txt' # Change this based on the dataset you choose to test on

# --- Load training data and validation/test data --- #
lbl_train_data_loader = DataLoader(TrainData(crop_size, train_data_dir,labeled_name), batch_size=train_batch_size, shuffle=True, num_workers=8)
val_data_loader1 = DataLoader(ValData(val_data_dir,val_filename1), batch_size=val_batch_size, shuffle=False, num_workers=8)

# --- Previous PSNR and SSIM in testing --- #
# net.eval()


# old_val_psnr1, old_val_ssim1 = validation(net, val_data_loader1, device, exp_name)

# print('allweather old_val_psnr: {0:.2f}, old_val_ssim: {1:.4f}'.format(old_val_psnr1, old_val_ssim1))



# net.train()
for epoch in range(start_epoch ,num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)

    total_loss = 0.0
    loss_count = 0
#-------------------------------------------------------------------------------------------------------------
    from tqdm import tqdm
    for batch_id, train_data in tqdm(enumerate(lbl_train_data_loader)):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()

        # --- Forward + Backward + Optimize --- #
        net.train()
        pred_image = net(input_image)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)

        loss = smooth_loss + lambda_loss*perceptual_loss 

        total_loss += loss
        loss_count += 1

        real_loss = total_loss / loss_count

        loss.backward()
        optimizer.step()

        # --- To calculate average PSNR --- #
        psnr_list.extend(to_psnr(pred_image, gt))

        if not (batch_id % 10):
            print(' epoch: {0}, iteration: {1}, loss: {2}, avg loss: {3}'.format(epoch, batch_id, loss, real_loss))

    wandb.log({ "loss": real_loss })

    state = {
        "model": net.state_dict(),
        "epoch": epoch
    }

    torch.save(state, '{}/{}/latest'.format(checkpoint_path, exp_name))

    if epoch % 5 == 0:
        torch.save(state, '{}/{}/epoch_{}'.format(checkpoint_path, exp_name, epoch))


        # --- Calculate the average training PSNR in one epoch --- #
        train_psnr = sum(psnr_list) / len(psnr_list)

        # --- Use the evaluation model in testing --- #
        net.eval()

        val_psnr1, val_ssim1 = validation(net, val_data_loader1, device, exp_name)

        one_epoch_time = time.time() - start_time

        wandb.log({ "psnr": val_psnr1, "ssim": val_ssim1 })

        print_log(epoch+1, num_epochs, one_epoch_time, train_psnr, val_psnr1, val_ssim1, exp_name)
 
