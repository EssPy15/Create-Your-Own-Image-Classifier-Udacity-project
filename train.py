import numpy as np
import os
import torch
from torch import nn, tensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torch.optim.lr_scheduler as lr_scheduler
import json
import argparse

import project_utils

# takes input about architecture of model, hyperparameters, and GPU/CPU, learning_parameters

def get_input_args():
    parser = argparse.ArgumentParser(description="Train.py")
    parser.add_argument('data_dir', default="flowers", help = "path of train dataset")
    parser.add_argument('--arch', dest="arch", default="densenet121", type = str, help = "choose model to train the dataset")
    parser.add_argument('--gpu', dest="gpu", action="store_true", default="gpu", help = "Use GPU or CPU to train model")
    parser.add_argument('--checkpoint_dir', dest="checkpoint_dir", action="store", default="./checkpoint.pth", help = "saving the model")
    parser.add_argument('--lr', dest="lr", type=float, default=0.001, help = "learning rate")
    parser.add_argument('--dropout', dest = "dropout", type=float, default = 0.3, help = "set the dropout probability")
    parser.add_argument('--epochs', dest="epochs", type=int, default=1, help = "set number of epochs")
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", default=512)
    parser.add_argument('--cat_names', dest="cat_names", default='cat_to_name.json')

    return parser.parse_args()

in_arg = get_input_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
len = len(cat_to_name)

data_dir = in_arg.data_dir
arch = in_arg.arch
gpu = in_arg.gpu
checkpoint = in_arg.checkpoint_dir
lr = in_arg.lr
dropout = in_arg.dropout
epochs = in_arg.epochs
hidden_units = in_arg.hidden_units

train_data, valid_data, test_data = project_utils.transform_data(data_dir)
trainloader, validloader, testloader = project_utils.load_data(train_data, valid_data, test_data)
device = project_utils.check_gpu(gpu)
model, criterion, optimizer  = project_utils.set_up_model(arch, dropout, lr, hidden_units, len)
project_utils.train_model(model, trainloader, validloader, device, criterion, optimizer, epochs)
# project_utils.test_model(model, testloader, device, criterion)
project_utils.save_checkpoint(model, arch, hidden_units, dropout, train_data, len, checkpoint)