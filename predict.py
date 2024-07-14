import numpy as np
import os
import torch
from torch import nn, tensor, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torch.optim.lr_scheduler as lr_scheduler
import json
from PIL import Image
import argparse

import project_utils

def get_input_args():
    parser = argparse.ArgumentParser(description="Predict.py")
    parser.add_argument('input_img', default="flowers/test/1/image_06752.jpg", help = "path of train dataset", metavar="FILE")
    parser.add_argument('--gpu', action="store_true", default="gpu", help = "Use GPU or CPU to train model")
    parser.add_argument('checkpoint', action="store", default="checkpoint.pth", help = "accessing saved checkpoint")
    parser.add_argument('--top_k', default=5, dest="top_k", type=int)
    parser.add_argument('--cat_names', dest="cat_names", default='cat_to_name.json')

    return parser.parse_args()

in_arg = get_input_args()

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

input_img = in_arg.input_img
gpu = in_arg.gpu
checkpoint = in_arg.checkpoint
topk = in_arg.top_k
cat_names = in_arg.cat_names

device = project_utils.check_gpu(gpu)
model = project_utils.load_checkpoint(checkpoint)
prob, classes = project_utils.predict(input_img, device, model, topk)
project_utils.show_pred(prob, classes, cat_to_name)