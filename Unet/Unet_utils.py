# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# import matplotlib.pyplot as plt
import sys
import os
import time
import glob
import argparse
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "skimage")

import shutil
import random
seed = 42
random.seed = seed

import json
import io
from fnmatch import fnmatch
from tqdm import tqdm
from sklearn.utils import shuffle
import pydicom as dicom
import rawpy
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, rescale, iradon
from skimage.transform import resize
import imageio



def Parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database',
                        default="./",
                        help='Database main folder')
    parser.add_argument('--trDatabase',
                        default="./",
                        help='Training database subfolder')
    parser.add_argument('--tsDatabase',
                        default="./",
                        help='Testing database subfolder')
    parser.add_argument('--vdDatabase',
                        default="./",
                        help='Validation database subfolder')
    parser.add_argument('--imageHeight', 
                        type=int, 
                        default=400,
                        help='Height of the input image')
    parser.add_argument('--imageWidth', 
                        type=int, 
                        default=400, 
                        help='Width of the input image')
    parser.add_argument('--imageChannels', 
                        type=int, 
                        default=1, 
                        help='Channels of the input image')
    parser.add_argument('--umin', 
                        type=int,                        
                        default=137, 
                        help='Lower grayscale integer')
    parser.add_argument('--umax',
                        type=int,
                        default=52578,
                        help='Upper grayscale integer')
    parser.add_argument('--epochs', 
                        type=int, 
                        default=100, 
                        help='Number of training epochs')
    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0001, 
                        help='AE learning rate, default=0.0001')
    parser.add_argument('--cuda', 
                        action='store_true', 
                        help='enables cuda')
    parser.add_argument('--gpu', 
                        type=int, 
                        default=2, 
                        help='number of GPUs to use')
    parser.add_argument('--plot',
                        action='store_true',
                        default=False,
                        help='plot figures')
    # parser.add_argument('--wtdof',nargs='+',default=[3],help='Specify the connection between wdof and tdof (mdof database only)')
    # parser.set_defaults(stack=False,ftune=False,feat=False,plot=True)
    opt = parser.parse_args().__dict__
    return opt

# Get train and test IDs
def allDivs(n):
    div = []
    for i in range(1, n):
        if((n % i) == 0):
            div.append(i)
    return div

def getPath(database="./",folder="./"):
    search_folder = os.path.join("{:>s}".format(database),
        "{:>s}".format(folder))
    filename_list = [os.path.join("{:>s}".format(search_folder),"{:>s}".format(f)) for f in os.listdir(search_folder) if fnmatch(f, '*.raw')]

    filename_list = sorted(filename_list)
    return filename_list

def saveHistory(filename,history):
    out_file = open(filename,"w")
    json.dump(history.history, out_file, indent = 6)
    out_file.close()
    return
