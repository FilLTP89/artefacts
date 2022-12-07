import sys
# sinogram completion CGAN
sys.path.insert(0, './CGAN')
from model import cgan
from main import main
from ops import *

import argparse
import os
import scipy.misc
import numpy as np
import os
import tensorflow as tf
import time
from glob import glob
import numpy as np
from six.moves import xrange
import scipy.io
from scipy import sparse
import math as m
# astra image recon related importsimport time
import scipy.sparse.linalg
import scipy.io
from glob import glob
import pylab
from sklearn.feature_extraction import image
import numbers
from scipy import sparse
from numpy.lib.stride_tricks import as_strided
from itertools import product
from decimal import Decimal
from sklearn.metrics import mean_squared_error as immse
import matplotlib.pyplot as plt

# new imports
import sys
import matplotlib.pyplot as plt
from numpy import linspace, pi, sin
import tensorflow as tf
import numpy as np
# from skimage.io import imread
# from skimage import data_dir
# from skimage.morphology import disk, erosion, dilation
from scipy.interpolate import griddata
from scipy import interpolate
import h5py
from scipy import sparse
import tables, warnings

print(tf.__version__)

# sinogram completion related variables
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='../data/checkpoint_sim', help='models are saved here')
parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=10, help='weight on L1 term in objective')
parser.add_argument('--example_num', dest='n', default='2', help='example number')

# projection data and reconstruction related parameters
thetas = np.linspace(0,180.0,720,False)
num_det = 1024
num_theta = 720
thresh = 0.8101 #this value is for per cm recons, otherwise use 4000 if recons are in MHU scale.
mu_w = 0.202527
sinogram_scaling = 4.2296
npix = 512.
dpix = 475/512.
nbins = 1024.

def load_sparse_matrix(fname):
    warnings.simplefilter("ignore", UserWarning)
    f = tables.open_file(fname)
    M = sparse.csc_matrix( (f.root.At.data[...], f.root.At.ir[...], f.root.At.jc[...]) )
    f.close()
    return M

## Sinogram completion using trained CGAN
def completeSinogram(model,incomplete_sino):
    t_c = time.time()
    complete_sino = model.sess.run(model.completed_sino,feed_dict={model.incomplete_sino:incomplete_sino})
    t_c = time.time() - t_c
    print('Sinogram completion time: %4.4f seconds.' % t_c)
    return complete_sino

def fbp_mat(sino, At, filt):
    t_start = time.time()
    sino_freq = np.fft.fft(sino,axis=0)
    filtered_freq = ramlak * sino_freq
    filtered_sino = np.fft.ifft(filtered_freq,axis=0)
    filtered_sino = np.real(filtered_sino[0:1024,:])
    filtered_sino = np.reshape(filtered_sino.flatten('F'),(At.shape[1],1))

    rec = At * filtered_sino
    rec = np.reshape(rec * np.pi/(2*720),(512, 512),'F')
    rec = rec * 10/dpix #  conversion to per cm
    # rec = 1000 + 1000 * (rec - mu_w)/mu_w # if want to convert to Modified Hounsfield Units (MHU)
    line = 'FBP Reconstruction time: %4.4f seconds.' % (time.time() - t_start)
    print(line)
    return rec

def imshow(ax, im, title=None,imkwargs=None):
    if title != None:
        ax.imshow(im, cmap=plt.cm.Greys_r, **imkwargs)
        ax.set_title(title)
    else:
        ax.imshow(im, cmap=plt.cm.Greys_r)
    ax.axis('off')

## sinogram completion model
args = parser.parse_args()
g = tf.Graph()
model = main(args,g)
print(model.batch_size)
if model.load(model.checkpoint_dir):
    print(' [*] Sinogram Completion Model Load SUCCESS')
    temp = model.sess.run(model.completed_sino,feed_dict={model.incomplete_sino:np.zeros((1,1024, 768,1))}) # just starting it off! 
else:
    print(' [!] Sinogram Completion Model Load failed, please set checkpoint_dir correctly!')
    sys.exit(2)

ramlak = np.squeeze(scipy.io.loadmat('../data/Ramlak.mat')['myFilter']) # loading Ram-Lak filter
ramlak = ramlak.astype(complex)
print('Loading A matrix.')
t_a = time.time()
At = load_sparse_matrix('../data/At.mat')
print('A matrix loading time: %4.4f' % (time.time() - t_a))

# Reading Projection Data
m1_m0 = scipy.io.loadmat('../data/sino130_n_'+args.n+'_m_m0.mat')['m1_m0'] # sinogram with metals
input_sino = scipy.io.loadmat('../data/sino130_n_'+args.n+'_m.mat')['sg_ln_noise_130'] # reference metal-free sinogram
li_sino = scipy.io.loadmat('../data/sino130_n_'+args.n+'_m_li.mat')['li_sino'] #
wnn_sino = scipy.io.loadmat('../data/sino130_n_'+args.n+'_m_wnn.mat')['wnn_sino'] #
phantom = scipy.io.loadmat('../data/ph_n_'+args.n+'_m.mat')['rec_gt'] #

gt_sino = m1_m0[:,0:num_theta]
incomplete_sino = m1_m0[:,2*num_theta:3*num_theta]
mask = m1_m0[:,num_theta:2*num_theta]

metals = phantom > thresh

uncorrected_rec = fbp_mat(input_sino, At, ramlak)
gt_rec = fbp_mat(gt_sino/sinogram_scaling, At, ramlak)
li_rec = fbp_mat(li_sino/sinogram_scaling, At, ramlak)
wnn_rec = fbp_mat(wnn_sino/sinogram_scaling, At, ramlak)

# ## Step 1: Uncorrected FBP Reconstruction and Metal Segmentation
# uncorrected_rec = fbp(input_sino,thetas,num_det,mu_w)
# metals = uncorrected_rec > thresh # thresholding
# eroded = erosion(metals, disk(2)) # eroding
# dilated = dilation(eroded, disk(4)) # dilating
# metals = dilated > 0
# metals_sino = radon(metals, theta=thetas, circle=True) # forward projecting metal objects
# mask = (metals_sino > 0).astype(np.double) # metal mask in the projection domain for data deletion

## Step 2: Sinogram completion
# incomplete_sino = (1 - mask) * input_sino # metal related data deletion in the projection domain
incomplete_sino_padded = np.pad(incomplete_sino, pad_width=((0,0),(0,48)), mode='constant') #zero padding to match network input dimensions 1024x768
incomplete_sino_padded = np.expand_dims(np.expand_dims(incomplete_sino_padded,axis=0),axis=3)
completed_sino = completeSinogram(model,incomplete_sino_padded) # completing sinogram using CNN
completed_sino = np.squeeze(completed_sino[:,:,0:720,:])
completed_sino = incomplete_sino + mask * completed_sino # Mask specific sinogram completion
## Step 3: FBP Reconsruction using completed data and metal insertion
deep_mar_rec = fbp_mat(completed_sino/sinogram_scaling, At, ramlak)
deep_mar_rec_m = (1-metals) * deep_mar_rec + metals * uncorrected_rec #inserting metals in "corrected" reconstruction
gt_rec_m = (1-metals) * gt_rec + metals * uncorrected_rec #inserting metals in "reference ground truth" reconstruction
li_rec_m = (1-metals) * li_rec + metals * uncorrected_rec #inserting metals in "linearly interpolated (LI) data" reconstruction
wnn_rec_m = (1-metals) * wnn_rec + metals * uncorrected_rec #inserting metals in "weighted nearest neighbors (WNN) interpolated data" reconstruction

# Show and save reconstruction results.
fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5)
fig.suptitle('Reconstruction Results', x=0.5, y=0.8,fontsize=14)
imkwargs = dict(vmin=0.0, vmax=0.15)
imshow(ax1, uncorrected_rec, title="Uncorrected",imkwargs=imkwargs)
imshow(ax2, li_rec_m, title="LI-MAR",imkwargs=imkwargs)
imshow(ax3, wnn_rec_m, title="WNN-MAR",imkwargs=imkwargs)
imshow(ax4, deep_mar_rec_m, title="Deep-MAR",imkwargs=imkwargs)
imshow(ax5, gt_rec_m, title="Reference",imkwargs=imkwargs)
plt.savefig('../results/rec_n_'+args.n+'.png', dpi=300.) #
plt.show()
# Show and save sinogram results.
fig, (ax1, ax2,ax3,ax4,ax5) = plt.subplots(1, 5)
fig.suptitle('Sinogram Results', x=0.5, y=0.8,fontsize=14)
imkwargs = dict(vmin=0.0, vmax=5)
imshow(ax1, input_sino, title="Uncorrected",imkwargs=imkwargs)
imshow(ax2, li_sino/sinogram_scaling, title="LI-MAR",imkwargs=imkwargs)
imshow(ax3, wnn_sino/sinogram_scaling, title="WNN-MAR",imkwargs=imkwargs)
imshow(ax4, completed_sino/sinogram_scaling, title="Deep-MAR",imkwargs=imkwargs)
imshow(ax5, gt_sino/sinogram_scaling, title="Reference",imkwargs=imkwargs)
plt.savefig('../results/sino_n_'+args.n+'.png', dpi=300.) #
plt.show()
sys.exit(1)
