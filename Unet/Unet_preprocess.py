# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import radon, iradon, resize
seed = 42
np.random.seed = seed

def CreateSinogram(image, 
                   flag=1,
                   plot=False,
                   **kwargs):
    # print("Creating the sinogram")
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)

    return sinogram, theta



def ReconstructSinogram(sinogram, theta):
    reconstruction_fbp = iradon(sinogram, 
                                theta=theta, 
                                filter_name='ramp')
    return reconstruction_fbp


def Raw2Sinogram(rawFilename,
                 imageWidth,
                 imageHeight,
                 mode,
                 umin = 137, 
                 umax = 52578,
                 plot = False,
                 **kwargs):

    rawInFile = open(rawFilename, 'rb')
    rawImageArray = np.fromfile(rawInFile,
                                dtype=np.uint16,
                                count=imageHeight*imageWidth)
    rawRGBimage = Image.frombuffer("I;16",
                                   [imageWidth, imageHeight],
                                   rawImageArray,  # .astype('I;16'),
                                   "raw",
                                   "I;16",
                                   0,
                                   1)
    rawRGBarray = np.array(rawRGBimage)

    rawRGBarray.resize((imageHeight, imageWidth))

    rawRGBarray -= umin
    rawRGBarray = rawRGBarray/(umax-umin)

    # if plot:
        # fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
        # ax1.set_title("Raw image")
        # ax1.imshow(rawRGBarray, cmap="gray")
        # fig.tight_layout()
        # plt.savefig("{:s}.png".format(rawFilename.strip(".raw")))
        # plt.close()

    sinogram, theta = CreateSinogram(rawRGBarray)
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        ax1.set_title("Original")
        ax1.imshow(rawRGBarray, 
                   cmap=plt.cm.Greys_r)
        dx = 0.5 * 180.0 / max(rawRGBarray.shape)
        dy = 0.5 / sinogram.shape[0]
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        # print("Ploting the sinogram")
        # ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
        #           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
        #           aspect='auto')
        ax2.imshow(sinogram, 
                   cmap=plt.cm.Greys_r,
                   extent=(0, 
                           180, 
                           -sinogram.shape[0]/2.0, 
                           sinogram.shape[0]/2.0), 
                   aspect='auto')
        fig.tight_layout()
        plt.savefig("{:s}_sin.png".format(rawFilename.strip(".raw")))
        plt.close()
        
    if(mode == 1):
        sinogram = np.expand_dims(resize(sinogram,
                                         (imageHeight, imageWidth),
                                         mode="constant",
                                         preserve_range=True),
                                  axis=-1)

    return sinogram, theta

def GetImages(fileNameList,
            mode = 0,
            imageHeight=400, 
            imageWidth=400,
            imageChannels=1,
            **kwargs):
    
    if(mode == 0):
        SinogramList = np.zeros((len(fileNameList),
                                imageHeight,
                                imageWidth),
                                dtype=np.uint16)    
    if(mode == 1):
        SinogramList = np.zeros((len(fileNameList),
                                imageHeight,
                                imageWidth,
                                imageChannels),
                                dtype=np.uint16)
    elif(mode != 0 and mode != 1):
        return -1

    ReconstructedSinogramArray = np.zeros((len(fileNameList), 
                                           imageHeight, 
                                           imageWidth),
                                          dtype = np.uint16)

    SinogramThetaList = [Raw2Sinogram(x, 
                                      imageWidth,
                                      imageHeight,
                                      mode,
                                      **kwargs) for x in fileNameList]
    SinogramArray = np.array([x[0][:, :]
                             for x in SinogramThetaList])
    ReconstructedSinogramArray = np.array([ReconstructSinogram(x) 
                                           for x in SinogramThetaList])
    return SinogramArray, ReconstructedSinogramArray
