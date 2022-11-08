# -*- coding: utf-8 -*-
#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import radon, iradon, resize
seed = 42
np.random.seed = seed

def createSin(image, flag=1, name=None):
    # print("Creating the sinogram")
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)

    if name:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))
        ax1.set_title("Original")
        ax1.imshow(image, cmap=plt.cm.Greys_r)
        dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
        ax2.set_title("Radon transform\n(Sinogram)")
        ax2.set_xlabel("Projection angle (deg)")
        ax2.set_ylabel("Projection position (pixels)")
        # print("Ploting the sinogram")
        # ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
        #           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
        #           aspect='auto')
        ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, -sinogram.shape[0]/2.0, sinogram.shape[0]/2.0), aspect='auto')
        fig.tight_layout()
        plt.savefig(name)
        plt.close()

    return sinogram, theta


def Raw2Sinogram(rawFilename, 
                 imageWidth, 
                 imageHeight, 
                 mode):
    # raw = rawpy.imread(rawFilename)
    # rgb = raw.postprocess()
    # imageio.imsave('default.tiff', rgb)
    
    rawInFile = open(rawFilename, 'rb')
    rawImageArray = np.fromfile(rawInFile,
                                dtype=np.uint16,
                                count=imageHeight*imageWidth)
    rawRGBimage = Image.frombuffer("I;16",
                                   [imageWidth, imageHeight],
                                   rawImageArray,#.astype('I;16'),
                                   "raw",
                                   "I;16",
                                   0,
                                   1)
    rawRGBarray = np.array(rawRGBimage)

    rawRGBarray.resize((imageHeight, imageWidth))
    
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1.set_title("Original")
    ax1.imshow((rawRGBarray-137)/(52578-137),
                   cmap="gray")
    # dx, dy = 0.5 * 180.0 / max(image.shape), 0.5 / sinogram.shape[0]
    # ax2.set_title("Radon transform\n(Sinogram)")
    # ax2.set_xlabel("Projection angle (deg)")
    # ax2.set_ylabel("Projection position (pixels)")
    # print("Ploting the sinogram")
    # ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
    #           extent=(-dx, 180.0 + dx, -dy, sinogram.shape[0] + dy),
    #           aspect='auto')
    # ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
    #            extent=(0, 180, -sinogram.shape[0]/2.0, sinogram.shape[0]/2.0), aspect='auto')
    fig.tight_layout()
    plt.savefig("{:s}.png".format(rawFilename.strip(".raw")))
    plt.close()
    
    
    
    sinogram, theta = createSin(rawRGBarray)
    if(mode==1):
        sinogram = np.expand_dims(resize(sinogram,
                                            (imageHeight, imageWidth),
                                            mode="constant", 
                                            preserve_range=True),
                                    axis=-1)
    
    return sinogram, theta


def reconstructSin(sinogram, theta):
    reconstruction_fbp = iradon(sinogram, 
                                theta=theta, 
                                filter_name='ramp')
    return reconstruction_fbp

def getImgs(fileNameList,
            mode = 0,
            imageHeight=400, 
            imageWidth=400,
            imageChannels=1,**kwargs):
    
    if(mode == 0):
        sinogramList = np.zeros((len(fileNameList),
                                imageHeight,
                                imageWidth),
                                dtype=np.uint16)    
    if(mode == 1):
        sinogramList = np.zeros((len(fileNameList),
                                imageHeight,
                                imageWidth,
                                imageChannels),
                                dtype=np.uint16)
    elif(mode != 0 and mode != 1):
        return -1

    recSinogramList = np.zeros((len(fileNameList), 
                                imageHeight, 
                                imageWidth), 
                                dtype = np.uint16)

    sinogramList = np.array([Raw2Sinogram(x, 
                               imageWidth, 
                               imageHeight, 
                               mode)[0][:,:] for x in fileNameList])
    recSinogramList = np.array([reconstructSin(*Raw2Sinogram(x,
                                                             imageWidth, 
                                                             imageHeight, 
                                                             mode)) for x in fileNameList])
    return sinogramList, recSinogramList