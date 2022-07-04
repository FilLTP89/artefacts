# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import h5py
from PIL import *
from skimage.transform import radon, iradon, resize
seed = 42
np.random.seed = seed

def createSin(image, flag=1, name=None):
    # print("Creating the sinogram")
    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)

    if name :
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

# 0 = Train
# 1 = Test
# Otherwise this function will return -1 as an error
def getImgs(filename_list, mode = 0, imageHeight=400, imageWidth=400,imageChannels=1,**kwargs):
    i = 0
    if(mode == 0):
        image_list = np.zeros((len(filename_list),imageHeight,imageWidth),
                              dtype=np.uint16)    
    if(mode == 1):
        image_list = np.zeros((len(filename_list),imageHeight,imageWidth,imageChannels),
                              dtype=np.uint16)
    elif(mode != 0 and mode != 1):
        return -1
        
    total = len(filename_list)
    j = 1
    image_recon = np.zeros((len(filename_list), imageHeight, imageWidth), dtype = np.uint16)

    for x in filename_list:

          scene_infile = open(x,'rb')
          scene_image_array = np.fromfile(scene_infile,dtype=np.uint16,
                                          count=imageHeight*imageWidth)
          rgb = Image.frombuffer("I",[imageWidth,imageHeight],
                                 scene_image_array.astype('I'),
                                'raw','I',0,1)
          # import pdb
          # pdb.set_trace()
          # plt.imshow(rgb)
          rgb=np.array(rgb)
          # rgb = np.fromfile(x, dtype=np.uint64, sep="") # Preciso verificar também aqui os diferentes valores de np.uintX -> melhor : 64|| float.32 também é uma boa
          # rgb = rgb/np.mean(rgb)
          # print("Dimension of the old image array: ", rgb.ndim)
          # print("Size of the old image array: ", rgb.size)
          rgb.resize((imageHeight, imageWidth)) # Esse comando me salvou aqui-> preciso mudar algumas características do sinograma para verificar se funciona aqui
          # print("----------------------------------------------------")
          # print("The shape of the original image array is: ", rgb.shape)
          # print("----------------------------------------------------")
          # print("New dimension of the array:", rgb.ndim)
          # print(rgb.shape)
          sino, theta = createSin(rgb) #,name="{:>s}png".format(x.strip('raw')))
          image_recon[i] = reconstructSin(sino, theta)

          if(mode == 1):
              sino = np.expand_dims(resize(sino, 
                                           (imageHeight, imageWidth), 
                                           mode = "constant", preserve_range=True), 
                                    axis = -1)
              # print("Sino.shape : {}".format(sino.shape))

          image_list[i] = sino[:, :]
          j += 1
          i += 1
          
    return image_list, image_recon

def reconstructSin(sinogram, theta):
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name='ramp')
    return reconstruction_fbp