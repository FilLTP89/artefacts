# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import matplotlib.pyplot as plt
import json
from Unet_preprocess import reconstructSin 
from os.path import join as opj
import numpy as np

def plot_learning_curve(historyfile,filename):
    fid = open(historyfile,)
    history = json.load(fid)
    plt.figure(figsize = (12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    plt.title("model accuracy", fontsize = 20)
    plt.ylabel("accuracy [%]", fontsize = 16)
    plt.xlabel("epoch", fontsize = 16)
    plt.legend(["train", "test"], loc = "upper left")
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history["loss"])
    plt.plot(history["val_loss"])
    plt.title("Model loss", fontsize = 20)
    plt.ylabel("Loss [1]", fontsize = 16)
    plt.xlabel("epoch", fontsize = 16)
    plt.legend(["train", "test"], loc = "upper left")
    plt.savefig(filename,bbox_inches="tight")
    plt.close()

def plot_train_samples(Unet,trDatabaseList,tsDatabaseList,X_train,Y_train,
    recon_X_train,database):
    trainLen = len(X_train)
    Xhat_train = Unet.predict(X_train[:trainLen - 2], verbose = 1)
    #Xhat_test = Unet.predict(X_train[trainLen - 2:], verbose = 1)
    #Xhat_val = Unet.predict(X_test, verbose = 1)

    Xhat_train_t = (Xhat_train > 0.25).astype(np.uint8)
    #Xhat_test_t = (Xhat_test > 0.25).astype(np.uint8)
    #Xhat_val_t = (Xhat_val > 0.25).astype(np.uint8)
    lastIndex = 0
    tam_ = len(Xhat_train)

    for i in range(0, len(Xhat_train)):

        plt.figure(figsize = (12, 12))

        plt.subplot(1, 3, 1)
        plt.title("Training entry {}".format(trDatabaseList[i].split("/")[-1]))
        plt.axis("off")
        plt.imshow(X_train[i], cmap = plt.cm.gray)
    
        plt.subplot(1, 3, 2)
        plt.title("Training response {}".format(trDatabaseList[i].split("/")[-1]))
        plt.axis("off")
        plt.imshow(Y_train[i][:,:,0], cmap = plt.cm.gray)

        plt.subplot(1, 3, 3)
        plt.title("Treated result {}".format(trDatabaseList[i].split("/")[-1]))
        plt.axis("off")
        plt.imshow(Xhat_train[i][:,:,0], cmap = plt.cm.gray)

        plt.savefig(opj(database,"train_{}.png".format(trDatabaseList[i].split("/")[-1])))
        plt.close()

        plt.figure(figsize = (12, 12))

        plt.subplot(1, 3, 1)
        plt.title("Reconstructed data {}".format(trDatabaseList[i].split("/")[-1]))
        plt.axis("off")
        plt.imshow(recon_X_train[i], cmap = plt.cm.gray)

        plt.subplot(1, 3, 2)
        plt.title("Reconstructed output data {}".format(trDatabaseList[i].split("/")[-1]))
        plt.axis("off")
        plt.imshow(reconstructSin(Y_train[i][:,:,0], theta = np.linspace(0., 180., max(Y_train[i][:,:, 0].shape), endpoint=False)), cmap = plt.cm.gray)

        plt.subplot(1, 3, 3)
        plt.title("Recon modelo dado {}".format(trDatabaseList[i].split("/")[-1]))
        plt.axis("off")
        plt.imshow(reconstructSin(Xhat_train[i][:,:,0], theta = np.linspace(0., 180., max(Xhat_train[i][:, :, 0].shape), endpoint=False)), cmap = plt.cm.gray)

        
        plt.savefig(opj(database,"train_recon_{}.png".format(trDatabaseList[i].split("/")[-1])))
        plt.close()
        
        lastIndex = i
