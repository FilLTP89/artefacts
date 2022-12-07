# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import sys
sys.path.append("../commons")
sys.path.append("./CGAN")

import CBCT_preprocess
import CBCT_postprocess
from DeepMAR_utils import *
from common_utils import *

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
from tensorflow import keras
tf.keras.backend.set_floatx('float32')
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


from DeepMAR_model import DeepMAR
import DeepMAR_losses

def Train(options):

    with tf.device(options["DeviceName"]):

        losses = DeepMAR_losses.getLosses(**options)
        optimizers = DeepMAR_losses.getOptimizers(**options)
        callbacks = DeepMAR_losses.getCallbacks(**options)

        # Instantiate the RepGAN model.
        DMR = DeepMAR(options)

        # Compile the RepGAN model.
        DMR.compile(optimizers, losses, metrics=[tf.keras.metrics.Accuracy()])

        # Build shapes
        # DMR.build(input_shape=(options['batchSize'],options['Xsize'],options['nXchannels']))

        # Build output shapes
        DMR.compute_output_shape(input_shape=(options['batchSize'], 
                                              options['Xsize'],
                                              options['nXchannels']))

        # Define databases
        if options["preprocess"]:
            trDatabaseList = common_utils.getPath(database=options["database"],
                                                  folder=options["trDatabase"])
            # tsDatabaseList = common_utils.getPath(database=opt["database"],
            #                                     folder=opt["tsDatabase"])
            # vdDatabaseList = common_utils.getPath(database=opt["database"],
            #                                     folder=opt["vdDatabase"])
            train_dataset, recon_X_train = CBCT_preprocess.GetImages(trDatabaseList,
                                                                     mode=0, 
                                                                     **options)
            # Y_train, _ = CBCT_preprocess.GetImages(trDatabaseList,
            #                                      mode = 1, **opt)
            # X_test, recon_X_test = CBCT_preprocess.GetImages(tsDatabaseList,
            #                                                mode = 0, **opt)
            h5f = CBCT_preprocess.h5py.File(os.path.join(options["database"],
                                                         "TrainData.h5"), 'w')
            h5f.create_dataset("Sinogram", data=train_dataset)
            h5f.create_dataset("ReconstructedSinogram", data=recon_X_train)
            h5f.close()
            # h5f = Unet_preprocess.h5py.File('TrainMaskData.h5','w')
            # h5f.create_dataset('image', data=Y_train)
            # h5f.close()
            # h5f = Unet_preprocess.h5py.File('TestData.h5','w')
            # h5f.create_dataset('image', data=X_test)
            # h5f.close()
        else:
            h5f = CBCT_preprocess.h5py.File(os.path.join(options["database"],
                                                         "TrainData.h5"), 'r')
            train_dataset = h5f["Sinogram"]
            h5f.close()

        # Train RepGAN
        history = DMR.fit(x=train_dataset, batch_size=options['batchSize'],
                          epochs=options["epochs"],
                          callbacks=callbacks,
                          validation_data=val_dataset, 
                          shuffle=True, 
                          validation_freq=100)

        DumpModels(DMR, options['results_dir'])


def Evaluate(options):
    return None

if __name__ == '__main__':
    options = ParseOptions()

    if options["trainVeval"].upper() == "TRAIN":
        Train(options)
    elif options["trainVeval"].upper() == "EVAL":
        Evaluate(options)
        
# if __name__=="__main__":
    
#     # Parse options
#     options = DeepMAR_utils.Parse()
    
#     # Define device
#     device_name = DeepMAR_model.tf.test.gpu_device_name()
#     if device_name != '/device:GPU:0':
#         raise SystemError('GPU device not found')
#         print('Found GPU at: {}'.format(device_name))

       

#     # Initialize Unet
#     CGAN = DeepMAR_model.CGAN_model(**opt)
    
#     # Fit model
#     earlystopper = common_utils.EarlyStopping(patience=15, verbose=1)
#     checkpointer = common_utils.ModelCheckpoint("model_unet_checkpoint.h5", verbose=1, save_best_only=True)
#     history = Unet.fit(X_train, 
#                        X_train, 
#                        validation_split=0.1, 
#                        batch_size=1, 
#                        epochs=opt["epochs"],
#                        callbacks=[checkpointer])
#     Unet.save("{:>s}/Unet.h5".format(opt["database"]))

#     common_utils.saveHistory(filename=os.path.join("{:>s}".format(opt["database"]),"history.json"),
#         history=history)

#     CBCT_postprocess.plot_learning_curve(os.path.join("{:>s}".format(opt["database"]),"history.json"),"loss.png")

#     # Prediction
#     CBCT_postprocess.plot_train_samples(Unet,trDatabaseList,tsDatabaseList,
#         X_train,Y_train,recon_X_train,opt["database"])