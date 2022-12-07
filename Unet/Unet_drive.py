# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import Unet_utils
import Unet_model
import os
import sys
sys.path.append("../commons")
import common_utils
import CBCT_preprocess
import CBCT_postprocess

if __name__=="__main__":
    
    # Parse options
    opt = Unet_utils.Parse()
    
    # Define device
    device_name = Unet_model.tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    # Define databases
    if opt["preprocess"]:
        trDatabaseList = common_utils.getPath(database=opt["database"],
                                              folder=opt["trDatabase"])
        # tsDatabaseList = common_utils.getPath(database=opt["database"],
        #                                     folder=opt["tsDatabase"])
        # vdDatabaseList = common_utils.getPath(database=opt["database"],
        #                                     folder=opt["vdDatabase"])

        X_train, recon_X_train = CBCT_preprocess.GetImages(trDatabaseList, 
                                                           mode = 0, **opt)
        # Y_train, _ = CBCT_preprocess.GetImages(trDatabaseList, 
        #                                      mode = 1, **opt)
        # X_test, recon_X_test = CBCT_preprocess.GetImages(tsDatabaseList, 
        #                                                mode = 0, **opt)

        h5f = CBCT_preprocess.h5py.File(os.path.join(opt["database"],
                                                     "TrainData.h5"), 'w')
        h5f.create_dataset("Sinogram", data=X_train)
        h5f.create_dataset("ReconstructedSinogram", data=recon_X_train)
        h5f.close()
    else:
        h5f = CBCT_preprocess.h5py.File(os.path.join(opt["database"],
                                                     "TrainData.h5"), 'r')
        X_train = h5f["Sinogram"]
        h5f.close()
        

    # h5f = Unet_preprocess.h5py.File('TrainMaskData.h5','w')
    # h5f.create_dataset('image', data=Y_train)
    # h5f.close()

    # h5f = Unet_preprocess.h5py.File('TestData.h5','w')
    # h5f.create_dataset('image', data=X_test)
    # h5f.close()

    # Initialize Unet
    Unet = Unet_model.Unet(**opt)
    
    # Fit model
    earlystopper = common_utils.EarlyStopping(patience=15, verbose=1)
    checkpointer = common_utils.ModelCheckpoint("model_unet_checkpoint.h5", verbose=1, save_best_only=True)
    history = Unet.fit(X_train, 
                       X_train, 
                       validation_split=0.1, 
                       batch_size=1, 
                       epochs=opt["epochs"],
                       callbacks=[checkpointer])
    Unet.save("{:>s}/Unet.h5".format(opt["database"]))

    common_utils.saveHistory(filename=os.path.join("{:>s}".format(opt["database"]),"history.json"),
        history=history)

    CBCT_postprocess.plot_learning_curve(os.path.join("{:>s}".format(opt["database"]),"history.json"),"loss.png")

    # Prediction
    CBCT_postprocess.plot_train_samples(Unet,trDatabaseList,tsDatabaseList,
        X_train,Y_train,recon_X_train,opt["database"])