# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import Unet_utils
import Unet_model
import Unet_preprocess
import Unet_postprocess
# import tensorflow as tf


if __name__=="__main__":
    
    opt = Unet_utils.Parse()

    device_name = Unet_model.tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    Unet = Unet_model.Unet(**opt)

    trDatabaseList = Unet_utils.getPath(database=opt["database"],folder=opt["trDatabase"])
    tsDatabaseList = Unet_utils.getPath(database=opt["database"],folder=opt["tsDatabase"])
    vdDatabaseList = Unet_utils.getPath(database=opt["database"],folder=opt["vdDatabase"])

    X_train, recon_X_train = Unet_preprocess.getImgs(trDatabaseList, mode = 0, **opt)
    Y_train, _ = Unet_preprocess.getImgs(trDatabaseList, mode = 1, **opt)
    X_test, recon_X_test = Unet_preprocess.getImgs(tsDatabaseList, mode = 0, **opt)


    h5f = Unet_preprocess.h5py.File('TrainData.h5','w')
    h5f.create_dataset('image', data=X_train)
    h5f.close()

    h5f = Unet_preprocess.h5py.File('TrainMaskData.h5','w')
    h5f.create_dataset('image', data=Y_train)
    h5f.close()

    h5f = Unet_preprocess.h5py.File('TestData.h5','w')
    h5f.create_dataset('image', data=X_test)
    h5f.close()

    # Fit model
    earlystopper = Unet_model.EarlyStopping(patience=15, verbose=1)
    checkpointer = Unet_model.ModelCheckpoint("model_unet_checkpoint.h5", verbose=1, save_best_only=True)
    history = Unet.fit(X_train, Y_train, validation_split=0.1, 
                       batch_size=1, epochs=opt["epochs"],
                       callbacks=[checkpointer])
    Unet.save("{:>s}/Unet.h5".format(opt["database"]))

    Unet_utils.saveHistory(filename=os.path.join("{:>s}".format(opt["database"]),"history.json"),
        history=history)

    Unet_postprocess.plot_learning_curve(os.path.join("{:>s}".format(opt["database"]),"history.json"),"loss.png")

    # Prediction
    Unet_postprocess.plot_train_samples(Unet,trDatabaseList,tsDatabaseList,
        X_train,Y_train,recon_X_train,opt["database"])
