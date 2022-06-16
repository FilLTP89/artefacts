import utils
import preprocess
import model
# import tensorflow as tf


if __name__=="__main__":
    
    opt = utils.Parse()

    device_name = model.tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
        print('Found GPU at: {}'.format(device_name))

    Unet = model.Unet(**opt)

    trDatabaseList = utils.getPath(database=opt["database"],folder=opt["trDatabase"])
    tsDatabaseList = utils.getPath(database=opt["database"],folder=opt["tsDatabase"])
    vdDatabaseList = utils.getPath(database=opt["database"],folder=opt["vdDatabase"])

    X_train, recon_X_train = preprocess.getImgs(trDatabaseList, mode = 0, **opt)
    Y_train, _ = preprocess.getImgs(tsDatabaseList, mode = 1, **opt)
    X_test, recon_X_test = preprocess.getImgs(vdDatabaseList, mode = 0, **opt)


    h5f = preprocess.h5py.File('TrainData.h5','w')
    h5f.create_dataset('image', data=X_train)
    h5f.close()

    h5f = preprocess.h5py.File('TrainMaskData.h5','w')
    h5f.create_dataset('image', data=Y_train)
    h5f.close()

    h5f = preprocess.h5py.File('TestData.h5','w')
    h5f.create_dataset('image', data=X_test)
    h5f.close()

    # Fit model
    earlystopper = model.EarlyStopping(patience=15, verbose=1)
    checkpointer = model.ModelCheckpoint("model_unet_checkpoint.h5", verbose=1, save_best_only=True)
    history = Unet.fit(X_train, Y_train, validation_split=0.1, 
                       batch_size=1, epochs=opt["epochs"],
                       callbacks=[checkpointer])
    Unet.save("{:>s}/Unet.h5".format(opt["database"]))

    out_file = open("{:>s}/history.json".format(opt["database"]),"w")
    utils.json.dump(history, out_file, indent = 6)
    out_file.close()


