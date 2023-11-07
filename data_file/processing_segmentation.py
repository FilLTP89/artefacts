import tensorflow as tf
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from visualize import visualize_from_dataset
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from segmentation_models import get_preprocessing

"""
Classic segmentation Dataset 
"""


"""
TODO:
    - Verify sample, it seems that some sample are incorrect
    - Verify how to use it, x and y seems to be way different, can we 
    really create substract the metal using (x - y_pred)? 
        - Create x-y and see how it goes
    """

# Try to save to use the save function to save the dataset in folder -> Actually took too much times
# See how much volume they take, and how long we take to load them


class SegmentationDataset:
    def __init__(
        self,
        path: str = "./data/segmentation/",
        width: int = 512,
        height: int = 512,
        batch_size: int = 32,
        saving_format: str = None,
        train_saving_path: str = "train/",
        test_saving_path: str = "test/",
        valid_saving_path: str = "valid/",
        seed: int = 2,  # 1612
        shuffle=True,
        backbone = None
    ) -> None:

        self.path = path
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.saving_format = saving_format
        self.train_saving_path = path + train_saving_path
        self.test_saving_path = path + test_saving_path
        self.valid_saving_path = path + valid_saving_path

        self.original_width = 400
        self.original_height = 400

        self.seed = seed
        self.shuffle = shuffle

        self.processor = get_preprocessing(backbone) if backbone else None

        
    def collect_data(self):

        images_path = f"{self.path}/images/"
        masks_path = f"{self.path}/masks/"
        
        images_folder = [
            sorted(
                glob(os.path.join(images_path, str(i) + "/*.tif")),
            )
            for i in range(1,6)
        ]
        mask_folder = [
            sorted(
                glob(os.path.join(masks_path, str(i) + "/*.tif")),
            )
            for i in range(1,6)
        ]

        images_list = [item for sublist in images_folder for item in sublist]
        masks_list = [item for sublist in mask_folder for item in sublist]

        return (images_list, masks_list)
        

    def load_data(self):
        """
        Generate the training and testing (X,y) couple using the previously generated list
        """
        images_list,masks_list = self.collect_data()

        X_train, X_test_1, y_train, y_test_1 = train_test_split(
            images_list,
            masks_list,
            test_size=(2 / 11),
            random_state=self.seed,
            shuffle=self.shuffle,
        )  # 8 acquisition for training
        X_test, X_valid, y_test, y_valid = train_test_split(
            X_test_1,
            y_test_1,
            test_size=0.5,
            random_state=self.seed,
            shuffle=self.shuffle,
        )  # 1 acquisition for validation & 1 acquisition for testing

        # X_train, y_train = shuffle(X_train, y_train, random_state=self.seed)
        # Train : acquisition 0 to 8
        # Test : acquisition 9
        # Valid : acquisition 10
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def preprocess(self, x, y):
        """
        TO DO : store the theta parameter somewhere
        TO DO : documentation for this function

        """

        def f(x, y):
            x = x.decode("utf-8")
            y = y.decode("utf-8")
            x = cv2.imread(x)
            y = cv2.imread(y)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY) 
            x = np.array(x, dtype=np.float32)
            y = np.array(y, dtype=np.float32)


            #x = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
            #y = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1


            x = x / 255
            y = y / 255
            y_binary = np.where(y > 0.5, 1, 0)
            """
            if np.any(np.isnan(x)):
                print("Found NaN in x")
            if np.any(np.isnan(y)):
                print("Found NaN in y")
            """
            if self.processor:
                x = self.processor(x)
                y = self.processor(y)
            return x, np.float32(y_binary)

        input, label = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
        input = tf.expand_dims(input, axis=-1)  # (Height,Width) -> (Height,Width,1)
        label = tf.expand_dims(label, axis=-1)  # (Height,Width) -> (Height,Width,1)
        input.set_shape([self.width, self.height, 1])
        label.set_shape([self.width, self.height, 1])

        input = tf.image.resize(input, [self.width, self.height])
        label = tf.image.resize(label, [self.width, self.height])

        return input, label

    def tf_dataset(self, x, y):
        """
        TO DO : Understand the buffer size
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (x, y)
        )  # Create a tf.data.Dataset from the couple
        ds = ds.map(
            map_func=self.preprocess,
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # apply the processing function on the couple (from path of raw image to sinongram)
        ds = ds.batch(
            self.batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
        )  # Batch the couple into batch of couple
        # ds = ds.prefetch(buffer_size=1024)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    def load_dataset(self):
        train_ds = self.tf_dataset(self.X_train, self.y_train)
        valid_ds = self.tf_dataset(self.X_valid, self.y_valid)
        test_ds = self.tf_dataset(self.X_test, self.y_test)
        return train_ds, valid_ds, test_ds

    def setup(self):
        """
        Generate the different train and test sample either as array or tf.data.Dataset
        """
        (
            (self.X_train, self.y_train),
            (self.X_valid, self.y_valid),
            (self.X_test, self.y_test),
        ) = self.load_data()
        self.train_ds, self.valid_ds, self.test_ds = self.load_dataset()

    def save(self):
        """
        TO DO : test this method for both h5 and normal format
        """
        """
        Save the train and test dataset in their corresponding path
        """
        self.train_ds.save(self.train_saving_path)
        self.test_ds.save(self.test_saving_path)
        self.valid_ds.save(self.valid_saving_path)

    def load(self):
        """
        TO DO : Loading h5 file.
        """
        self.train_ds = tf.data.Dataset.load(self.train_saving_path)
        self.test_ds = tf.data.Dataset.load(self.test_saving_path)
        self.valid_ds = tf.data.Dataset.load(self.valid_saving_path)

    def load_single_acquisition(self, acquistion_number = 1, low = False):
        no_metal_folder = self.no_metal_folder[acquistion_number]
        metal_folder = self.low_metal_fodler[acquistion_number] if low else self.high_metal_folder[acquistion_number]
        ds = self.tf_dataset(metal_folder, no_metal_folder)
        return ds
        


if __name__ == "__main__":
    print("Generating sample ....")
    dataset = SegmentationDataset(path="../data/segmentation", batch_size=32)
    dataset.setup()
    train_ds, valid_ds, test_ds = dataset.train_ds, dataset.valid_ds, dataset.test_ds
    for idx, (x, y) in enumerate(test_ds):
        fig,(ax1,ax2) = plt.subplots(1,2, figsize = (20,20))
        ax1.imshow(x[20], cmap = "gray")
        ax2.imshow(y[20], cmap = "gray")
        plt.show()
        print(y.shape)
        if idx > 1:
            break