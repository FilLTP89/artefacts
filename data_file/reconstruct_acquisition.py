import tensorflow as tf
from glob import glob
import os
import re 
from CBCT_preprocess import read_raw


def load_acquisition(folder_path = "High_metal/acquisition_1"):
    all_images = sorted(
                glob(os.path.join("../data/", folder_path + "/*")),
                key=lambda x: [
                    int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)
                ],
            )
    return all_images

def read_images(all_images):
    images = []
    for i in range(len(all_images)):
        image = read_raw(all_images[i], image_size=(400,400),big_endian=True)
        image = tf.expand_dims(image, axis=-1)
        image = tf.image.resize(image, (512, 512))
        images.append(image)
    return images


def pred_images(images, model):
    predictions = []
    for image in images:
        prediction = model.predict(image)
        predictions.append(prediction)
    return predictions

if __name__ == "__main__":
    images_path = load_acquisition()
    images = read_images(images_path)
    print(images[0].shape)
    model