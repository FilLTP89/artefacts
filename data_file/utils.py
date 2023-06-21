import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np


def save_file(
    x,
    preds,
    y,
    name,
    path="images/generated_images/",
    extension=".png",
    brightness_fact=3,
    big_endian=False,
    dicom=False,
):
    if big_endian:
        x = Image.fromarray(np.array(tf.squeeze(x, axis=-1) * 255, dtype=np.uint8))
        preds = Image.fromarray(
            np.array(tf.squeeze(preds, axis=-1) * 255, dtype=np.uint8)
        )
        y = Image.fromarray(np.array(tf.squeeze(y, axis=-1) * 255, dtype=np.uint8))

        enhancer_x = ImageEnhance.Brightness(x)
        enhancer_preds = ImageEnhance.Brightness(preds)
        enhancer_y = ImageEnhance.Brightness(y)

        x = enhancer_x.enhance(brightness_fact)
        preds = enhancer_preds.enhance(brightness_fact)
        y = enhancer_y.enhance(brightness_fact)

    else:
        cmap = plt.cm.bone if dicom else plt.cm.gray
        x = np.array(tf.squeeze(x, axis=-1))
        preds = np.array(tf.squeeze(preds, axis=-1))
        y = np.array(tf.squeeze(y, axis=-1))

    plt.imsave(
        path + name + "_original_image" + extension,
        x,
        cmap=cmap,
    )
    plt.imsave(
        path + name + "_predicted_image" + extension,
        preds,
        cmap=cmap,
    )
    plt.imsave(
        path + name + "_ground_truth_image" + extension,
        y,
        cmap=cmap,
    )
