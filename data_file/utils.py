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
):
    x = Image.fromarray(np.array(tf.squeeze(x, axis=-1) * 255, dtype=np.uint8))
    preds = Image.fromarray(np.array(tf.squeeze(preds, axis=-1) * 255, dtype=np.uint8))
    y = Image.fromarray(np.array(tf.squeeze(y, axis=-1) * 255, dtype=np.uint8))

    enhancer_x = ImageEnhance.Brightness(x)
    enhancer_preds = ImageEnhance.Brightness(preds)
    enhancer_y = ImageEnhance.Brightness(y)

    x = enhancer_x.enhance(brightness_fact)
    preds = enhancer_preds.enhance(brightness_fact)
    y = enhancer_y.enhance(brightness_fact)

    plt.imsave(
        path + name + "_original_image" + extension,
        x,
        cmap="gray",
    )
    plt.imsave(
        path + name + "_predicted_image" + extension,
        preds,
        cmap="gray",
    )
    plt.imsave(
        path + name + "_ground_truth_image" + extension,
        y,
        cmap="gray",
    )