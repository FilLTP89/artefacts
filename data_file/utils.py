import matplotlib.pyplot as plt
import tensorflow as tf


def save_file(x, preds, y, name, path="images/generated_images/", extension=".png"):
    plt.imsave(
        path + name + "original_image" + extension, tf.squeeze(x, axis=-1), cmap="gray"
    )
    plt.imsave(
        path + name + "predicted_image" + extension,
        tf.squeeze(preds, axis=-1),
        cmap="gray",
    )
    plt.imsave(
        path + name + "ground_truth_image" + extension,
        tf.squeeze(y, axis=-1),
        cmap="gray",
    )
