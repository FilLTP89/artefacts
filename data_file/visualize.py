from CBCT_preprocess import Raw_to_Sinogram
import matplotlib.pyplot as plt
import random
from random import randint
import os
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf


def visualize_sample(raw_name: str, folder: list = [1, 2, 3, 4, 5], save_image=False):
    """
    Take a random file from a random folder,
    (note that is the folder is 1 or 2 or 4 the file will also exist in the other folder except 3)
    It then generate the sinogram of the different raw files, and show a plot of different images and their corresponding
    sinogram.

    Args
    ----
    raw_name(str) : name of the file
    folder(list) : list of folder to test
    save_image(Boolean) : if true we save the image else we don't
    """
    image_list = []
    folder_present = []
    for x in folder:
        try:
            image_list.append(
                Raw_to_Sinogram(
                    rawFilename=f"../data/{x}/{raw_name}",
                    imageWidth=400,
                    imageHeight=400,
                    mode=1,
                )
            )
            folder_present.append(x)
        except:
            print(f"Wasn't able to get the file in the {x} folder")

    fig, axs = plt.subplots(2, len(image_list))
    for i, image in enumerate(image_list):
        axs[0, i].set_title(str(folder_present[i]))
        axs[0, i].imshow(image[0], cmap=plt.cm.Greys_r)

        axs[1, i].imshow(
            image[1],
            cmap=plt.cm.Greys_r,
            extent=(0, 180, -image[1].shape[0] / 2.0, image[1].shape[0] / 2.0),
            aspect="auto",
        )
        axs[1, i].set_xlabel("Projection angle (deg)")
        axs[1, i].set_ylabel("Projection position (pixels)")

    fig.tight_layout()
    if save_image:
        plt.savefig("../generated_images/{:s}_sin.png".format(raw_name.strip(".raw")))
    plt.show()


def visualize_from_dataset(x, y, big_endian=True, brightness_fact=3):
    if big_endian:
        x = Image.fromarray(np.array(tf.squeeze(x, axis=-1) * 255, dtype=np.uint8))
        y = Image.fromarray(np.array(tf.squeeze(y, axis=-1) * 255, dtype=np.uint8))

        enhancer_x = ImageEnhance.Brightness(x)
        enhancer_y = ImageEnhance.Brightness(y)

        x = enhancer_x.enhance(brightness_fact)
        y = enhancer_y.enhance(brightness_fact)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(
            x,
            cmap=plt.cm.Greys_r,
            aspect="auto",
            # extent=(0, 180, -x[1].shape[0] / 2.0, x[1].shape[0] / 2.0),
        )
        ax1.set_title("With metal image")

        ax2.imshow(
            y,
            cmap=plt.cm.Greys_r,
            aspect="auto",
            # extent=(0, 180, -y[1].shape[0] / 2.0, y[1].shape[0] / 2.0),
        )
        ax2.set_title("Without metal image")
        plt.show()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(
            x,
            cmap=plt.cm.Greys_r,
            aspect="auto",
            # extent=(0, 180, -x[1].shape[0] / 2.0, x[1].shape[0] / 2.0),
        )
        ax1.set_title("With artefacts image")
        ax2.imshow(
            y,
            cmap=plt.cm.Greys_r,
            aspect="auto",
            # extent=(0, 180, -y[1].shape[0] / 2.0, y[1].shape[0] / 2.0),
        )
        ax2.set_title("Without metal image")
        plt.show()


def save_image_result(x, preds, y, name):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(
        x,
        cmap=plt.cm.Greys_r,
        aspect="auto",
        # extent=(0, 180, -x[1].shape[0] / 2.0, x[1].shape[0] / 2.0),
    )
    ax1.set_title("With artefacts image")

    ax2.imshow(
        preds,
        cmap=plt.cm.Greys_r,
        aspect="auto",
        # extent=(0, 180, -preds[1].shape[0] / 2.0, preds[1].shape[0] / 2.0),
    )
    ax2.set_title("Predicted image")

    ax3.imshow(
        y,
        cmap=plt.cm.Greys_r,
        aspect="auto",
        # extent=(0, 180, -y[1].shape[0] / 2.0, y[1].shape[0] / 2.0),
    )
    ax3.set_title("Without artefacts image")
    plt.savefig(f"images/generated_images/{name}.png")


def visualize_results(x, y, preds):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(
        x,
        cmap=plt.cm.Greys_r,
        aspect="auto",
        extent=(0, 180, -x[1].shape[0] / 2.0, x[1].shape[0] / 2.0),
    )
    ax1.set_title("With artefacts image")

    ax2.imshow(
        preds,
        cmap=plt.cm.Greys_r,
        aspect="auto",
        extent=(0, 180, -preds[1].shape[0] / 2.0, preds[1].shape[0] / 2.0),
    )
    ax2.set_title("Predicted image")

    ax3.imshow(
        y,
        cmap=plt.cm.Greys_r,
        aspect="auto",
        extent=(0, 180, -y[1].shape[0] / 2.0, y[1].shape[0] / 2.0),
    )
    ax3.set_title("Without artefacts image")
    plt.show()


if __name__ == "__main__":
    folder = f"{randint(1, 4)}"
    files = os.listdir(f"../data/{folder}/")

    random_file = random.choice(files)
    visualize_sample(random_file, save_image=False)

    random_file_w_path = f"../data/{folder}/{random_file}"


"""
Folder 3 has 254 images where 
Folder 1, 2 & 3 has 890
"""
