from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import radon, iradon, resize

seed = 42
np.random.seed = seed


def open_raw(
    raw_file,
    imageWidth,
    imageHeight,
    umin=137,
    umax=52578,
):
    rawInFile = open(raw_file, "rb")
    rawImageArray = np.fromfile(
        rawInFile, dtype=np.float32, count=imageHeight * imageWidth
    )
    rawRGBimage = Image.frombuffer(
        "I;16",
        [imageWidth, imageHeight],
        rawImageArray,  # .astype('I;16'),
        "raw",
        "I;16",
        0,
        1,
    )
    rawRGBarray = np.array(rawRGBimage)
    rawRGBarray.resize((imageHeight, imageWidth))

    rawRGBarray -= umin
    rawRGBarray = rawRGBarray / (umax - umin)

    return rawRGBarray


def CreateSinogram(image, theta=None, falag=1, plot=False, **kwargs):
    """
    Take an image an output the associate sinogram and the theta used.
    infomation about radon function : https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html

    Args
    -----
    image(array) : Image to be transformed into a sinogram.
    """
    if theta is not None:
        theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta)
    return sinogram, theta


def Signogram_to_Raw(sinogram, theta):
    """
    Take a sinogram and return the original raw image.

    Args
    ------
    Sinogram(array) :
    """
    reconstruction_fbp = iradon(sinogram, theta=theta, filter_name="ramp")
    return reconstruction_fbp


def Raw_to_Sinogram(
    rawFilename: str,
    imageWidth: int,
    imageHeight: int,
    mode: int,
    theta=None,
    **kwargs,
):

    rawRGBarray = open_raw(rawFilename, imageWidth, imageHeight)

    if theta is not None:
        sinogram, theta = CreateSinogram(rawRGBarray, theta)
    else:
        sinogram, theta = CreateSinogram(rawRGBarray)

    if mode == 1:
        sinogram = np.expand_dims(
            resize(
                sinogram,
                (imageHeight, imageWidth),
                mode="constant",
                preserve_range=True,
            ),
            axis=-1,
        ).astype(np.float32)

    return rawRGBarray, sinogram, theta


if __name__ == "__main__":
    Raw_to_Sinogram(
        rawFilename="../data/3/IE1705794_P406.i180073.raw",
        imageHeight=400,
        imageWidth=400,
        mode=1,
        plot=True,
    )
