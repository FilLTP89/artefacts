from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import radon, iradon, resize
import SimpleITK as sitk
import tempfile
import os

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

    return rawRGBarray.astype(np.float32)


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
    mode: int = 0,
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




def read_raw(
    binary_file_name,
    image_size=(400, 400),
    sitk_pixel_type=sitk.sitkUInt16,
    image_spacing=None,
    image_origin=None,
    big_endian=True,
    umin=137,
    umax=52578,
):
    pixel_dict = {
        sitk.sitkUInt16: "MET_USHORT",
    }
    direction_cosine = [
        "1 0 0 1",
        "1 0 0 0 1 0 0 0 1",
        "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1",
    ]
    dim = len(image_size)
    header = [
        "ObjectType = Image\n".encode(),
        (f"NDims = {dim}\n").encode(),
        ("DimSize = " + " ".join([str(v) for v in image_size]) + "\n").encode(),
        (
            "ElementSpacing = "
            + (
                " ".join([str(v) for v in image_spacing])
                if image_spacing
                else " ".join(["1"] * dim)
            )
            + "\n"
        ).encode(),
        (
            "Offset = "
            + (
                " ".join([str(v) for v in image_origin])
                if image_origin
                else " ".join(["0"] * dim) + "\n"
            )
        ).encode(),
        ("TransformMatrix = " + direction_cosine[dim - 2] + "\n").encode(),
        ("ElementType = " + pixel_dict[sitk_pixel_type] + "\n").encode(),
        "BinaryData = True\n".encode(),
        ("BinaryDataByteOrderMSB = " + str(big_endian) + "\n").encode(),
        # ElementDataFile must be the last entry in the header
        ("ElementDataFile = " + os.path.abspath(binary_file_name) + "\n").encode(),
    ]
    fp = tempfile.NamedTemporaryFile(suffix=".mhd", delete=False)

    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    img = sitk.ReadImage(fp.name)
    os.remove(fp.name)
    image = np.array(sitk.GetArrayFromImage(img))  # convert to numpy array
    image = (image - umin) / (umax - umin)  # normalize the array
    return image.astype(np.float32)



if __name__ == "__main__":

    Raw_to_Sinogram(
        rawFilename="../data/1/IE1705794_P406.i180395.raw",
        imageHeight=400,
        imageWidth=400,
        mode=1,
        plot=True,
    )
