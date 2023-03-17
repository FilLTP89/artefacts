import pydicom
from pydicom.dataset import Dataset, FileDataset
import numpy as np
import datetime, time
from skimage.transform import radon, iradon
import os
from glob import glob
import re


def write_dicom(pixel_array, filename):
    """
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## This code block was taken from the output of a MATLAB secondary
    ## capture.  I do not know what the long dotted UIDs mean, but
    ## this code works.
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "Secondary Capture Image Storage"
    file_meta.MediaStorageSOPInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780"
    )
    file_meta.ImplementationClassUID = "1.3.6.1.4.1.9590.100.1.0.100.4.0"
    ds = FileDataset(filename, {}, file_meta=file_meta, preamble="\0" * 128)
    ds.Modality = "WSD"
    ds.ContentDate = str(datetime.date.today()).replace("-", "")
    ds.ContentTime = str(time.time())  # milliseconds since the epoch
    ds.StudyInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.124313977412360175234271287472804872093"
    )
    ds.SeriesInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.369231118011061003403421859172643143649"
    )
    ds.SOPInstanceUID = (
        "1.3.6.1.4.1.9590.100.1.1.111165684411017669021768385720736873780"
    )
    ds.SOPClassUID = "Secondary Capture Image Storage"
    ds.SecondaryCaptureDeviceManufctur = "Python 2.7.3"

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = "\\x00\\x00"
    ds.LargestImagePixelValue = "\\xff\\xff"
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(r"test.dcm")
    return


if __name__ == "__main__":
    path = "../data/dicom/"
    high_metal_folder = [
        sorted(
            glob(os.path.join(path, "high_metal/acquisition_" + str(i) + "/*")),
            key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)],
        )
        for i in range(11)
    ]
    y = pydicom.dcmread(high_metal_folder[1][200])
    x = radon(y.pixel_array, theta=np.arange(0, 180, 1))
    x = iradon(x, theta=np.arange(0, 180, 1))

    #    pixel_array = np.arange(256*256).reshape(256,256)
    #    pixel_array = np.tile(np.arange(256).reshape(16,16),(16,16))
    write_dicom(x, "pretty.dcm")
