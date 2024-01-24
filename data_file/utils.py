import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
import SimpleITK as sitk
import os
import cv2

"""
TODO:
    - Redo the saving so it is in big endian, np.uint16, good scale (137, 52578)
    - Try to not have to save the little endian image in the first place
"""
def save_file(
    x,
    preds,
    y,
    name,
    path="images/generated_images/",
    extension=".raw",
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

        cmap = plt.get_cmap('gray')
    else:
        cmap = plt.cm.bone if dicom else plt.cm.gray
        x = np.array(tf.squeeze(x, axis=-1))
        preds = np.array(tf.squeeze(preds, axis=-1))
        y = np.array(tf.squeeze(y, axis=-1))

    plt.imsave(
        path + name + "_original_image" + extension,
        x,
        cmap=cmap,
        dpi = 1
    )
    plt.imsave(
        path + name + "_predicted_image" + extension,
        preds,
        cmap=cmap,
        dpi = 1
    )
    plt.imsave(
        path + name + "_ground_truth_image" + extension,
        y,
        cmap=cmap,
        dpi = 1
    )

def write_raw(array,
             file_name, 
             image_size=(400, 400), 
             image_spacing=None, 
             image_origin=None, 
             big_endian=True, 
             umin=137, 
             umax=52578):

    # Scale array back to the original range
    array = (array * (umax - umin)) + umin
    array = array.astype(np.uint16)  # convert back to uint16 (based on sitk_pixel_type)

    
    # Write to file
    # Write raw file
    """
    with open(file_name + '.raw', 'wb') as f:
        if big_endian:
            f.write(array.astype('>u2').tobytes())
        else:
            f.write(array.astype('<u2').tobytes())
    """
    
    # Convert numpy array to sitk image
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(image_spacing if image_spacing else [1]*len(image_size))
    img.SetOrigin(image_origin if image_origin else [0]*len(image_size))

    # Prepare metadata dictionary
    
    direction_cosine = [
    "1 0 0 1",  # 2D identity matrix
    "1 0 0 0 1 0 0 0 1",  # 3D identity matrix
    "1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1"  # 4D identity matrix
    ]
    
    meta_dict = {
    'ObjectType': 'Image',
    'NDims': str(len(image_size)),
    'DimSize': ' '.join(map(str, image_size)),
    'ElementSpacing': ' '.join(map(str, image_spacing)) if image_spacing else ' '.join(['1'] * len(image_size)),
    'Offset': ' '.join(map(str, image_origin)) if image_origin else ' '.join(['0'] * len(image_size)),
    'TransformMatrix': direction_cosine[len(image_size) - 2],
    'ElementType': 'MET_USHORT',  # hardcoded as we're converting array to uint16
    'BinaryData': 'True',
    'BinaryDataByteOrderMSB': str(big_endian),
    'ElementDataFile': os.path.abspath(file_name) + '.raw',
    }
    
    # Add metadata to the image
    for k, v in meta_dict.items():
        img.SetMetaData(k, v)
    
    # Use ImageFileWriter to write the image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_name + '.mhd')
    writer.SetUseCompression(False)
    writer.Execute(img)

def save_to_raw(
    x,
    preds,
    y,
    name,
    path="generated_images/",
    shape = (400,400),
    big_endian=True,
):
 
    x = x.numpy().squeeze(axis = -1)
    y = y.numpy().squeeze(axis = -1)
    preds = preds.numpy().squeeze(axis = -1)


    x = cv2.resize(x, shape)
    preds = cv2.resize(preds, shape)
    y = cv2.resize(y, shape)

    write_raw(
       array= x,
       file_name = path + name + "_original",
       image_size=shape,
       big_endian= big_endian
   )
    
    write_raw(
       array= preds,
       file_name = path + name + "_predicted",
       image_size=shape,
       big_endian= big_endian
   )
    write_raw(
       array= y,
       file_name = path + name + "_ground_truth",
       image_size=shape, 
       big_endian= big_endian
   )
    transform_to_big_endian(path + name + "_original.raw")
    transform_to_big_endian(path + name + "_predicted.raw")
    transform_to_big_endian(path + name + "_ground_truth.raw")
"""
IMPORTANT !!!!
Maybe this is useless as the bytes order doesnt seems to matter right now -> it seems to matters
"""


def transform_to_big_endian(filename):
    """
    filename (string): The path to the file containing the data to be converted. 
    """
    
    if "ground_truth"  in filename :
        category = "ground_truth/"
    elif "original" in filename:
        category = "original/"
    else : 
        category = "predicted/"
    dir = "".join([z + "/" for z in (filename.split("/")[:-1])])
    name = filename.split("/")[-1]
    big_endian_dir = dir + "big_endian/" + category
    if not os.path.exists(big_endian_dir):
        os.makedirs(big_endian_dir)
    data_little_endian = np.fromfile(filename, dtype='<u2')
    data_big_endian = data_little_endian.byteswap()
    output_file_path = big_endian_dir + name    
    data_big_endian.tofile(output_file_path)


    

if __name__ == "__main__":
    path = "../generated_test/experiment_4/acquisition_1/"
    big_endian_dir = path + "big_endian/"
    if not os.path.exists(big_endian_dir):
        os.makedirs(big_endian_dir + "predicted/")
        os.makedirs(big_endian_dir + "original/")
        os.makedirs(big_endian_dir + "ground_truth/")
    for file in os.listdir(path):
        if ".raw" in file:
            transform_to_big_endian(filename = path + file)