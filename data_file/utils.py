import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageEnhance
import numpy as np
import SimpleITK as sitk
import os


def write_raw(array,
             file_name, 
             image_size=(400, 400), 
             sitk_pixel_type=sitk.sitkUInt16, 
             image_spacing=None, 
             image_origin=None, 
             big_endian=True, 
             umin=137, 
             umax=52578):


    # Scale array back to the original range
    array = (array * (umax - umin)) + umin
    array = array.astype(np.uint16)  # convert back to uint16 (based on sitk_pixel_type)

    # Write raw file
    with open(file_name + '.raw', 'wb') as f:
        if big_endian:
            f.write(array.astype('>u2').tobytes())
        else:
            f.write(array.astype('<u2').tobytes())
    
    # Convert numpy array to sitk image
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(image_spacing if image_spacing else [1]*len(image_size))
    img.SetOrigin(image_origin if image_origin else [0]*len(image_size))

    # Prepare metadata dictionary
    meta_dict = {
        'NDims': str(len(image_size)),
        'DimSize': ' '.join(map(str, image_size)),
        'ElementType': 'MET_USHORT',
        'BinaryDataByteOrderMSB': 'True' if big_endian else 'False',
        'ElementSpacing': ' '.join(map(str, img.GetSpacing())),
        'Offset': ' '.join(map(str, img.GetOrigin())),
        'ElementDataFile': os.path.basename(file_name) + '.raw',
    }

    # Add metadata to the image
    for k, v in meta_dict.items():
        img.SetMetaData(k, v)

    # Use ImageFileWriter to write the image
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_name + '.mhd')
    writer.SetUseCompression(False)
    writer.Execute(img)


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


def save_to_raw(
    x,
    preds,
    y,
    name,
    path="generated_images/",
):
    if not os.path.exists(path + name):
        os.makedirs(path + name)
 
    write_raw(
       array= x.numpy(),
       file_name = path + name + "_original",
       image_size=(512,512),
       big_endian= True
   )
    
    write_raw(
       array= preds.numpy(),
       file_name = path + name + "_predicted",
       image_size=(512,512),
       big_endian= True
   )
    write_raw(
       array= y.numpy(),
       file_name = path + name + "_ground_truth",
       image_size=(512,512), 
       big_endian= True
   )