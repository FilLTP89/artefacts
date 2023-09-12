import numpy as np
import os

tag = "original"
images = range(19,890)
datatype = np.uint16
im_size = (400, 400)
endianness = 'big'  # 'b' in MATLAB is big-endian
filename = "../data/No_metal/acquisition_0/IE1705794_P406.i18{:04d}.raw"

print(filename)
sino = np.zeros((400, 400, 871))

for i in images:
    im_name = filename.format(i)
    print(im_name)
    
    # Read the data
    with open(im_name, 'rb') as f:
        if endianness == 'big':
            V = np.fromfile(f, dtype=datatype).byteswap()
        else:
            V = np.fromfile(f, dtype=datatype)
    
    # Reshape it
    sino[:, :, i-1] = V.reshape(im_size)

# Get background intensity
intensity = np.mean(sino[-21:, 100:300, 350])

sino = -np.log(sino/intensity)
sino[sino < 0] = 0

np.savez_compressed(f"sino_{tag}", sino=sino)
