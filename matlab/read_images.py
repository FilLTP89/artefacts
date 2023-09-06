import os
import numpy as np

currentPath = os.getcwd()
print(f"The current path is: {currentPath}")

images = range(19, 890)
datatype = np.uint16
im_size = (400, 400)
endianness = 'big' if 'b' == 'b' else 'little'
filename = "../data/No_metal/acquisition_0/IE1705794_P406.i18{:04d}.raw"

sino = np.zeros((400, 400, 871), dtype=datatype)

for i in images:
    im_name = filename.format(i)
    
    with open(im_name, 'rb') as file:
        V = np.fromfile(file, dtype=datatype)
        
    # Shape it
    sino[:, :, i - 19] = V.reshape(im_size)  # adjusted the indexing here

# Get background intensity
intensity = np.mean(sino[-20:, 100:300, 349])

sino = -np.log(sino / intensity)
sino[sino < 0] = 0

np.save("sino.npy", sino)
