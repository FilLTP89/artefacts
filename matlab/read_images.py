import os
import numpy as np

# Get current path
current_path = os.getcwd()
print(f"The current path is: {current_path}")

# Define parameters
images = range(19, 890)  # This creates a range equivalent to 19:889 in MATLAB
datatype = np.uint16  # The numpy equivalent for MATLAB's 'uint16'
im_size = (400, 400)  # In python, tuples are often used to represent sizes or shapes
endianness = '<'  # Little-endian. Equivalent to 'b' in MATLAB.
filename = "../data/No_metal/acquisition_0/IE1705794_P406.i18{:04d}.raw"

sino = np.zeros((400, 400, 871))

# Loop through and read the images
for i in images:
    im_name = filename.format(i)
    with open(im_name, 'rb') as f:
        # Read the data
        V = np.fromfile(f, dtype=datatype)

        # Reshape it
        sino[:, :, i - 18] = V.reshape(im_size)

# Get background intensity
intensity = np.mean(sino[-21:, 100:300, 349])

sino = -np.log(sino / intensity)
sino[sino < 0] = 0

# Save the data
np.save("sino.npy", sino)  # This saves the data in numpy's .npy format