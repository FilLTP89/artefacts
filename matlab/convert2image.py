import numpy as np
import os
from PIL import Image

tag = "original"
images = range(588)
im_size = (400, 400)
filename = os.path.join(tag, f"{tag}_{{:04d}}.tiff")

data = np.load(f"reconstruction_{tag}.npz")
Vol = data['Vol']

# Normalize the volume to the range [0, 2^16-1]
Vol = (Vol - np.min(Vol)) / (np.max(Vol) - np.min(Vol)) * (2**16 - 1)
Vol = Vol.astype(np.uint16)

for i in images:
    im_name = filename.format(i)
    img = Image.fromarray(Vol[:, :, i])
    img.save(im_name)
