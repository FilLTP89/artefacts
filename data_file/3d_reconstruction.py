import numpy as np
import astra
import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import imread, imwrite
from glob import glob
import sys, os
import re
sys.path.append(os.path.abspath('..'))
from data_file.CBCT_preprocess import read_raw

# Read data
def reconstruction_3d(input_dir, output_dir)
    test_folder = sorted(glob(join(input_dir, "*.raw")),key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", x)])
    
    # PARAMS
    distance_source_origin = 300  # [mm]
    distance_origin_detector = 100  # [mm]
    detector_pixel_size = 0.085  # [mm]
    detector_rows = 588  # Vertical size of detector [pixels].
    detector_cols = 588  # Horizontal size of detector [pixels].
    num_of_projections = len(test_folder)  # Number of files in the different acquisition folders
    angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)


    projections = np.zeros((detector_rows, num_of_projections, detector_cols))
    for i in range(num_of_projections):
        im = read_raw(test_folder[i], image_size = (detector_rows, detector_cols))
        im.shape
        im = im/np.max(im)
        projections[:, i, :] = im


    proj_geom = \
    astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                            (distance_source_origin + distance_origin_detector) /
                            detector_pixel_size, 0)
    projections_id = astra.data3d.create('-sino', proj_geom, projections)

    vol_geom = astra.creators.create_vol_geom(detector_cols, detector_cols,
                                            detector_rows)
    reconstruction_id = astra.data3d.create('-vol', vol_geom, data=0)
    alg_cfg = astra.astra_dict('FDK_CUDA')
    alg_cfg['ProjectionDataId'] = projections_id
    alg_cfg['ReconstructionDataId'] = reconstruction_id
    algorithm_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(algorithm_id)
    reconstruction = astra.data3d.get(reconstruction_id)


    reconstruction[reconstruction < 0] = 0
    reconstruction /= np.max(reconstruction)
    reconstruction = np.round(reconstruction * 255).astype(np.uint8)
    
    for i in range(detector_rows):
        im = reconstruction[i, :, :]
        im = np.flipud(im)
        imwrite(join(output_dir, 'reco%04d.png' % i), im)


if __name__ == '__main__':
    input_dir = "../data/no_metal/acquisition_1/"
    output_dir = '../data/3D/test/'
    reconstruction_3d(input_dir, output_dir)