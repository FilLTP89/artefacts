import numpy as np
import astra
import scipy.io

# Astra setup: In the Python version of ASTRA, you don't need to add paths.

values = np.linspace(-5, 5, 10)

for v in values:
    
    # Load the sino.mat
    sino = np.load("sino.npy")

    RotAxisXray = 435
    RotAxisDetect = 700 - 435
    pixelpitch = 0.085
    agreg_pixel = 1
    SizeImage = [400, 400]
    CenterImage = (np.array(SizeImage) / 2 + [5, 0]) * pixelpitch
    axis_angles = [0, 0, 0]
    n_images = 851

    angles = -np.linspace(0, (200 + v) / 180 * np.pi, n_images)
    sino = sino[:, :, 20:]
    vertical = 1

    # ASTRA setup parameters
    voxel_size = RotAxisXray / (RotAxisXray + RotAxisDetect) * (pixelpitch * agreg_pixel)
    pixel_size = 1

    source_pos = np.zeros(3)
    detect_pos = np.zeros(3)
    source_pos[1] = -(RotAxisXray + RotAxisDetect) / (pixelpitch * agreg_pixel)
    source_pos /= 2
    detect_pos = -source_pos

    shift = -(np.array(SizeImage) / 2 - CenterImage / pixelpitch) / agreg_pixel
    detect_pos -= [shift[0], 0, shift[1]]

    axis_pos = [0, 0, 0]
    vol_pos = [-100, -100, 0]
    vol_size = [700, 700, 588]

    # Compute rotation matrix of the rotation axis transformation
    # Compute rotation matrix of the rotation axis transformation
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(axis_angles[0]), -np.sin(axis_angles[0])],
                   [0, np.sin(axis_angles[0]), np.cos(axis_angles[0])]])

    Ry = np.array([[np.cos(axis_angles[1]), 0, np.sin(axis_angles[1])],
                   [0, 1, 0],
                   [-np.sin(axis_angles[1]), 0, np.cos(axis_angles[1])]])

    Rz = np.array([[np.cos(axis_angles[2]), -np.sin(axis_angles[2]), 0],
                   [np.sin(axis_angles[2]), np.cos(axis_angles[2]), 0],
                   [0, 0, 1]])

    n_angles = len(angles)
    vectors = np.zeros((n_angles, 12))

    for i in range(n_angles):

        R = np.array([[np.cos(angles[i]), -np.sin(angles[i]), 0],
                      [np.sin(angles[i]), np.cos(angles[i]), 0],
                      [0, 0, 1]])
        R = np.dot(R, np.dot(Rz, np.dot(Ry, Rx)))

        # source position
        vectors[i, 0:3] = np.dot(R, (source_pos - axis_pos)) + axis_pos - vol_pos

        # detector position
        vectors[i, 3:6] = np.dot(R, (detect_pos - axis_pos)) + axis_pos - vol_pos

        # vector from detector pixel (0,0) to (0,1)
        vectors[i, 6:9] = pixel_size * np.dot(R, [1, 0, 0])

        # vector from detector pixel (0,0) to (1,0)
        vectors[i, 9:12] = pixel_size * np.dot(R, [0, 0, 1])



    sino = np.transpose(sino, (1, 2, 0))
    proj_size = SizeImage

    vol_geom = astra.create_vol_geom(vol_size[0], vol_size[1], vol_size[2])
    proj_geom = astra.create_proj_geom('cone_vec', proj_size[1], proj_size[0], vectors)

    dims = sino.shape
    print(dims)
    id_sino = astra.data3d.create('-proj3d', proj_geom, sino)
    id_vol = astra.data3d.create('-vol', vol_geom)

    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = id_vol
    cfg['ProjectionDataId'] = id_sino
    cfg['option'] = {}
    cfg['option']['MinConstraint'] = 0
    cfg['option']['GPUIndex'] = 0

    id_alg = astra.algorithm.create(cfg)
    astra.algorithm.run(id_alg, 20)
    astra.algorithm.delete(id_alg)

    Vol = astra.data3d.get(id_vol)

    id_proj, proj_data = astra.create_sino3d_gpu(Vol, proj_geom, vol_geom)
    res = proj_data - sino

    astra.data3d.delete(id_proj)
    astra.data3d.delete(id_vol)
    astra.data3d.delete(id_sino)

    print(np.sqrt(np.mean(np.square(res[np.logical_not(np.isnan(res))]))))
