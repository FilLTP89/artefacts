import numpy as np
import astra
import os

tag = "original"

values = [0] + np.linspace(-5, 5, 10).tolist()

for v in values:
    data = np.load(f"sino_{tag}.npz")
    sino = data['sino']
    
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

    voxel_size = RotAxisXray / (RotAxisXray + RotAxisDetect) * (pixelpitch * agreg_pixel)
    pixel_size = 1
    
    source_pos = np.array([0, -(RotAxisXray + RotAxisDetect) / (pixelpitch * agreg_pixel) / 2, 0])
    detect_pos = -source_pos

    shift = -(np.array(SizeImage) / 2 - np.array(CenterImage) / pixelpitch) / agreg_pixel
    detect_pos -= np.array([shift[0], 0, shift[1]])

    axis_pos = np.array([0, 0, 0])
    vol_pos = np.array([-100, -100, 0])

    vol_size = [700, 700, 588]

    # Compute rotation matrices
    Rx = np.array([[1, 0, 0], [0, np.cos(axis_angles[0]), -np.sin(axis_angles[0])], [0, np.sin(axis_angles[0]), np.cos(axis_angles[0])]])
    Ry = np.array([[np.cos(axis_angles[1]), 0, np.sin(axis_angles[1])], [0, 1, 0], [-np.sin(axis_angles[1]), 0, np.cos(axis_angles[1])]])
    Rz = np.array([[np.cos(axis_angles[2]), -np.sin(axis_angles[2]), 0], [np.sin(axis_angles[2]), np.cos(axis_angles[2]), 0], [0, 0, 1]])

    n_angles = len(angles)
    vectors = np.zeros((n_angles, 12))

    for i, angle in enumerate(angles):
        R = np.array([[np.cos(angles[i]), -np.sin(angles[i]), 0],
              [np.sin(angles[i]), np.cos(angles[i]), 0],
              [0, 0, 1]]) @ Rz @ Ry @ Rx
        # source position
        vectors[i, 0:3] = (R @ (source_pos - axis_pos)).T + axis_pos.T - vol_pos.T
        # detector position
        vectors[i, 3:6] = (R @ (detect_pos - axis_pos)).T + axis_pos.T - vol_pos.T
        # vector from detector pixel (0,0) to (0,1)
        vectors[i, 6:9] = pixel_size * (R @ np.array([1, 0, 0])).T
        # vector from detector pixel (0,0) to (1,0)
        vectors[i, 9:12] = pixel_size * (R @ np.array([0, 0, 1])).T

    sino = np.transpose(sino, (1, 2, 0))
    proj_size = SizeImage

    vol_geom = astra.create_vol_geom(vol_size[0], vol_size[1], vol_size[2])
    proj_geom = astra.create_proj_geom('cone_vec', proj_size[1], proj_size[0], vectors)

    id_sino = astra.data3d.create('-proj3d', proj_geom, sino)
    id_vol = astra.data3d.create('-vol', vol_geom)

    # Create and run reconstruction
    cfg = astra.astra_dict('FDK_CUDA')
    cfg['ReconstructionDataId'] = id_vol
    cfg['ProjectionDataId'] = id_sino
    cfg['option'] = {}
    cfg['option']['MinConstraint'] = 0
    cfg['option']['GPUIndex'] = 0

    id_alg = astra.algorithm.create(cfg)
    print("Run first reconstruction")
    astra.algorithm.run(id_alg, 1)
    print("End first reconstruction")

    cfg['type'] = 'SIRT3D_CUDA'
    id_alg = astra.algorithm.create(cfg)
    print("Run second reconstruction")
    astra.algorithm.run(id_alg, 20)
    print("End second reconstruction")

    Vol = astra.data3d.get(id_vol)

    id_proj, proj_data = astra.create_sino3d_gpu(Vol, proj_geom, vol_geom)
    res = proj_data - sino

    # Clean up
    astra.data3d.delete(id_proj)
    astra.data3d.delete(id_vol)
    astra.data3d.delete(id_sino)
    astra.algorithm.delete(id_alg)

    print(np.sqrt(np.nanmean(res**2)))
    np.savez_compressed(f"reconstruction_{tag}", Vol=Vol)
