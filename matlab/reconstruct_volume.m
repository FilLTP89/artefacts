%close all
%clear all

% Astra:
addpath('/gpfs/workdir/candemilam/astra/share/astra/matlab/mex/')
addpath('/gpfs/workdir/candemilam/astra/share/astra/matlab/tools/')


values = 0;linspace(-5,5,10);

for v = values

load("sino.mat")

RotAxisXray = 435;
RotAxisDetect = 700 - 435;
pixelpitch = 0.085;
agreg_pixel = 1;
SizeImage = [400,400];
CenterImage = (SizeImage / 2 + [5,0])* pixelpitch;

axis_angles = [0,0,0];

n_images = 870;

angles = -linspace(0,(200 +v)/180*pi,n_images);

sino = sino(:,:,21:end);
%angles = angles(20:end);

vertical = 1;

% setup astra parameters
voxel_size  = RotAxisXray / (RotAxisXray + RotAxisDetect) * (pixelpitch * agreg_pixel);
pixel_size = 1;
  
source_pos = zeros(3,1);
detect_pos = zeros(3,1);
source_pos(2) = -(RotAxisXray + RotAxisDetect) / (pixelpitch * agreg_pixel);
source_pos = source_pos /2;
detect_pos = -source_pos;

shift = -(SizeImage/2 - CenterImage / pixelpitch) / agreg_pixel;
detect_pos = detect_pos - [shift(1); 0; shift(2)];

axis_pos = [0;0;0];
vol_pos = [-100;-100;0];

vol_size = [700,700,588];

% Compute rotation matrix of the rotation axis transformation
Rx = [1, 0, 0; 0, cos(axis_angles(1)), -sin(axis_angles(1)); 0, sin(axis_angles(1)), cos(axis_angles(1))];
Ry = [cos(axis_angles(2)), 0, sin(axis_angles(2)); 0, 1, 0; -sin(axis_angles(2)), 0, cos(axis_angles(2))];
Rz = [cos(axis_angles(3)), -sin(axis_angles(3)), 0; sin(axis_angles(3)), cos(axis_angles(3)), 0; 0, 0, 1];

% preallocate space
n_angles = numel(angles);
vectors  = zeros(n_angles, 12);

for i = 1:n_angles

    R = [cos(angles(i)), -sin(angles(i)), 0; sin(angles(i)), cos(angles(i)), 0; 0, 0, 1] * Rz * Ry * Rx;

    % source position
    vectors(i, 1:3)  = (R * (source_pos - axis_pos))' + axis_pos' - vol_pos';

    % detector position
    vectors(i, 4:6)  = (R * (detect_pos - axis_pos))' + axis_pos' - vol_pos';

    % vector from detector pixel (0,0) to (0,1)
    vectors(i, 7:9)  = pixel_size * (R * [1; 0; 0])';

    % vector from detector pixel (0,0) to (1,0)
    vectors(i,10:12) = pixel_size * (R * [0; 0; 1])';

end

sino = permute(sino,[2 3 1]);
proj_size = SizeImage;

vol_geom  = astra_create_vol_geom (vol_size(1), vol_size(2), vol_size(3));
proj_geom = astra_create_proj_geom('cone_vec', proj_size(2), proj_size(1), vectors);

%astra_geom_visualize(proj_geom,vol_geom)

dims = size(sino);
disp(dims);
id_sino = astra_mex_data3d('create', '-proj3d', proj_geom, sino);
id_vol  = astra_mex_data3d('create', '-vol', vol_geom);

cfg = astra_struct('FDK_CUDA');
cfg.ReconstructionDataId = id_vol;
cfg.ProjectionDataId     = id_sino;
cfg.option.MinConstraint = 0;
cfg.option.GPUIndex      = 0;

id_alg = astra_mex_algorithm('create', cfg);

% Run reconstruction
astra_mex_algorithm('iterate', id_alg, 20);
astra_mex_algorithm('delete', id_alg);

%cfg.type = 'SIRT3D_CUDA';

%id_alg = astra_mex_algorithm('create', cfg);

% Run reconstruction
%astra_mex_algorithm('iterate', id_alg, 20);
%astra_mex_algorithm('delete', id_alg);

% Collect volume
Vol = astra_mex_data3d('get_single', id_vol);

[id_proj, proj_data] = astra_create_sino3d_cuda(Vol, proj_geom, vol_geom);
res = proj_data - sino;

astra_mex_data3d('delete', id_proj);

% Delete generated storage
astra_mex_data3d('delete', id_vol);
astra_mex_data3d('delete', id_sino);


disp(rms(res(:),"omitnan"))

end
