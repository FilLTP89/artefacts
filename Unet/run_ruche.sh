module load anaconda3/2021.05/gcc-9.2.0
module load cuda/11.4.0/gcc-9.2.0
conda create --name tf
source activate tf
conda install -c conda-forge cudatoolkit cudnn
pip install tensorflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/gpfs/users/gattif/.conda/envs/tf/lib
pip install opencv-python-headless pillow rawpy pydicom scikit-image h5py tqdm scikit-learn
