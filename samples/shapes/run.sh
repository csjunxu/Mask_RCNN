#!/bin/bash


echo "Add conda path"
source $HOME/anaconda3/etc/profile.d/conda.sh

echo "Activate py36"
source $HOME/anaconda3/bin/activate py36

echo "export cuda path"
export LD_LIBRARY_PATH=/home/gxdai/cudnn6/lib64:/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=$1 py_gxdai $2 
