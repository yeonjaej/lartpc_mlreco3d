#!/bin/bash 

#SBATCH --account=ml
#SBATCH --partition=ampere

#SBATCH --job-name=blabla
#SBATCH --output=./batch_outputs/numu_select_out.txt 
#SBATCH --error=./batch_outputs/numu_select_err.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15g
#SBATCH --time=1:00:00 
#SBATCH --gpus a100:1
#SBATCH --array=0-2%2
#SBATCH --time=0-03:00:00
#
#SBATCH --gpus a100:1

singularity exec --bind /sdf,/fs/ddn/sdf --nv /sdf/group/neutrino/images/develop.sif python3 /sdf/group/neutrino/yjwa/lartpc_mlreco3d/bin/numu_2023June.py
