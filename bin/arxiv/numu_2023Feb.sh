#!/bin/bash 

#SBATCH --account=ml 
#SBATCH --partition=ml

#SBATCH --job-name=blabla
#SBATCH --output=./batch_outputs/numu_select_out.txt 
#SBATCH --error=./batch_outputs/numu_select_err.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8g
#SBATCH --time=10:00:00 
#SBATCH --gpus a100:1

singularity exec --bind /sdf/ --nv /sdf/group/neutrino/images/develop.sif bash -c "python3 /sdf/group/neutrino/yjwa/lartpc_mlreco3d/bin/numu_2023Feb.py"
