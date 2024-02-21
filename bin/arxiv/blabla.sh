#!/bin/bash 

#SBATCH --account=neutrino 
#SBATCH --partition=neutrino 

#SBATCH --job-name=blabla
#SBATCH --output=./batch_outputs/blabla.txt 
#SBATCH --error=./batch_outputs/blabla.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=18:00:00 
#SBATCH --gpus a100:1

singularity exec --bind /sdf/ --nv /sdf/group/neutrino/images/latest.sif bash -c "python3 /sdf/group/neutrino/yjwa/lartpc_mlreco3d/bin/blabla.py"
