#!/bin/bash 

#SBATCH --account=ml 
#SBATCH --partition=ml

#SBATCH --job-name=blabla
#SBATCH --output=./batch_outputs/mpr_noweight.txt 
#SBATCH --error=./batch_outputs/mpr_noweight.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4g
#SBATCH --time=2:00:00 
#SBATCH --gpus a100:1

singularity exec --bind /sdf/ --nv /sdf/group/neutrino/images/latest.sif bash -c "python3 /sdf/group/neutrino/yjwa/lartpc_mlreco3d/bin/blabla_numu_model_test_label_restrict_mpr_noweight.py"
