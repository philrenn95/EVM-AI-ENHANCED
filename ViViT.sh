#!/bin/bash
#SBATCH --job-name=VideoVisionTransformer   
#SBATCH --nodes=1            
#SBATCH --ntasks=1                
#SBATCH --partition=p2        
#SBATCH --time=18:00:00           
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G                 
#SBATCH --gres=gpu:1
#SBATCH --qos=ultimate
#SBATCH --mail-type=ALL           
#SBATCH --mail-user=philipp.renner@th-nuernberg.de
echo "=================================================================="
echo "Starting Batch Job at $(date)"
echo "Job submitted to partition ${SLURM_JOB_PARTITION} on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "Requested ${SLURM_CPUS_ON_NODE} CPUs on compute node $(hostname)"
echo "Working directory: $(pwd)"
echo "=================================================================="
CACHE_DIR=/nfs/scratch/students/$USER/.cache
export PIP_CACHE_DIR=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_HOME=$CACHE_DIR
mkdir -p CACHE_DIR

module purge
module load python/anaconda3
module load cuda/cuda-12.3 

eval "$(conda shell.bash hook)"
conda activate framatome

srun /home/rennerph/EVM_AI_Enhanced/Main.py