#!/bin/bash
#SBATCH --job-name=sum_b_8r2
#SBATCH --output=/hits/basement/nlp/fatimamh/summarization_pytorch/out-%j
#SBATCH --error=/hits/basement/nlp/fatimamh/summarization_pytorch/err-%j
#SBATCH --time=6-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=pascal-deep.p

module load CUDA/9.2.88-GCC-7.3.0-2.30

. /home/fatimamh/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-gpu

python main.py 
