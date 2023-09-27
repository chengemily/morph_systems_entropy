#!/bin/bash
# Parameters
#SBATCH --mem=128G
#SBATCH --qos=alien
#SBATCH --partition=alien
#SBATCH --error=/home/echeng/%j_0_log.err
#SBATCH --job-name=jupyter
#SBATCH --output=/home/echeng/%j_0_log.out
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

cd /home/echeng/morph_systems_entropy/scripts
source ~/.bashrc;
conda activate compsem;

jupyter notebook --ip $(ip addr show eno1 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1) --no-browser
