#!/bin/bash
#SBATCH --mem=96G
#SBATCH --qos=alien
#SBATCH --partition=alien
#SBATCH --error=/home/echeng/%j_0_log.err
#SBATCH --job-name=make_wiki
#SBATCH --output=/home/echeng/%j_0_log.out
#SBATCH --time=2-0:00:00

cd /home/echeng/morph_systems_entropy/scripts
source ~/.bashrc;
conda activate compsem;

python3 subsample_nouns_for_context_entropy.py \
    --lang $LANG \
    --subset $SUBSET \
    --file_number $CHUNK
