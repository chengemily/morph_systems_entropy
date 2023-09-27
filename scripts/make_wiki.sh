#!/bin/bash
#SBATCH --mem=256G
#SBATCH --qos=alien
#SBATCH --partition=alien
#SBATCH --error=/home/echeng/%j_0_log.err
#SBATCH --job-name=make_wiki
#SBATCH --output=/home/echeng/%j_0_log.out
#SBATCH --gres=gpu:1
#SBATCH --time=2-0:00:00
#SBATCH --exclude=node044
#SBATCH --mail-user=emilyshana.cheng@upf.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

cd /home/echeng/morph_systems_entropy/scripts
source ~/.bashrc;
conda activate compsem;

python3 wiki_to_conllu.py \
    --lang $LANG \
    --chunk $CHUNK
