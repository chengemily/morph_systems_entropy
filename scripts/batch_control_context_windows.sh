#!/bin/bash

for lang in en
do
    for chunk in {0..19}
    do
        for subset in all
        do
            export SUBSET=$subset
            export LANG=$lang
            export CHUNK=$chunk
            sbatch --export=ALL ./collect_contexts.sh
        done
    done
done
