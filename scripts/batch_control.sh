#!/bin/bash

for lang in he # de en nl ru pl sl mt he ar sv #es fr it #ca es
do
    for chunk in {0..19}
    do
        export LANG=$lang
        export CHUNK=$chunk
        sbatch --export=ALL ./make_wiki.sh
    done
done
