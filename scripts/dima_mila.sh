#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export NMN=$HOME/Dist/film
export PYTHONPATH=$PYTHONPATH:$NMN
source activate py36clone

# override any default checkpoint path
"$@" --checkpoint_path='' --use_local_copies=0
