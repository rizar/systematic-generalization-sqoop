#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export NMN=$HOME/Dist/film
export PYTHONPATH=$PYTHONPATH:$NMN
source activate py36clone

"$@"
