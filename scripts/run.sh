#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$HOME/Dist/film
source activate py36clone

"$@"
