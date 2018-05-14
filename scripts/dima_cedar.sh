#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export NMN=$HOME/Dist/nmn-iwp
export PYTHONPATH=$PYTHONPATH:$NMN
source activate nmn

# override any default checkpoint path
"$@" --checkpoint_path='' 
