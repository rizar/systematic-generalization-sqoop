#!/usr/bin/env bash

set -x

export CUDA_VISIBLE_DEVICES=$ID
export PYTHONUNBUFFERED=1
export NMN=/workspace/nmn-iwp
export PYTHONPATH=$PYTHONPATH:$NMN
source activate nmn

# override any default checkpoint path
"$@" 
