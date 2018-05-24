#!/usr/bin/env bash

nvidia-smi -c EXCLUSIVE_PROCESS
export PYTHONUNBUFFERED=1
export NMN=/workspace/nmn-iwp
export PYTHONPATH=$PYTHONPATH:$NMN
source activate nmn

# override any default checkpoint path
"$@" 
