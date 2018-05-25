#!/usr/bin/env bash

set -x

export PYTHONUNBUFFERED=1
export NMN=/workspace/nmn-iwp
export PYTHONPATH=$PYTHONPATH:$NMN
source activate nmn

# override any default checkpoint path
"$@" 
