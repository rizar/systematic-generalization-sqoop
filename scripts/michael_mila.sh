#!/usr/bin/env bash

source ~/.bashrc

export PYTHONUNBUFFERED=1
export NMN=$HOME/nmn-iwp
export PYTHONPATH=$PYTHONPATH:$NMN
source activate nmn

"$@"
