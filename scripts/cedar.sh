#!/usr/bin/env bash

set -x

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:$HOME/Dist/nmn-iwp
source activate nmn

"$@"
