#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export PYTHONPATH=$PYTHONPATH:`dirname $0`/..
source activate py36clone

"$@"
