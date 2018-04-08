#!/bin/bash

source activate nmn
export PYTHONPATH="$PYTHONPATH:$PWD"

python -m scripts.train_model \
    --data_dir /data/milatmp1/noukhovm/nmn-iwp/data/shapes_dataset \
    --model_type Fixed \
    --checkpoint_every 100 \
    --record_loss_every 100 \
    --num_val_samples 149991 \
    --optimizer Adam \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --feature_dim=3,30,30 \
    --module_stem_num_layers 1 \
    --module_batchnorm 1 $@
