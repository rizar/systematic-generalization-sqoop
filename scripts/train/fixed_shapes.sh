#!/bin/bash

THIS_SCRIPT_DIR=`dirname $0`

python $THIS_SCRIPT_DIR/../train_model.py \
  --model_type Fixed \
  --checkpoint_path "./chkpnt.pt"
  --checkpoint_every 100 \
  --record_loss_every 100 \
  --num_val_samples 149991 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --weight_decay 1e-5 \
  --feature_dim=3,30,30 \
  --module_stem_num_layers 1 \
  --module_batchnorm 1 \
