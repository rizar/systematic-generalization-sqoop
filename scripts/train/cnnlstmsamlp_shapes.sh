#!/bin/bash

THIS_SCRIPT_DIR=`dirname $0`

python $THIS_SCRIPT_DIR/../train_model.py \
  --model_type CNN+LSTM+SA \
  --classifier_fc_dims 1024 \
  --feature_dim=3,30,30 \
  --checkpoint_every 100 \
  --record_loss_every 10 \
  $@

