#!/bin/bash

python $NMN/scripts/train_model.py \
  --model_type CNN+LSTM+SA \
  --classifier_fc_dims 1024 \
  --feature_dim=3,30,30 \
  --num_val_samples 1000 \
  --checkpoint_every 100 \
  --record_loss_every 10 \
  $@

