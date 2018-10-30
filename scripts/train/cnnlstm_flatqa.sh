#!/bin/bash

python $NMN/scripts/train_model.py \
  --model_type CNN+LSTM \
  --num_iterations 50000 \
  --feature_dim=3,64,64 \
  --num_val_samples 1000 \
  --checkpoint_every 1000 \
  --record_loss_every 10 \
  --classifier_fc_dims 1024 \
  --classifier_downsample none \
  $@

