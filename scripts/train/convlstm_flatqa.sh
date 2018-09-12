#!/bin/bash

python $NMN/scripts/train_model.py \
  --model_type ConvLSTM \
  --feature_dim 3,64,64 \
  --num_iterations 100000 \
  --num_val_samples 1000 \
  --checkpoint_every 100 \
  --record_loss_every 10 \
  --optimizer Adam \
  --batch_size 128 \
  --learning_rate 1e-4 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 6 \
  --module_stem_subsample_layers 1,3 \
  --module_dim 64 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 200 \
  --rnn_hidden_dim 300 \
  --classifier_fc_dims 1024 \
  --classifier_downsample none \
  --loader_num_workers 0 \
  --load_features \
  $@
  #--weight_decay 1e-5 \
  #--classifier_batchnorm 1 \
  #--classifier_downsample maxpoolfull \
  #--classifier_proj_dim 512 \

