#!/bin/bash

python $NMN/scripts/train_model.py \
  --model_type RelNet \
  --num_iterations 50000 \
  --feature_dim 3,64,64 \
  --checkpoint_every 100 \
  --record_loss_every 10 \
  --num_val_samples 1000 \
  --optimizer Adam \
  --learning_rate 2.5e-4 \
  --batch_size 64 \
  --module_stem_num_layers 4 \
  --module_stem_batchnorm 1 \
  --module_stem_kernel_size 3 \
  --module_stem_stride 2 \
  --module_stem_feature_dim 24 \
  --module_batchnorm 0 \
  --module_dim 256 \
  --module_num_layers 4 \
  --module_dropout 0 \
  --classifier_batchnorm 0 \
  --classifier_fc_dims 256,256,29 \
  --classifier_dropout 0,0.5,0 \
  --classifier_downsample none \
  --classifier_proj_dim 0 \
  --bidirectional 0 \
  --decoder_type linear \
  --encoder_type gru \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 32 \
  --rnn_hidden_dim 128 `#was 4096 in original FiLM` \
  --rnn_output_batchnorm 0 \
  $@


#RelNet CLEVR
  #--learning_rate 2.5e-4 \
  #--module_stem_num_layers 4 \
  #--module_stem_batchnorm 1 \
  #--module_stem_kernel_size 3 \
  #--module_stem_stride 2 \
  #--module_batchnorm 0 \
  #--module_dim 256 \
  #--module_num_layers 4 \
  #--classifier_fc_dims 256,256,29 \
  #--classifier_dropout 0,0.5,0 \
  #--rnn_hidden_dim 128 `#was 4096 in original FiLM` \
  #--rnn_wordvec_dim 32 \

#RelNet SortOf-CLEVR
  #--learning_rate 1e-4 \
  #--module_stem_num_layers 4 \
  #--module_stem_batchnorm 1 \
  #--module_stem_kernel_size 3 \
  #--module_stem_stride 2 \
  #--module_batchnorm 0 \
  #--module_dim 2000 \
  #--module_num_layers 4 \
  #--module_dropout 0 \
  #--classifier_fc_dims 2000,1000,500,100 \
  #--classifier_dropout 0 \