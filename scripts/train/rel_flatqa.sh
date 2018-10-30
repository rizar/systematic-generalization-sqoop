#!/bin/bash
python $NMN/scripts/train_model.py \
  --feature_dim 3,64,64 \
  --checkpoint_every 1000 \
  --record_loss_every 10 \
  --num_val_samples 1000 \
\
  --model_type RelNet \
  --num_iterations 200000 \
\
  --optimizer Adam \
  --learning_rate 1e-4 \
  --batch_size 64 \
\
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 64 \
  --rnn_hidden_dim 128 \
\
  --module_stem_num_layers 8 \
  --module_stem_subsample_layers 1,3,5 \
  --module_stem_batchnorm 1 \
  --stem_dim 64 \
\
  --module_dim 256 \
  --module_num_layers 4 \
  --module_dropout 0 \
\
  --classifier_fc_dims 1024 \
  --classifier_dropout 0 \
  --classifier_batchnorm 0 \
  --classifier_downsample none \
  --classifier_proj_dim 0 \
  $@

  #--module_stem_num_layers 4 \
  #--module_stem_stride 2 \

#RelNet CLEVR
  #--learning_rate 2.5e-4 \
  #--module_stem_num_layers 4 \
  #--module_stem_batchnorm 1 \
  #--module_stem_kernel_size 3 \
  #--module_stem_stride 2 \
  #--module_batchnorm 0 \
  #--module_dim 256 \
  #--module_num_layers 4 \
  #--module_dropout 0,0.5,0 \
  #--classifier_fc_dims 256,256,29 \
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
