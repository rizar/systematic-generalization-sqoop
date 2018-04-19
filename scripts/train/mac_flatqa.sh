#!/bin/bash

THIS_SCRIPT_DIR=`dirname $0`

python $THIS_SCRIPT_DIR/../train_model.py \
  --model_type MAC \
  --feature_dim=3,64,64 \
  --num_iterations 20000 \
  --checkpoint_every 100 \
  --record_loss_every 10 \
  --num_val_samples 1000 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --batch_size 64 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 6 \
  --module_stem_subsample_layers 1,3 \
  --module_stem_kernel_size 3 \
  --mac_question_embedding_dropout 0.08 \
  --mac_stem_dropout 0.18 \
  --mac_memory_dropout 0.15 \
  --mac_read_dropout 0.15 \
  --mac_write_dropout 0. \
  --mac_use_prior_control_in_control_unit 0 \
  --variational_embedding_dropout 0.15 \
  --module_dim 256 \
  --num_modules 4 \
  --mac_sharing_params_patterns 0,1,1,1 \
  --mac_use_self_attention 0 \
  --mac_use_memory_gate 0 \
  --bidirectional 1 \
  --encoder_type lstm \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 300 \
  --rnn_hidden_dim 256 \
  --rnn_dropout 0 \
  --rnn_output_batchnorm 0 \
  --classifier_fc_dims 1024 \
  --classifier_batchnorm 0 \
  --classifier_dropout 0.15 \
  --use_local_copies 0 \
  --program_generator_parameter_efficient 1 $@
