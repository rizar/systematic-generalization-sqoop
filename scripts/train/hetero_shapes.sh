#!/bin/bash

python -m scripts.train_model \
    --data_dir /data/milatmp1/noukhovm/nmn-iwp/data/shapes_dataset \
    --model_type Hetero \
    --checkpoint_every 100 \
    --record_loss_every 10 \
    --num_val_samples 1000 \
    --optimizer Adam \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --feature_dim=3,30,30 \
    --module_stem_num_layers 2 \
    --module_stem_kernel_size 10,1 \
    --module_stem_stride 10,1 \
    --module_stem_padding 0,0 \
    --module_batchnorm 1 $@
