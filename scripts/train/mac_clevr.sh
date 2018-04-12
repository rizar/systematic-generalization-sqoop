#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=rpp-bengioy
##SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:0:0
#SBATCH --mem=8G
#SBATCH --array=1-1   # Run a 10-job array, one job at a time.
#SBATCH --output=logs/mac2-%j.out
# ---------------------------------------------------------------------

## setrpaths.sh --path ~/.conda/envs/nmn

export PYTHONUNBUFFERED=1
source activate nmn
argm='mac2-'
name=$argm`python getJobID.py`
PYTHONPATH=/home/thiennh/project/thiennh/projects/nmn-iwp python scripts/train_model.py \
  --data_dir /home/thiennh/project/thiennh/datasets/clevr \
  --model_type MAC \
  --num_iterations 20000000 \
  --print_verbose_every 20000000 \
  --checkpoint_every 11000 \
  --checkpoint_path logs/$name.pt \
  --record_loss_every 100 \
  --num_val_samples 149991 \
  --optimizer Adam \
  --learning_rate 1e-4 \
  --batch_size 64 \
  --use_coords 1 \
  --module_stem_batchnorm 1 \
  --module_stem_num_layers 2 \
  --module_stem_kernel_size 2 \
  --module_dropout 0. \
  --module_dim 512 \
  --num_modules 12 \
  --mac_sharing_params_patterns 0,0,1,1 \
  --mac_use_self_attention 1 \
  --mac_use_memory_gate 1 \
  --bidirectional 1 \
  --encoder_type lstm \
  --weight_decay 1e-5 \
  --rnn_num_layers 1 \
  --rnn_wordvec_dim 300 \
  --rnn_hidden_dim 512 \
  --rnn_dropout 0 \
  --rnn_output_batchnorm 0 \
  --classifier_fc_dims 1024 \
  --classifier_batchnorm 0 \
  --classifier_dropout 0 \
  --use_local_copies 2 \
  --program_generator_parameter_efficient 1 $@ 
