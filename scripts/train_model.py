#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import pdb
import random
import shutil
import sys
import time
import itertools

import h5py
import numpy as np
from termcolor import colored
import torch
torch.backends.cudnn.enabled = True
from torch.autograd import Variable
import torch.nn.functional as F

import vr.utils
import vr.preprocess
from vr.data import (ClevrDataset,
                     ClevrDataLoader)
from vr.models import (ModuleNet,
                       Seq2Seq,
                       LstmModel,
                       CnnLstmModel,
                       CnnLstmSaModel,
                       FiLMedNet,
                       TFiLMedNet,
                       RTFiLMedNet,
                       FiLMGen,
                       MAC,
                       TMAC,
                       HeteroModuleNet)
from vr.treeGenerator import TreeGenerator

parser = argparse.ArgumentParser()


def parse_int_list(input_):
  if input_ == None:
    return []
  return list(map(int, input_.split(',')))

# Input data
parser.add_argument('--data_dir', default='data')
parser.add_argument('--train_question_h5', default='train_questions.h5')
parser.add_argument('--train_features_h5', default='train_features.h5')
parser.add_argument('--val_question_h5', default='val_questions.h5')
parser.add_argument('--val_features_h5', default='val_features.h5')

parser.add_argument('--valB_question_h5', default=None)
parser.add_argument('--valB_features_h5', default=None)

parser.add_argument('--feature_dim', default='1024,14,14')
parser.add_argument('--vocab_json', default='vocab.json')

parser.add_argument('--load_features', action="store_true")
parser.add_argument('--loader_num_workers', type=int, default=1)
parser.add_argument('--use_local_copies', default=0, type=int)
parser.add_argument('--cleanup_local_copies', default=1, type=int)

parser.add_argument('--family_split_file', default=None)
parser.add_argument('--num_train_samples', default=None, type=int)
parser.add_argument('--num_val_samples', default=None, type=int)
parser.add_argument('--shuffle_train_data', default=1, type=int)

parser.add_argument('--percent_of_data_for_training', default=1., type=float)

# What type of model to use and which parts to train
parser.add_argument('--model_type', default='PG',
  choices=['RTfilm', 'Tfilm', 'FiLM', 'PG', 'EE', 'PG+EE', 'LSTM', 'CNN+LSTM', 'CNN+LSTM+SA', 'Hetero', 'MAC', 'TMAC'])
parser.add_argument('--train_program_generator', default=1, type=int)
parser.add_argument('--train_execution_engine', default=1, type=int)
parser.add_argument('--baseline_train_only_rnn', default=0, type=int)

# Start from an existing checkpoint
parser.add_argument('--program_generator_start_from', default=None)
parser.add_argument('--execution_engine_start_from', default=None)
parser.add_argument('--baseline_start_from', default=None)

# RNN options
parser.add_argument('--rnn_wordvec_dim', default=300, type=int)
parser.add_argument('--rnn_hidden_dim', default=256, type=int)
parser.add_argument('--rnn_num_layers', default=2, type=int)
parser.add_argument('--rnn_dropout', default=0, type=float)

# Module net / FiLMedNet options
parser.add_argument('--module_stem_num_layers', default=2, type=int)
parser.add_argument('--module_stem_subsample_layers', default=[], type=parse_int_list)
parser.add_argument('--module_stem_batchnorm', default=0, type=int)
parser.add_argument('--module_dim', default=128, type=int)
parser.add_argument('--module_residual', default=1, type=int)
parser.add_argument('--module_batchnorm', default=0, type=int)
parser.add_argument('--module_intermediate_batchnorm', default=0, type=int)

# FiLM only options
parser.add_argument('--set_execution_engine_eval', default=0, type=int)
parser.add_argument('--program_generator_parameter_efficient', default=1, type=int)
parser.add_argument('--rnn_output_batchnorm', default=0, type=int)
parser.add_argument('--bidirectional', default=0, type=int)
parser.add_argument('--encoder_type', default='gru', type=str,
  choices=['linear', 'gru', 'lstm'])
parser.add_argument('--decoder_type', default='linear', type=str,
  choices=['linear', 'gru', 'lstm'])
parser.add_argument('--gamma_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp'])
parser.add_argument('--gamma_baseline', default=1, type=float)
parser.add_argument('--num_modules', default=4, type=int)
parser.add_argument('--module_stem_kernel_size', default=[3], type=parse_int_list)
parser.add_argument('--module_stem_stride', default=[1], type=parse_int_list)
parser.add_argument('--module_stem_padding', default=None, type=parse_int_list)
parser.add_argument('--module_num_layers', default=1, type=int)  # Only mnl=1 currently implemented
parser.add_argument('--module_batchnorm_affine', default=0, type=int)  # 1 overrides other factors
parser.add_argument('--module_dropout', default=5e-2, type=float)
parser.add_argument('--module_input_proj', default=1, type=int)  # Inp conv kernel size (0 for None)
parser.add_argument('--module_kernel_size', default=3, type=int)
parser.add_argument('--condition_method', default='bn-film', type=str,
  choices=['nothing', 'block-input-film', 'block-output-film', 'bn-film', 'concat', 'conv-film', 'relu-film'])
parser.add_argument('--condition_pattern', default='', type=str)  # List of 0/1's (len = # FiLMs)
parser.add_argument('--use_gamma', default=1, type=int)
parser.add_argument('--use_beta', default=1, type=int)
parser.add_argument('--use_coords', default=1, type=int)  # 0: none, 1: low usage, 2: high usage
parser.add_argument('--grad_clip', default=0, type=float)  # <= 0 for no grad clipping
parser.add_argument('--debug_every', default=float('inf'), type=float)  # inf for no pdb
parser.add_argument('--print_verbose_every', default=float('inf'), type=float)  # inf for min print

parser.add_argument('--film_use_attention', default=0, type=int)

#Tfilm options
parser.add_argument('--max_program_module_arity', default=2, type=int)
parser.add_argument('--max_program_tree_depth', default=5, type=int)

#RTfilm options
parser.add_argument('--tree_type_for_RTfilm', default='complete_binary3', type=str)
parser.add_argument('--share_module_weight_at_depth', default=0, type=int)

#MAC options
parser.add_argument('--mac_write_unit', default='original', type=str)
parser.add_argument('--mac_read_connect', default='last', type=str)
parser.add_argument('--mac_vib_start', default=0, type=float)
parser.add_argument('--mac_vib_coof', default=0., type=float)
parser.add_argument('--mac_use_self_attention', default=1, type=int)
parser.add_argument('--mac_use_memory_gate', default=1, type=int)
parser.add_argument('--mac_nonlinearity', default='ELU', type=str)

parser.add_argument('--mac_question_embedding_dropout', default=0.08, type=float)
parser.add_argument('--mac_stem_dropout', default=0.18, type=float)
parser.add_argument('--mac_memory_dropout', default=0.15, type=float)
parser.add_argument('--mac_read_dropout', default=0.15, type=float)
parser.add_argument('--mac_use_prior_control_in_control_unit', default=0, type=int)
parser.add_argument('--variational_embedding_dropout', default=0.15, type=float)
parser.add_argument('--mac_embedding_uniform_boundary', default=1., type=float)

parser.add_argument('--exponential_moving_average_weight', default=1., type=float)

#TMAC options
parser.add_argument('--tree_type_for_TMAC', default='complete_binary3', type=str)
parser.add_argument('--tmac_sharing_params_patterns', default='0,1,1,1')

#NMNFilm2 options
parser.add_argument('--nmnfilm2_sharing_params_patterns', default='0,0')
parser.add_argument('--nmn_use_film', default=0, type=int)

# CNN options (for baselines)
parser.add_argument('--cnn_res_block_dim', default=128, type=int)
parser.add_argument('--cnn_num_res_blocks', default=0, type=int)
parser.add_argument('--cnn_proj_dim', default=512, type=int)
parser.add_argument('--cnn_pooling', default='maxpool2',
  choices=['none', 'maxpool2'])

# Stacked-Attention options
parser.add_argument('--stacked_attn_dim', default=512, type=int)
parser.add_argument('--num_stacked_attn', default=2, type=int)

# Classifier options
parser.add_argument('--classifier_proj_dim', default=512, type=int)
parser.add_argument('--classifier_downsample', default='maxpool2',
  choices=['maxpool2', 'maxpool3', 'maxpool4', 'maxpool5', 'maxpool7', 'maxpoolfull', 'none',
           'avgpool2', 'avgpool3', 'avgpool4', 'avgpool5', 'avgpool7', 'avgpoolfull', 'aggressive'])
parser.add_argument('--classifier_fc_dims', default='1024')
parser.add_argument('--classifier_batchnorm', default=0, type=int)
parser.add_argument('--classifier_dropout', default=0, type=float)

# Optimization options
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_iterations', default=100000, type=int)
parser.add_argument('--optimizer', default='Adam',
  choices=['Adadelta', 'Adagrad', 'Adam', 'Adamax', 'ASGD', 'RMSprop', 'SGD'])
parser.add_argument('--learning_rate', default=5e-4, type=float)
parser.add_argument('--reward_decay', default=0.9, type=float)
parser.add_argument('--weight_decay', default=0, type=float)

# Output options
parser.add_argument('--checkpoint_path', default='')
parser.add_argument('--allow-resume', action='store_true')
parser.add_argument('--randomize_checkpoint_path', type=int, default=0)
parser.add_argument('--avoid_checkpoint_override', default=0, type=int)
parser.add_argument('--record_loss_every', default=1, type=int)
parser.add_argument('--checkpoint_every', default=10000, type=int)
parser.add_argument('--time', default=0, type=int)


def main(args):
  if args.randomize_checkpoint_path == 1:
    name, ext = os.path.splitext(args.checkpoint_path)
    num = random.randint(1, 1000000)
    args.checkpoint_path = '%s_%06d%s' % (name, num, ext)
  elif not args.checkpoint_path:
    if 'SLURM_JOB_ID' in os.environ:
      args.checkpoint_path = os.environ['SLURM_JOB_ID'] + '.pt'
    else:
      raise Exception('checkpoint path not defined')
  print('Will save checkpoints to %s' % args.checkpoint_path)

  if args.data_dir:
    args.train_question_h5 = os.path.join(args.data_dir, args.train_question_h5)
    args.train_features_h5 = os.path.join(args.data_dir, args.train_features_h5)
    args.val_question_h5 = os.path.join(args.data_dir, args.val_question_h5)
    args.val_features_h5 = os.path.join(args.data_dir, args.val_features_h5)

    if args.valB_question_h5 and args.valB_features_h5:
      args.valB_question_h5 = os.path.join(args.data_dir, args.valB_question_h5)
      args.valB_features_h5 = os.path.join(args.data_dir, args.valB_features_h5)

    args.vocab_json = os.path.join(args.data_dir, args.vocab_json)

  if not args.checkpoint_path:
    if 'SLURM_JOB_ID' in os.environ:
      args.checkpoint_path = os.environ['SLURM_JOB_ID'] + '.pt'
    else:
      raise NotImplementedError()

    args.vocab_json = os.path.join(args.data_dir, args.vocab_json)
  vocab = vr.utils.load_vocab(args.vocab_json)

  def rsync(src, dst):
    os.system("rsync -vrz --progress {} {}".format(src, dst))

  def rsync_copy_if_not_exists(src, dst):
    if not os.path.exists(dst):
      rsync(src, dst)

  if args.use_local_copies == 1:
    # version for MILA, copy only if doesn't exist
    if os.path.exists('/Tmpfast'):
      tmp = '/Tmpfast/'
    else:
      tmp = '/Tmp/'
    if not os.path.exists(tmp + 'bahdanau'):
      os.mkdir(tmp + 'bahdanau')
    if not os.path.exists(tmp + 'bahdanau/clevr'):
      os.mkdir(tmp + 'bahdanau/clevr')
    root = tmp + 'bahdanau/clevr/'

    rsync_copy_if_not_exists(args.train_question_h5, root + 'train_questions.h5')
    rsync_copy_if_not_exists(args.train_features_h5, root + 'train_features.h5')
    rsync_copy_if_not_exists(args.val_question_h5, root + 'val_questions.h5')
    rsync_copy_if_not_exists(args.val_features_h5, root + 'val_features.h5')
    args.train_question_h5 = root + 'train_questions.h5'
    args.train_features_h5 = root + 'train_features.h5'
    args.val_question_h5 = root + 'val_questions.h5'
    args.val_features_h5 = root + 'val_features.h5'

    if args.valB_question_h5 and args.valB_features_h5:
      rsync_copy_if_not_exists(args.valB_question_h5, root + 'valB_questions.h5')
      rsync_copy_if_not_exists(args.valB_features_h5, root + 'valB_features.h5')
      args.valB_question_h5 = root + 'valB_questions.h5'
      args.valB_features_h5 = root + 'valB_features.h5'

  if args.use_local_copies == 2:
    tmp = os.environ['SLURM_TMPDIR']
    jobid = os.environ['SLURM_JOB_ID']
    root = os.path.join(tmp, str(jobid))
    os.mkdir(root)

    rsync(args.train_question_h5, root + 'train_questions.h5')
    rsync(args.train_features_h5, root + 'train_features.h5')
    rsync(args.val_question_h5, root + 'val_questions.h5')
    rsync(args.val_features_h5, root + 'val_features.h5')
    args.train_question_h5 = root + 'train_questions.h5'
    args.train_features_h5 = root + 'train_features.h5'
    args.val_question_h5 = root + 'val_questions.h5'
    args.val_features_h5 = root + 'val_features.h5'

  print(args)
  question_families = None
  if args.family_split_file is not None:
    with open(args.family_split_file, 'r') as f:
      question_families = json.load(f)

  train_loader_kwargs = {
    'question_h5': args.train_question_h5,
    'feature_h5': args.train_features_h5,
    'load_features': args.load_features,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'shuffle': args.shuffle_train_data == 1,
    'question_families': question_families,
    'max_samples': args.num_train_samples,
    'num_workers': args.loader_num_workers,
    'percent_of_data': args.percent_of_data_for_training,
  }
  val_loader_kwargs = {
    'question_h5': args.val_question_h5,
    'feature_h5': args.val_features_h5,
    'load_features': args.load_features,
    'vocab': vocab,
    'batch_size': args.batch_size,
    'question_families': question_families,
    'max_samples': args.num_val_samples,
    'num_workers': args.loader_num_workers,
  }

  if args.valB_question_h5 and args.valB_features_h5:
    valB_loader_kwargs = {
      'question_h5': args.valB_question_h5,
      'feature_h5': args.valB_features_h5,
      'load_features': args.load_features,
      'vocab': vocab,
      'batch_size': args.batch_size,
      'question_families': question_families,
      'max_samples': args.num_val_samples,
      'num_workers': args.loader_num_workers,
    }

  if args.valB_question_h5 and args.valB_features_h5:
    with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
         ClevrDataLoader(**val_loader_kwargs) as val_loader, \
         ClevrDataLoader(**valB_loader_kwargs) as valB_loader:
      train_loop(args, train_loader, val_loader, valB_loader)
  else:
    with ClevrDataLoader(**train_loader_kwargs) as train_loader, \
         ClevrDataLoader(**val_loader_kwargs) as val_loader:
      train_loop(args, train_loader, val_loader)

  if args.use_local_copies == 1 and args.cleanup_local_copies == 1:
    os.remove('/tmp/train_questions.h5')
    os.remove('/tmp/train_features.h5')
    os.remove('/tmp/val_questions.h5')
    os.remove('/tmp/val_features.h5')


def train_loop(args, train_loader, val_loader, valB_loader=None):
  vocab = vr.utils.load_vocab(args.vocab_json)
  program_generator, pg_kwargs, pg_optimizer = None, None, None
  execution_engine, ee_kwargs, ee_optimizer = None, None, None
  baseline_model, baseline_kwargs, baseline_optimizer = None, None, None
  baseline_type = None

  pg_best_state, ee_best_state, baseline_best_state = None, None, None

  stats = {
    'train_losses': [], 'train_rewards': [], 'train_losses_ts': [],
    'train_accs': [], 'val_accs': [], 'val_accs_ts': [],
    'best_val_acc': -1, 'model_t': 0, 'model_epoch': 0
  }
  if valB_loader:
    stats['valB_accs'] = []
    stats['valB_accs_ts'] = []
    stats['bestB_val_acc'] = []

  # Set up model
  optim_method = getattr(torch.optim, args.optimizer)
  if args.model_type in ['TMAC', 'MAC', 'RTfilm', 'Tfilm', 'FiLM', 'PG', 'PG+EE']:
    program_generator, pg_kwargs = get_program_generator(args)

    print('Here is the conditioning network:')
    print(program_generator)
  if args.model_type in ['TMAC', 'MAC', 'RTfilm', 'Tfilm', 'FiLM', 'EE', 'PG+EE', 'Hetero']:
    execution_engine, ee_kwargs = get_execution_engine(args)
    print('Here is the conditioned network:')
    print(execution_engine)
  if args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
    baseline_model, baseline_kwargs = get_baseline_model(args)
    params = baseline_model.parameters()
    if args.baseline_train_only_rnn == 1:
      params = baseline_model.rnn.parameters()
    print('Here is the baseline model')
    print(baseline_model)
    baseline_type = args.model_type

  if args.allow_resume and os.path.exists(args.checkpoint_path):
    program_generator, _ = vr.utils.load_program_generator(args.checkpoint_path, model_type=args.model_type)
    execution_engine, _  = vr.utils.load_execution_engine(args.checkpoint_path, model_type=args.model_type)
    program_generator.cuda()
    execution_engine.cuda()
    with open(args.checkpoint_path + '.json', 'r') as f:
      checkpoint = json.load(f)
    for key in list(stats.keys()):
      if key in checkpoint:
        stats[key] = checkpoint[key]
    stats['model_epoch'] -= 1
    best_pg_state = get_state(program_generator)
    best_ee_state = get_state(execution_engine)
    # no support for PG+EE her
    best_baseline_state = None

  if program_generator:
    pg_optimizer = optim_method(program_generator.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
  if execution_engine:
    ee_optimizer = optim_method(execution_engine.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
  if baseline_model:
    baseline_optimizer = optim_method(params,
                                      lr=args.learning_rate,
                                      weight_decay=args.weight_decay)

  loss_fn = torch.nn.CrossEntropyLoss().cuda()

  if args.exponential_moving_average_weight < 1. and args.model_type == 'MAC':
    EMA = vr.utils.EMA(args.exponential_moving_average_weight)

    for name, param in program_generator.named_parameters():
      if param.requires_grad:
        EMA.register('prog', name, param.data)

    for name, param in execution_engine.named_parameters():
      if param.requires_grad:
        EMA.register('exec', name, param.data)

  t, epoch, reward_moving_average = stats['model_t'], stats['model_epoch'], 0

  set_mode('train', [program_generator, execution_engine, baseline_model])

  print('train_loader has %d samples' % len(train_loader.dataset))
  print('val_loader has %d samples' % len(val_loader.dataset))
  if valB_loader:
    print('valB_loader has %d samples' % len(valB_loader.dataset))

  num_checkpoints = 0
  epoch_start_time = 0.0
  epoch_total_time = 0.0
  train_pass_total_time = 0.0
  val_pass_total_time = 0.0
  valB_pass_total_time = 0.0
  running_loss = 0.0
  while t < args.num_iterations:
    if (epoch > 0) and (args.time == 1):
      epoch_time = time.time() - epoch_start_time
      epoch_total_time += epoch_time
      print(colored('EPOCH PASS AVG TIME: ' + str(epoch_total_time / epoch), 'white'))
      print(colored('Epoch Pass Time      : ' + str(epoch_time), 'white'))
    epoch_start_time = time.time()

    epoch += 1
    print('Starting epoch %d' % epoch)

    batch_start_time = time.time()
    for batch in train_loader:
      compute_start_time = time.time()

      t += 1
      questions, _, feats, answers, programs, _ = batch
      if isinstance(questions, list):
        questions = questions[0]
      questions_var = Variable(questions.cuda())
      feats_var = Variable(feats.cuda())
      answers_var = Variable(answers.cuda())
      if programs[0] is not None:
        programs_var = Variable(programs.cuda())

      reward = None
      if args.model_type == 'PG':
        # Train program generator with ground-truth programs
        pg_optimizer.zero_grad()
        loss = program_generator(questions_var, programs_var)
        loss.backward()
        pg_optimizer.step()
      elif args.model_type in ['EE', 'Hetero']:
        # Train execution engine with ground-truth programs
        ee_optimizer.zero_grad()
        scores = execution_engine(feats_var, programs_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        ee_optimizer.step()
      elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
        baseline_optimizer.zero_grad()
        baseline_model.zero_grad()
        scores = baseline_model(questions_var, feats_var)
        loss = loss_fn(scores, answers_var)
        loss.backward()
        baseline_optimizer.step()
      elif args.model_type == 'PG+EE':
        programs_pred = program_generator.reinforce_sample(questions_var)
        scores = execution_engine(feats_var, programs_pred)

        loss = loss_fn(scores, answers_var)
        _, preds = scores.data.cpu().max(1)
        raw_reward = (preds == answers).float()
        reward_moving_average *= args.reward_decay
        reward_moving_average += (1.0 - args.reward_decay) * raw_reward.mean()
        centered_reward = raw_reward - reward_moving_average

        if args.train_execution_engine == 1:
          ee_optimizer.zero_grad()
          loss.backward()
          ee_optimizer.step()

        if args.train_program_generator == 1:
          pg_optimizer.zero_grad()
          program_generator.reinforce_backward(centered_reward.cuda())
          pg_optimizer.step()
      elif args.model_type == 'FiLM' or args.model_type == 'MAC' or args.model_type == 'TMAC':
        if args.set_execution_engine_eval == 1:
          set_mode('eval', [execution_engine])
        programs_pred = program_generator(questions_var)
        scores = execution_engine(feats_var, programs_pred)
        loss = loss_fn(scores, answers_var)
        full_loss = loss.clone()
        if args.mac_vib_coof:
          if t > args.mac_vib_start:
            print(execution_engine.vib_costs.mean())
            full_loss += args.mac_vib_coof * execution_engine.vib_costs.mean()


        pg_optimizer.zero_grad()
        ee_optimizer.zero_grad()
        if args.debug_every <= -2:
          pdb.set_trace()
        full_loss.backward()
        if args.debug_every < float('inf'):
          check_grad_num_nans(execution_engine, 'FiLMedNet' if args.model_type == 'FiLM' else args.model_type)
          check_grad_num_nans(program_generator, 'FiLMGen')

        if args.model_type == 'MAC' or args.model_type == 'TMAC':
          if args.train_program_generator == 1 or args.train_execution_engine == 1:
            if args.grad_clip > 0:
              allMacParams = itertools.chain(program_generator.parameters(), execution_engine.parameters())
              torch.nn.utils.clip_grad_norm(allMacParams, args.grad_clip)
            pg_optimizer.step()
            ee_optimizer.step()

            if args.exponential_moving_average_weight < 1. and args.model_type == 'MAC':
              for name, param in program_generator.named_parameters():
                if param.requires_grad:
                  param.data = EMA('prog', name, param.data)
              for name, param in execution_engine.named_parameters():
                if param.requires_grad:
                  param.data = EMA('exec', name, param.data)
        else:
          if args.train_program_generator == 1:
            if args.grad_clip > 0:
              torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
            pg_optimizer.step()
          if args.train_execution_engine == 1:
            if args.grad_clip > 0:
              torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
            ee_optimizer.step()
      elif args.model_type == 'Tfilm':
        if args.set_execution_engine_eval == 1:
          set_mode('eval', [execution_engine])
        programs_pred = program_generator(questions_var)
        scores = execution_engine(feats_var, programs_pred, programs_var)
        loss = loss_fn(scores, answers_var)

        pg_optimizer.zero_grad()
        ee_optimizer.zero_grad()
        if args.debug_every <= -2:
          pdb.set_trace()
        loss.backward()
        if args.debug_every < float('inf'):
          check_grad_num_nans(execution_engine, 'TFiLMedNet' if args.model_type == 'Tfilm' else 'NMNFiLMedNet')
          check_grad_num_nans(program_generator, 'FiLMGen')

        if args.train_program_generator == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
          pg_optimizer.step()
        if args.train_execution_engine == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
          ee_optimizer.step()
      elif args.model_type == 'RTfilm':
        if args.set_execution_engine_eval == 1:
          set_mode('eval', [execution_engine])
        programs_pred = program_generator(questions_var)
        scores = execution_engine(feats_var, programs_pred)
        loss = loss_fn(scores, answers_var)

        pg_optimizer.zero_grad()
        ee_optimizer.zero_grad()
        if args.debug_every <= -2:
          pdb.set_trace()
        loss.backward()
        if args.debug_every < float('inf'):
          check_grad_num_nans(execution_engine, 'RTFiLMedNet')
          check_grad_num_nans(program_generator, 'FiLMGen')

        if args.train_program_generator == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(program_generator.parameters(), args.grad_clip)
          pg_optimizer.step()
        if args.train_execution_engine == 1:
          if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm(execution_engine.parameters(), args.grad_clip)
          ee_optimizer.step()


      if t % args.record_loss_every == 0:
        running_loss += loss.data[0]
        avg_loss = running_loss / args.record_loss_every
        print(t, time.time() - batch_start_time, time.time() - compute_start_time,
              loss.data[0])
        stats['train_losses'].append(avg_loss)
        stats['train_losses_ts'].append(t)
        if reward is not None:
          stats['train_rewards'].append(reward)
        running_loss = 0.0
      else:
        running_loss += loss.data[0]


      if t % args.checkpoint_every == 0:
        num_checkpoints += 1
        print('Checking training accuracy ... ')
        start = time.time()
        train_acc = check_accuracy(args, program_generator, execution_engine,
                                   baseline_model, train_loader)
        if args.time == 1:
          train_pass_time = (time.time() - start)
          train_pass_total_time += train_pass_time
          print(colored('TRAIN PASS AVG TIME: ' + str(train_pass_total_time / num_checkpoints), 'red'))
          print(colored('Train Pass Time      : ' + str(train_pass_time), 'red'))
        print('train accuracy is', train_acc)
        print('Checking validation accuracy ...')
        start = time.time()
        val_acc = check_accuracy(args, program_generator, execution_engine,
                                 baseline_model, val_loader)
        if args.time == 1:
          val_pass_time = (time.time() - start)
          val_pass_total_time += val_pass_time
          print(colored('VAL PASS AVG TIME:   ' + str(val_pass_total_time / num_checkpoints), 'cyan'))
          print(colored('Val Pass Time        : ' + str(val_pass_time), 'cyan'))
        print('val accuracy is ', val_acc)
        stats['train_accs'].append(train_acc)
        stats['val_accs'].append(val_acc)
        stats['val_accs_ts'].append(t)

        if valB_loader:
          start=time.time()
          valB_acc = check_accuracy(args, program_generator, execution_engine,
                                    baseline_model, valB_loader)
          if args.time == 1:
            valB_pass_time = (time.time() - start)
            valB_pass_total_time += valB_pass_time
            print(colored('VAL B PASS AVG TIME:   ' + str(valB_pass_total_time / num_checkpoints), 'cyan'))
            print(colored('Val B Pass Time        : ' + str(valB_pass_time), 'cyan'))
          print('val B accuracy is ', valB_acc)
          stats['valB_accs'].append(valB_acc)
          stats['valB_accs_ts'].append(t)

        if val_acc > stats['best_val_acc']:
          stats['best_val_acc'] = val_acc
          if valB_loader: stats['bestB_val_acc'] = valB_acc
          best_pg_state = get_state(program_generator)
          best_ee_state = get_state(execution_engine)
          best_baseline_state = get_state(baseline_model)

        stats['model_t'] = t
        stats['model_epoch'] = epoch

        checkpoint = {
          'args': args.__dict__,
          'program_generator_kwargs': pg_kwargs,
          'program_generator_state': best_pg_state,
          'execution_engine_kwargs': ee_kwargs,
          'execution_engine_state': best_ee_state,
          'baseline_kwargs': baseline_kwargs,
          'baseline_state': best_baseline_state,
          'baseline_type': baseline_type,
          'vocab': vocab
        }
        for k, v in stats.items():
          checkpoint[k] = v
        print('Saving checkpoint to %s' % args.checkpoint_path)
        torch.save(checkpoint, args.checkpoint_path)
        del checkpoint['program_generator_state']
        del checkpoint['execution_engine_state']
        del checkpoint['baseline_state']
        with open(args.checkpoint_path + '.json', 'w') as f:
          json.dump(checkpoint, f, indent=2, sort_keys=True)

      if t == args.num_iterations:
        break

      batch_start_time = time.time()


def parse_int_list(s):
  if s == '': return ()
  return tuple(int(n) for n in s.split(','))


def get_state(m):
  if m is None:
    return None
  state = {}
  for k, v in m.state_dict().items():
    state[k] = v.clone()
  return state


def get_program_generator(args):
  vocab = vr.utils.load_vocab(args.vocab_json)
  if args.program_generator_start_from is not None:
    pg, kwargs = vr.utils.load_program_generator(
      args.program_generator_start_from, model_type=args.model_type)
    cur_vocab_size = pg.encoder_embed.weight.size(0)
    if cur_vocab_size != len(vocab['question_token_to_idx']):
      print('Expanding vocabulary of program generator')
      pg.expand_encoder_vocab(vocab['question_token_to_idx'])
      kwargs['encoder_vocab_size'] = len(vocab['question_token_to_idx'])
  else:
    kwargs = {
      'encoder_vocab_size': len(vocab['question_token_to_idx']),
      'decoder_vocab_size': len(vocab['program_token_to_idx']),
      'wordvec_dim': args.rnn_wordvec_dim,
      'hidden_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
    }
    if args.model_type in ['FiLM', 'Tfilm', 'RTfilm', 'MAC', 'TMAC']:
      kwargs['parameter_efficient'] = args.program_generator_parameter_efficient == 1
      kwargs['output_batchnorm'] = args.rnn_output_batchnorm == 1
      kwargs['bidirectional'] = args.bidirectional == 1
      kwargs['encoder_type'] = args.encoder_type
      kwargs['decoder_type'] = args.decoder_type
      kwargs['gamma_option'] = args.gamma_option
      kwargs['gamma_baseline'] = args.gamma_baseline
      
      kwargs['use_attention'] = args.film_use_attention == 1
      
      if args.model_type == 'FiLM' or args.model_type == 'MAC':
        kwargs['num_modules'] = args.num_modules
      elif args.model_type == 'Tfilm':
        kwargs['num_modules'] = args.max_program_module_arity * args.max_program_tree_depth + 1
      elif args.model_type == 'RTfilm':
        treeArities = TreeGenerator().gen(args.tree_type_for_RTfilm)
        kwargs['num_modules'] = len(treeArities)
      if args.model_type == 'MAC' or args.model_type == 'TMAC':
        kwargs['taking_context'] = True
        kwargs['use_attention'] = False
        kwargs['variational_embedding_dropout'] = args.variational_embedding_dropout
        kwargs['embedding_uniform_boundary'] = args.mac_embedding_uniform_boundary
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_dim'] = args.module_dim
      kwargs['debug_every'] = args.debug_every
      pg = FiLMGen(**kwargs)
    else:
      pg = Seq2Seq(**kwargs)
  pg.cuda()
  pg.train()
  return pg, kwargs


def get_execution_engine(args):
  vocab = vr.utils.load_vocab(args.vocab_json)
  if args.execution_engine_start_from is not None:
    ee, kwargs = vr.utils.load_execution_engine(
      args.execution_engine_start_from, model_type=args.model_type)
  else:
    kwargs = {
      'vocab': vocab,
      'feature_dim': parse_int_list(args.feature_dim),
      'stem_batchnorm': args.module_stem_batchnorm == 1,
      'stem_num_layers': args.module_stem_num_layers,
      'stem_subsample_layers': args.module_stem_subsample_layers,
      'stem_kernel_size': args.module_stem_kernel_size,
      'stem_stride': args.module_stem_stride,
      'stem_padding': args.module_stem_padding,
      'module_dim': args.module_dim,
      'module_kernel_size': args.module_kernel_size,
      'module_residual': args.module_residual == 1,
      'module_input_proj': args.module_input_proj,
      'module_batchnorm': args.module_batchnorm == 1,
      'classifier_proj_dim': args.classifier_proj_dim,
      'classifier_downsample': args.classifier_downsample,
      'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
      'classifier_batchnorm': args.classifier_batchnorm == 1,
      'classifier_dropout': args.classifier_dropout,
    }
    if args.model_type == 'FiLM':
      kwargs['num_modules'] = args.num_modules
      kwargs['stem_kernel_size'] = args.module_stem_kernel_size
      kwargs['stem_stride'] = args.module_stem_stride
      kwargs['stem_padding'] = args.module_stem_padding
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_intermediate_batchnorm'] = args.module_intermediate_batchnorm == 1
      kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
      kwargs['module_dropout'] = args.module_dropout
      kwargs['module_input_proj'] = args.module_input_proj
      kwargs['module_kernel_size'] = args.module_kernel_size
      kwargs['use_gamma'] = args.use_gamma == 1
      kwargs['use_beta'] = args.use_beta == 1
      kwargs['use_coords'] = args.use_coords
      kwargs['debug_every'] = args.debug_every
      kwargs['print_verbose_every'] = args.print_verbose_every
      kwargs['condition_method'] = args.condition_method
      kwargs['condition_pattern'] = parse_int_list(args.condition_pattern)
      ee = FiLMedNet(**kwargs)
    elif args.model_type == 'Tfilm':
      kwargs['num_modules'] = args.max_program_module_arity * args.max_program_tree_depth + 1

      kwargs['max_program_module_arity'] = args.max_program_module_arity
      kwargs['max_program_tree_depth'] = args.max_program_tree_depth

      kwargs['stem_kernel_size'] = args.module_stem_kernel_size
      kwargs['stem_stride'] = args.module_stem_stride
      kwargs['stem_padding'] = args.module_stem_padding
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_intermediate_batchnorm'] = args.module_intermediate_batchnorm == 1
      kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
      kwargs['module_dropout'] = args.module_dropout
      kwargs['module_input_proj'] = args.module_input_proj
      kwargs['module_kernel_size'] = args.module_kernel_size
      kwargs['use_gamma'] = args.use_gamma == 1
      kwargs['use_beta'] = args.use_beta == 1
      kwargs['use_coords'] = args.use_coords
      kwargs['debug_every'] = args.debug_every
      kwargs['print_verbose_every'] = args.print_verbose_every
      kwargs['condition_method'] = args.condition_method
      kwargs['condition_pattern'] = parse_int_list(args.condition_pattern)
      ee = TFiLMedNet(**kwargs)
    elif args.model_type == 'RTfilm':
      treeArities = TreeGenerator().gen(args.tree_type_for_RTfilm)
      kwargs['num_modules'] = len(treeArities)
      kwargs['treeArities'] = treeArities
      kwargs['tree_type_for_RTfilm'] = args.tree_type_for_RTfilm
      kwargs['share_module_weight_at_depth'] = args.share_module_weight_at_depth

      kwargs['stem_kernel_size'] = args.module_stem_kernel_size
      kwargs['stem_stride'] = args.module_stem_stride
      kwargs['stem_padding'] = args.module_stem_padding
      kwargs['module_num_layers'] = args.module_num_layers
      kwargs['module_intermediate_batchnorm'] = args.module_intermediate_batchnorm == 1
      kwargs['module_batchnorm_affine'] = args.module_batchnorm_affine == 1
      kwargs['module_dropout'] = args.module_dropout
      kwargs['module_input_proj'] = args.module_input_proj
      kwargs['module_kernel_size'] = args.module_kernel_size
      kwargs['use_gamma'] = args.use_gamma == 1
      kwargs['use_beta'] = args.use_beta == 1
      kwargs['use_coords'] = args.use_coords
      kwargs['debug_every'] = args.debug_every
      kwargs['print_verbose_every'] = args.print_verbose_every
      kwargs['condition_method'] = args.condition_method
      kwargs['condition_pattern'] = parse_int_list(args.condition_pattern)
      ee = RTFiLMedNet(**kwargs)
    elif args.model_type == 'MAC':
      kwargs = {
                'vocab': vocab,
                'feature_dim': parse_int_list(args.feature_dim),
                'stem_num_layers': args.module_stem_num_layers,
                'stem_batchnorm': args.module_stem_batchnorm == 1,
                'stem_kernel_size': args.module_stem_kernel_size,
                'stem_subsample_layers': args.module_stem_subsample_layers,
                'stem_stride': args.module_stem_stride,
                'stem_padding': args.module_stem_padding,
                'num_modules': args.num_modules,
                'module_dim': args.module_dim,

                #'module_dropout': args.module_dropout,
                'question_embedding_dropout': args.mac_question_embedding_dropout,
                'stem_dropout': args.mac_stem_dropout,
                'memory_dropout': args.mac_memory_dropout,
                'read_dropout': args.mac_read_dropout,
                'write_unit': args.mac_write_unit,
                'read_connect': args.mac_read_connect,
                'noisy_controls': bool(args.mac_vib_coof),
                'use_prior_control_in_control_unit': args.mac_use_prior_control_in_control_unit == 1,
                'use_self_attention': args.mac_use_self_attention,
                'use_memory_gate': args.mac_use_memory_gate,
                'nonlinearity': args.mac_nonlinearity,

                'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
                'classifier_batchnorm': args.classifier_batchnorm == 1,
                'classifier_dropout': args.classifier_dropout,
                'use_coords': args.use_coords,
                'debug_every': args.debug_every,
                'print_verbose_every': args.print_verbose_every,
                }
      ee = MAC(**kwargs)
    elif args.model_type == 'TMAC':
      kwargs = {
                'vocab': vocab,
                'feature_dim': parse_int_list(args.feature_dim),
                'stem_num_layers': args.module_stem_num_layers,
                'stem_batchnorm': args.module_stem_batchnorm == 1,
                'stem_kernel_size': args.module_stem_kernel_size,
                'stem_subsample_layers': args.module_stem_subsample_layers,
                'stem_stride': args.module_stem_stride,
                'stem_padding': args.module_stem_padding,
                'children_list': TreeGenerator().genHeap(args.tree_type_for_TMAC),
                'module_dim': args.module_dim,

                #'module_dropout': args.module_dropout,
                'question_embedding_dropout': args.mac_question_embedding_dropout,
                #'stem_dropout': args.mac_stem_dropout,
                #'memory_dropout': args.mac_memory_dropout,
                'read_dropout': args.mac_read_dropout,
                'use_prior_control_in_control_unit': args.mac_use_prior_control_in_control_unit == 1,
                'sharing_params_patterns': parse_int_list(args.tmac_sharing_params_patterns),
                #'use_self_attention': args.mac_use_self_attention == 1,
                #'use_memory_gate': args.mac_use_memory_gate == 1,
                #'use_memory_lstm': args.mac_use_memory_lstm == 1,
                'classifier_fc_layers': parse_int_list(args.classifier_fc_dims),
                'classifier_batchnorm': args.classifier_batchnorm == 1,
                'classifier_dropout': args.classifier_dropout,
                'use_coords': args.use_coords,
                'debug_every': args.debug_every,
                'print_verbose_every': args.print_verbose_every,
                }
      ee = TMAC(**kwargs)
    elif args.model_type == 'Hetero':
      kwargs = {
        'vocab': vocab,
        'feature_dim': parse_int_list(args.feature_dim),
        'stem_batchnorm': args.module_stem_batchnorm == 1,
        'stem_num_layers': args.module_stem_num_layers,
        'stem_kernel_size': args.module_stem_kernel_size,
        'stem_stride': args.module_stem_stride,
        'stem_padding': args.module_stem_padding,
        'module_dim': args.module_dim,
        'module_batchnorm': args.module_batchnorm == 1,
      }
      ee = HeteroModuleNet(**kwargs)
    else:
      kwargs['sharing_patterns'] = parse_int_list(args.nmnfilm2_sharing_params_patterns)
      kwargs['use_film'] = args.nmn_use_film
      ee = ModuleNet(**kwargs)
  ee.cuda()
  ee.train()
  return ee, kwargs

def get_baseline_model(args):
  vocab = vr.utils.load_vocab(args.vocab_json)
  if args.baseline_start_from is not None:
    model, kwargs = vr.utils.load_baseline(args.baseline_start_from)
  elif args.model_type == 'LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = LstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'cnn_num_res_blocks': args.cnn_num_res_blocks,
      'cnn_res_block_dim': args.cnn_res_block_dim,
      'cnn_proj_dim': args.cnn_proj_dim,
      'cnn_pooling': args.cnn_pooling,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmModel(**kwargs)
  elif args.model_type == 'CNN+LSTM+SA':
    kwargs = {
      'vocab': vocab,
      'rnn_wordvec_dim': args.rnn_wordvec_dim,
      'rnn_dim': args.rnn_hidden_dim,
      'rnn_num_layers': args.rnn_num_layers,
      'rnn_dropout': args.rnn_dropout,
      'cnn_feat_dim': parse_int_list(args.feature_dim),
      'stacked_attn_dim': args.stacked_attn_dim,
      'num_stacked_attn': args.num_stacked_attn,
      'fc_dims': parse_int_list(args.classifier_fc_dims),
      'fc_use_batchnorm': args.classifier_batchnorm == 1,
      'fc_dropout': args.classifier_dropout,
    }
    model = CnnLstmSaModel(**kwargs)
  if model.rnn.token_to_idx != vocab['question_token_to_idx']:
    # Make sure new vocab is superset of old
    for k, v in model.rnn.token_to_idx.items():
      assert k in vocab['question_token_to_idx']
      assert vocab['question_token_to_idx'][k] == v
    for token, idx in vocab['question_token_to_idx'].items():
      model.rnn.token_to_idx[token] = idx
    kwargs['vocab'] = vocab
    model.rnn.expand_vocab(vocab['question_token_to_idx'])
  model.cuda()
  model.train()
  return model, kwargs


def set_mode(mode, models):
  assert mode in ['train', 'eval']
  for m in models:
    if m is None: continue
    if mode == 'train': m.train()
    if mode == 'eval': m.eval()


def check_accuracy(args, program_generator, execution_engine, baseline_model, loader):
  set_mode('eval', [program_generator, execution_engine, baseline_model])
  num_correct, num_samples = 0, 0
  for batch in loader:
    questions, _, feats, answers, programs, _ = batch
    if isinstance(questions, list):
      questions = questions[0]

    questions_var = Variable(questions.cuda(), volatile=True)
    feats_var = Variable(feats.cuda(), volatile=True)
    answers_var = Variable(feats.cuda(), volatile=True)
    if programs[0] is not None:
      programs_var = Variable(programs.cuda(), volatile=True)

    scores = None  # Use this for everything but PG
    if args.model_type == 'PG':
      vocab = vr.utils.load_vocab(args.vocab_json)
      for i in range(questions.size(0)):
        program_pred = program_generator.sample(Variable(questions[i:i+1].cuda(), volatile=True))
        program_pred_str = vr.preprocess.decode(program_pred, vocab['program_idx_to_token'])
        program_str = vr.preprocess.decode(programs[i], vocab['program_idx_to_token'])
        if program_pred_str == program_str:
          num_correct += 1
        num_samples += 1
    elif args.model_type in ['EE', 'Hetero']:
      scores = execution_engine(feats_var, programs_var)
    elif args.model_type == 'PG+EE':
      programs_pred = program_generator.reinforce_sample(
                          questions_var, argmax=True)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type == 'FiLM' or args.model_type == 'RTfilm':
      programs_pred = program_generator(questions_var)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type == 'Tfilm':
      programs_pred = program_generator(questions_var)
      scores = execution_engine(feats_var, programs_pred, programs_var)
    elif args.model_type == 'MAC':
      programs_pred = program_generator(questions_var)
      scores = execution_engine(feats_var, programs_pred, isTest=True)
    elif args.model_type == 'TMAC':
      programs_pred = program_generator(questions_var)
      scores = execution_engine(feats_var, programs_pred)
    elif args.model_type in ['LSTM', 'CNN+LSTM', 'CNN+LSTM+SA']:
      scores = baseline_model(questions_var, feats_var)

    if scores is not None:
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == answers).sum()
      num_samples += preds.size(0)

    if args.num_val_samples is not None and num_samples >= args.num_val_samples:
      break

  set_mode('train', [program_generator, execution_engine, baseline_model])
  acc = float(num_correct) / num_samples
  return acc

def check_grad_num_nans(model, model_name='model'):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    num_nans = [np.sum(np.isnan(grad.data.cpu().numpy())) for grad in grads]
    nan_checks = [num_nan == 0 for num_nan in num_nans]
    if False in nan_checks:
      print('Nans in ' + model_name + ' gradient!')
      print(num_nans)
      pdb.set_trace()
      raise(Exception)

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
