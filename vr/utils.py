#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import ipdb as pdb
import json
import torch

from vr.models import ModuleNet, Seq2Seq, LstmModel, CnnLstmModel, CnnLstmSaModel
from vr.models import FiLMedNet
from vr.models import FiLMGen

def invert_dict(d):
  return {v: k for k, v in d.items()}


def load_vocab(path):
  with open(path, 'r') as f:
    vocab = json.load(f)
    vocab['question_idx_to_token'] = invert_dict(vocab['question_token_to_idx'])
    vocab['program_idx_to_token'] = invert_dict(vocab['program_token_to_idx'])
    vocab['answer_idx_to_token'] = invert_dict(vocab['answer_token_to_idx'])
  # Sanity check: make sure <NULL>, <START>, and <END> are consistent
  assert vocab['question_token_to_idx']['<NULL>'] == 0
  assert vocab['question_token_to_idx']['<START>'] == 1
  assert vocab['question_token_to_idx']['<END>'] == 2
  assert vocab['program_token_to_idx']['<NULL>'] == 0
  assert vocab['program_token_to_idx']['<START>'] == 1
  assert vocab['program_token_to_idx']['<END>'] == 2
  return vocab


def load_cpu(path):
  """
  Loads a torch checkpoint, remapping all Tensors to CPU
  """
  return torch.load(path, map_location=lambda storage, loc: storage)


def load_program_generator(path, model_type='PG+EE'):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['program_generator_kwargs']
  state = checkpoint['program_generator_state']
  if model_type == 'FiLM':
    print('Loading FiLMGen from ' + path)
    kwargs = get_updated_args(kwargs, FiLMGen)
    model = FiLMGen(**kwargs)
  else:
    print('Loading PG from ' + path)
    model = Seq2Seq(**kwargs)
  model.load_state_dict(state)
  return model, kwargs


def load_execution_engine(path, verbose=True, model_type='PG+EE'):
  checkpoint = load_cpu(path)
  kwargs = checkpoint['execution_engine_kwargs']
  state = checkpoint['execution_engine_state']
  kwargs['verbose'] = verbose
  if model_type == 'FiLM':
    print('Loading FiLMedNet from ' + path)
    kwargs = get_updated_args(kwargs, FiLMedNet)
    model = FiLMedNet(**kwargs)
  else:
    print('Loading EE from ' + path)
    model = ModuleNet(**kwargs)
  cur_state = model.state_dict()
  model.load_state_dict(state)
  return model, kwargs


def load_baseline(path):
  model_cls_dict = {
    'LSTM': LstmModel,
    'CNN+LSTM': CnnLstmModel,
    'CNN+LSTM+SA': CnnLstmSaModel,
  }
  checkpoint = load_cpu(path)
  baseline_type = checkpoint['baseline_type']
  kwargs = checkpoint['baseline_kwargs']
  state = checkpoint['baseline_state']

  model = model_cls_dict[baseline_type](**kwargs)
  model.load_state_dict(state)
  return model, kwargs


def get_updated_args(kwargs, object_class):
  """
  Returns kwargs with renamed args or arg valuesand deleted, deprecated, unused args.
  Useful for loading older, trained models.
  If using this function is neccessary, use immediately before initializing object.
  """
  # Update arg values
  for arg in arg_value_updates:
    if arg in kwargs and kwargs[arg] in arg_value_updates[arg]:
      kwargs[arg] = arg_value_updates[arg][kwargs[arg]]

  # Delete deprecated, unused args
  valid_args = inspect.getargspec(object_class.__init__)[0]
  new_kwargs = {valid_arg: kwargs[valid_arg] for valid_arg in valid_args if valid_arg in kwargs}
  return new_kwargs


arg_value_updates = {
  'condition_method': {
    'block-input-fac': 'block-input-film',
    'block-output-fac': 'block-output-film',
    'cbn': 'bn-film',
    'conv-fac': 'conv-film',
    'relu-fac': 'relu-film',
  },
  'module_input_proj': {
    True: 1,
  },
}
