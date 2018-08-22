#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

from vr.models.layers import init_modules, ResidualBlock, SimpleVisualBlock, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem, ConcatBlock
import vr.programs

from torch.nn.init import kaiming_normal, kaiming_uniform, xavier_uniform, xavier_normal, constant

from vr.models.tfilmed_net import ConcatFiLMedResBlock

from vr.models.filmed_net import FiLM, FiLMedResBlock, coord_map
from functools import partial

# helper functions

# === Definition of modules for NMN === #
def shape_module(shape):
  return "Shape[{}]".format(shape)

def binary_shape_module(shape):
  return "Shape2[{}]".format(shape)

def color_module(color):
  return "Color[{}]".format(color)

def binary_color_module(color):
  return "Color2[{}]".format(color)

def relation_module(relation):
  return "Relate[{}]".format(relation)

def unary_relation_module(relation):
  return "Relate1[{}]".format(relation)


def forward_chain(image_tensor, vocab, function_modules, item_list):
  h_cur = image_tensor
  for input_ in item_list:
    h_next = []
    for j in range(input_.shape[0]):
      module_name = vocab['program_idx_to_token'][int(input_[j])]
      mod = function_modules[module_name]
      h_next.append(mod(h_cur[[j]]))
    h_cur = torch.cat(h_next)

  return h_cur

def forward_chain1(image, question, stem, vocab, function_modules, binary_function_modules, color=False):
  color_lhs = question[:, 3]
  lhs = question[:, 4]
  color_rhs = question[:, 6]
  rhs = question[:, 7]
  rel = question[:, 5]
  h_cur = stem(image)

  item_list = [color_lhs, lhs, rel, color_rhs, rhs] if color else [lhs, rel, rhs]
  return forward_chain(h_cur, vocab, function_modules, item_list)

def forward_chain2(image, question, stem, vocab, function_modules, binary_function_modules, color=False):
  color_lhs = question[:, 3]
  lhs = question[:, 4]
  color_rhs = question[:, 6]
  rhs = question[:, 7]
  rel = question[:, 5]
  h_cur = stem(image)

  item_list = [rel, color_lhs, lhs, color_rhs, rhs] if color else [rel, lhs, rhs]
  return forward_chain(h_cur, vocab, function_modules, item_list)

def forward_chain3(image, question, stem, vocab, function_modules, binary_function_modules, color=False):
  color_lhs = question[:, 3]
  lhs = question[:, 4]
  color_rhs = question[:, 6]
  rhs = question[:, 7]
  rel = question[:, 5]
  h_cur = stem(image)

  item_list = [color_lhs, lhs, color_rhs, rhs, rel] if color else [lhs, rhs, rel]
  return forward_chain(h_cur, vocab, function_modules, item_list)

def forward_tree(image, question, stem, vocab, unary_function_modules, binary_function_modules, color=False):
  h_cur = stem(image)
  h_out = []

  for j in range(question.shape[0]): 
    color_lhs = color_module(vocab['question_idx_to_token'][int(question[j, 3])])
    lhs = shape_module(vocab['question_idx_to_token'][int(question[j, 4])])
    color_rhs = color_module(vocab['question_idx_to_token'][int(question[j, 6])])
    rhs = shape_module(vocab['question_idx_to_token'][int(question[j, 7])])
    rel = relation_module(vocab['question_idx_to_token'][int(question[j, 5])])

    rel_lhs = unary_function_modules[lhs](h_cur[[j]])
    rel_rhs = unary_function_modules[rhs](h_cur[[j]])

    if color:
      lhs_color_out = unary_function_modules[color_lhs](h_cur[[j]])
      rhs_color_out = unary_function_modules[color_rhs](h_cur[[j]])
      rel_lhs = binary_function_modules['And'](rel_lhs, lhs_color_out) 
      rel_rhs = binary_function_modules['And'](rel_rhs, rhs_color_out) 

    h_out.append(binary_function_modules[rel](rel_lhs, rel_rhs))

  h_out = torch.cat(h_out)
  return h_out


FUNC_DICT = {'chain1' : forward_chain1, 'chain2' : forward_chain2, 'chain3' : forward_chain3, 'tree' : forward_tree}



class SimpleModuleNet(nn.Module):
  def __init__(self, vocab, feature_dim,
               stem_num_layers,
               stem_batchnorm,
               stem_subsample_layers,
               stem_kernel_size,
               stem_stride,
               stem_padding,
               module_dim,
               module_kernel_size,
               module_input_proj,
               forward_func,
               use_color,
               module_residual=True,
               module_batchnorm=False,
               classifier_proj_dim=512,
               classifier_downsample='maxpool2',
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               verbose=True):
    super().__init__()

    self.module_dim = module_dim
    self.func = FUNC_DICT[forward_func]
    self.use_color = use_color

    self.stem = build_stem(feature_dim[0], module_dim,
                           num_layers=stem_num_layers,
                           subsample_layers=stem_subsample_layers,
                           kernel_size=stem_kernel_size,
                           padding=stem_padding,
                           with_batchnorm=stem_batchnorm)
    tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
    module_H = tmp.size(2)
    module_W = tmp.size(3)

    self.coords = coord_map((module_H, module_W))

    if verbose:
      print('Here is my stem:')
      print(self.stem)

    num_answers = len(vocab['answer_idx_to_token'])
    self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                                       classifier_fc_layers,
                                       classifier_proj_dim,
                                       classifier_downsample,
                                       with_batchnorm=classifier_batchnorm,
                                       dropout=classifier_dropout)
    if verbose:
      print('Here is my classifier:')
      print(self.classifier)

    self.unary_function_modules = {}
    self.binary_function_modules = {}
    self.vocab = vocab
 
 
    for fn_str in vocab['program_token_to_idx']:
      if forward_func == 'tree' and fn_str in ['Relate[right_of]', 'Relate[left_of]', 'Relate[below]', 'Relate[above]', 'And'] : 
        binary_mod = ConcatBlock(
                     module_dim,
                     kernel_size=module_kernel_size,
                     with_residual=module_residual,
                     with_batchnorm=module_batchnorm)

        self.add_module(fn_str, binary_mod)
        self.binary_function_modules[fn_str] = binary_mod

      elif not fn_str.startswith('Shape2') and not fn_str.startswith('Color2'):
        mod = ResidualBlock(
              module_dim,
              kernel_size=module_kernel_size,
              with_residual=module_residual,
              with_batchnorm=module_batchnorm)

        self.add_module(fn_str, mod)
        self.unary_function_modules[fn_str] = mod

  def forward(self, image, question):
    return self.classifier(self.func(image, question, self.stem, self.vocab, self.unary_function_modules, self.binary_function_modules, self.use_color))



