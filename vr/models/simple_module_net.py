#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
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

    self.function_modules = {}
    self.vocab = vocab
    for fn_str in vocab['program_token_to_idx']:
      mod = ResidualBlock(
              module_dim,
              kernel_size=module_kernel_size,
              with_residual=module_residual,
              with_batchnorm=module_batchnorm)
      self.add_module(fn_str, mod)
      self.function_modules[fn_str] = mod

  def forward(self, image, question):
    color_lhs = question[:, 3]
    lhs = question[:, 4]
    color_rhs = question[:, 6]
    rhs = question[:, 7]
    rel = question[:, 5]
    h_cur = self.stem(image)

    for input_ in [color_lhs, lhs, color_rhs, rhs, rel]:
      h_next = []
      for j in range(input_.shape[0]):
        module_name = self.vocab['program_idx_to_token'][int(input_[j])]
        mod = self.function_modules[module_name]
        h_next.append(mod(h_cur[[j]]))
      h_cur = torch.cat(h_next)

    return self.classifier(h_cur)
