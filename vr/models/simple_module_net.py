#!/usr/bin/env python3

# Copyright 2019-present, Mila
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

from vr.models.filmed_net import FiLM, FiLMedResBlock, ConcatFiLMedResBlock,coord_map
from functools import partial

# helper functions

# === Definition of modules for NMN === #
def shape_module(shape):
    return "Shape[{}]".format(shape)

def binary_shape_module(shape):
    return "Shape2[{}]".format(shape)

def relation_module(relation):
    return "Relate[{}]".format(relation)

def unary_relation_module(relation):
    return "Relate1[{}]".format(relation)


def forward_chain(image_tensor, vocab, function_modules, item_list, film_params):
    gammas, betas, coords = None, None, None
    if film_params is not None:
        gammas, betas, coords = film_params

    h_cur = image_tensor
    for input_ in item_list:
        h_next = []
        for j in range(input_.shape[0]):

            if gammas is not None:
                item_idx = int(input_[j])
                mod = function_modules['film']
                h_next.append(mod(h_cur[[j]], gammas[:, item_idx, :], betas[:, item_idx, :], coords))
            else:
                module_name = vocab['program_idx_to_token'][int(input_[j])]
                mod = function_modules[module_name]
                h_next.append(mod(h_cur[[j]]))

        h_cur = torch.cat(h_next)

    return h_cur


def forward_chain1(image, question, stem, vocab, function_modules, binary_function_modules, film_params=None):
    lhs = question[:, 0]
    rhs = question[:, 2]
    rel = question[:, 1]

    item_list = [lhs, rel, rhs]
    return forward_chain(stem(image), vocab, function_modules, item_list, film_params)


def forward_chain2(image, question, stem, vocab, function_modules, binary_function_modules, film_params=None):
    lhs = question[:, 0]
    rhs = question[:, 2]
    rel = question[:, 1]

    item_list = [lhs, rhs, rel]
    return forward_chain(stem(image), vocab, function_modules, item_list, film_params)


def forward_chain3(image, question, stem, vocab, function_modules, binary_function_modules, film_params=None):
    lhs = question[:, 0]
    rhs = question[:, 2]
    rel = question[:, 1]

    item_list = [rel, lhs, rhs]
    return forward_chain(stem(image), vocab, function_modules, item_list, film_params)


def forward_tree(image, question, stem, vocab, unary_function_modules, binary_function_modules, film_params=None):
    h_cur = stem(image)
    h_out = []

    gammas, betas, coords = None, None, None
    if film_params is not None:
        gammas, betas, coords = film_params

    lhs = question[:, 0]
    rhs = question[:, 2]
    rel = question[:, 1]

    for j in range(question.shape[0]):

        lhs_idx = int(lhs[j])
        rel_idx = int(rel[j])
        rhs_idx = int(rhs[j])

        lhs = shape_module(vocab['question_idx_to_token'][lhs_idx])
        rel = relation_module(vocab['question_idx_to_token'][rel_idx])
        rhs = shape_module(vocab['question_idx_to_token'][rhs_idx])

        if gammas is not None:
            rel_lhs = unary_function_modules['film'](h_cur[[j]], gammas[:, lhs_idx, :], betas[:, lhs_idx, :], coords)
            rel_rhs = unary_function_modules['film'](h_cur[[j]], gammas[:, rhs_idx, :], betas[:, rhs_idx, :], coords)

            h_out.append(binary_function_modules['film']([rel_lhs, rel_rhs], gammas[:, rel_idx, :], betas[:, rel_idx, :], coords ))
        else:
            rel_lhs = unary_function_modules[lhs](h_cur[[j]])
            rel_rhs = unary_function_modules[rhs](h_cur[[j]])

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
                 stem_dim,
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
                 use_film=False,
                 verbose=True):
        super().__init__()

        self.module_dim = module_dim
        self.func = FUNC_DICT[forward_func]
        self.use_color = use_color

        self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
                               num_layers=stem_num_layers,
                               subsample_layers=stem_subsample_layers,
                               kernel_size=stem_kernel_size,
                               padding=stem_padding,
                               with_batchnorm=stem_batchnorm)
        tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
        module_H = tmp.size(2)
        module_W = tmp.size(3)

        self.coords = coord_map((module_H, module_W)).unsqueeze(0)

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
        self.use_film = use_film

        if self.use_film:
            unary_mod = FiLMedResBlock(module_dim, with_residual=module_residual,
                          with_intermediate_batchnorm=False, with_batchnorm=False,
                          with_cond=[True, True],
                          num_extra_channels=2, # was 2 for original film,
                          extra_channel_freq=1,
                          with_input_proj=module_input_proj,
                          num_cond_maps=0,
                          kernel_size=module_kernel_size,
                          batchnorm_affine=False,
                          num_layers=1,
                          condition_method='bn-film',
                          debug_every=float('inf'))
            binary_mod = ConcatFiLMedResBlock(2, module_dim, with_residual=module_residual,
                          with_intermediate_batchnorm=False, with_batchnorm=False,
                          with_cond=[True, True],
                          num_extra_channels=2, #was 2 for original film,
                          extra_channel_freq=1,
                          with_input_proj=module_input_proj,
                          num_cond_maps=0,
                          kernel_size=module_kernel_size,
                          batchnorm_affine=False,
                          num_layers=1,
                          condition_method='bn-film',
                          debug_every=float('inf'))

            self.unary_function_modules['film'] = unary_mod
            self.binary_function_modules['film'] = binary_mod
            self.add_module('film_unary', unary_mod)
            self.add_module('film_binary', binary_mod)


        else:
            for fn_str in vocab['program_token_to_idx']:
                arity = self.vocab['program_token_arity'][fn_str]
                if arity == 2 and forward_func == 'tree':
                    binary_mod = ConcatBlock(
                                 module_dim,
                                 kernel_size=module_kernel_size,
                                 with_residual=module_residual,
                                 with_batchnorm=module_batchnorm,
                                 use_simple=False)

                    self.add_module(fn_str, binary_mod)
                    self.binary_function_modules[fn_str] = binary_mod

                else:
                    mod = ResidualBlock(
                          module_dim,
                          kernel_size=module_kernel_size,
                          with_residual=module_residual,
                          with_batchnorm=module_batchnorm)

                    self.add_module(fn_str, mod)
                    self.unary_function_modules[fn_str] = mod

        self.declare_film_coefficients()

    def declare_film_coefficients(self):
        num_coeff = 1+len(self.vocab['question_idx_to_token'])
        if self.use_film:
            self.gammas = nn.Parameter(torch.Tensor(1, num_coeff, self.module_dim))
            xavier_uniform(self.gammas)
            self.betas = nn.Parameter(torch.Tensor(1, num_coeff, self.module_dim))
            xavier_uniform(self.betas)

        else:
            self.gammas = None
            self.betas = None

    def forward(self, image, question):
        return self.classifier(self.func(image, question, self.stem, self.vocab, self.unary_function_modules, self.binary_function_modules, [self.gammas, self.betas, self.coords]))
