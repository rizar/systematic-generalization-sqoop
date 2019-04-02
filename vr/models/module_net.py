#!/usr/bin/env python3

# Copyright 2019-present, Mila
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

from vr.models.filmed_net import FiLM, FiLMedResBlock, ConcatFiLMedResBlock, coord_map

class ModuleNet(nn.Module):
    def __init__(self, vocab, feature_dim,
                 use_film,
                 use_simple_block,
                 sharing_patterns,
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
                 module_residual=True,
                 module_batchnorm=False,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.module_dim = module_dim

        # should be 0 or 1 to indicate the use of film block or not (0 would bring you back to the original EE model)
        self.use_film = use_film
        # should be 0 or 1 to indicate if we are using ResNets or a simple 3x3 conv followed by ReLU
        self.use_simple_block = use_simple_block

        # this should be a list of two elements (either 0 or 1). It's only active if self.use_film == 1
        # The first element of 1 indicates the sharing of CNN weights in the film blocks, 0 otheriwse
        # The second element of 1 indicate the sharing of film coefficient in the film blocks, 0 otherwise
        # so [1,0] would be sharing the CNN weights while having different film coefficients for different modules in the program
        self.sharing_patterns = sharing_patterns

        self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
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
        self.stem_times = []
        self.module_times = []
        self.classifier_times = []
        self.timing = False

        self.function_modules = {}
        self.function_modules_num_inputs = {}
        self.fn_str_2_filmId = {}
        self.vocab = vocab
        for fn_str in vocab['program_token_to_idx']:
            num_inputs = vocab['program_token_arity'][fn_str]
            self.function_modules_num_inputs[fn_str] = num_inputs

            if self.use_film:
                if self.sharing_patterns[1] == 1:
                    self.fn_str_2_filmId[fn_str] = 0
                else:
                    self.fn_str_2_filmId[fn_str] = len(self.fn_str_2_filmId)

            if fn_str == 'scene' or num_inputs == 1:
                if self.use_film:
                    if self.sharing_patterns[0] == 1:
                        mod = None
                    else:
                        mod = FiLMedResBlock(module_dim, with_residual=module_residual,
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
                else:
                    if self.use_simple_block:
                        mod = SimpleVisualBlock(module_dim, kernel_size=module_kernel_size)
                    else:
                        mod = ResidualBlock(
                                module_dim,
                                kernel_size=module_kernel_size,
                                with_residual=module_residual,
                                with_batchnorm=module_batchnorm)
            elif num_inputs == 2:
                if self.use_film:
                    if self.sharing_patterns[0] == 1:
                        mod = None
                    else:
                        mod = ConcatFiLMedResBlock(2, module_dim, with_residual=module_residual,
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
                else:
                    mod = ConcatBlock(
                              module_dim,
                              kernel_size=module_kernel_size,
                              with_residual=module_residual,
                              with_batchnorm=module_batchnorm)
            else:
                raise Exception('Not implemented!')

            if mod is not None:
                self.add_module(fn_str, mod)
                self.function_modules[fn_str] = mod

        if self.use_film and self.sharing_patterns[0] == 1:
            mod = ConcatFiLMedResBlock(2, module_dim, with_residual=module_residual,
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
            self.add_module('shared_film', mod)
            self.function_modules['shared_film'] = mod

        self.declare_film_coefficients()

        self.save_module_outputs = False

    def declare_film_coefficients(self):
        if self.use_film:
            self.gammas = nn.Parameter(torch.Tensor(1, len(self.fn_str_2_filmId), self.module_dim))
            xavier_uniform(self.gammas)
            self.betas = nn.Parameter(torch.Tensor(1, len(self.fn_str_2_filmId), self.module_dim))
            xavier_uniform(self.betas)

        else:
            self.gammas = None
            self.betas = None

    def expand_answer_vocab(self, answer_to_idx, std=0.01, init_b=-50):
        # TODO: This is really gross, dipping into private internals of Sequential
        final_linear_key = str(len(self.classifier._modules) - 1)
        final_linear = self.classifier._modules[final_linear_key]

        old_weight = final_linear.weight.data
        old_bias = final_linear.bias.data
        old_N, D = old_weight.size()
        new_N = 1 + max(answer_to_idx.values())
        new_weight = old_weight.new(new_N, D).normal_().mul_(std)
        new_bias = old_bias.new(new_N).fill_(init_b)
        new_weight[:old_N].copy_(old_weight)
        new_bias[:old_N].copy_(old_bias)

        final_linear.weight.data = new_weight
        final_linear.bias.data = new_bias

    def _forward_modules_json(self, feats, program):
        def gen_hook(i, j):
            def hook(grad):
                self.all_module_grad_outputs[i][j] = grad.data.cpu().clone()
            return hook

        self.all_module_outputs = []
        self.all_module_grad_outputs = []
        # We can't easily handle minibatching of modules, so just do a loop
        N = feats.size(0)
        final_module_outputs = []
        for i in range(N):
            if self.save_module_outputs:
                self.all_module_outputs.append([])
                self.all_module_grad_outputs.append([None] * len(program[i]))
            module_outputs = []
            for j, f in enumerate(program[i]):
                f_str = vr.programs.function_to_str(f)
                module = self.function_modules[f_str]
                if f_str == 'scene':
                    module_inputs = [feats[i:i+1]]
                else:
                    module_inputs = [module_outputs[j] for j in f['inputs']]
                module_outputs.append(module(*module_inputs))
                if self.save_module_outputs:
                    self.all_module_outputs[-1].append(module_outputs[-1].data.cpu().clone())
                    module_outputs[-1].register_hook(gen_hook(i, j))
            final_module_outputs.append(module_outputs[-1])
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return final_module_outputs

    def _forward_modules_ints_helper(self, feats, program, i, j):
        used_fn_j = True
        if j < program.size(1):
            fn_idx = program.data[i, j]
            fn_str = self.vocab['program_idx_to_token'][fn_idx.item()]
        else:
            used_fn_j = False
            fn_str = 'scene'
        if fn_str == '<NULL>':
            used_fn_j = False
            fn_str = 'scene'
        elif fn_str == '<START>':
            used_fn_j = False
            return self._forward_modules_ints_helper(feats, program, i, j + 1)
        if used_fn_j:
            self.used_fns[i, j] = 1
        j += 1

        num_inputs = self.function_modules_num_inputs[fn_str]
        if fn_str == 'scene': num_inputs = 1

        if self.use_film:
            assert fn_str in self.fn_str_2_filmId
            midx = self.fn_str_2_filmId[fn_str]

            if self.sharing_patterns[0] == 1:
                query_id = 'shared_film'
            else:
                query_id = fn_str
            assert query_id in self.function_modules
            module = self.function_modules[query_id]
        else:
            midx = -1
            module = self.function_modules[fn_str]

        if fn_str == 'scene':
            module_inputs = [feats[i:i+1]]
        else:
            #num_inputs = self.function_modules_num_inputs[fn_str]
            module_inputs = []
            while len(module_inputs) < num_inputs:
                cur_input, j = self._forward_modules_ints_helper(feats, program, i, j)
                module_inputs.append(cur_input)

        if self.use_film:
            igammas = self.gammas[:,midx,:] + 1
            ibetas =  self.betas[:,midx,:]
            bcoords = self.coords.unsqueeze(0)
            if len(module_inputs) == 1:
                if self.sharing_patterns[0] == 1:
                    module_inputs = [module_inputs[0], module_inputs[0]]
                else:
                    module_inputs = module_inputs[0]
            module_output = module(module_inputs, igammas, ibetas, bcoords)
        else:
            module_output = module(*module_inputs)
        return module_output, j

    def _forward_modules_ints(self, feats, program):
        """
        feats: FloatTensor of shape (N, C, H, W) giving features for each image
        program: LongTensor of shape (N, L) giving a prefix-encoded program for
          each image.
        """
        N = feats.size(0)
        final_module_outputs = []
        self.used_fns = torch.Tensor(program.size()).fill_(0)
        for i in range(N):
            cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0)
            final_module_outputs.append(cur_output)
        self.used_fns = self.used_fns.type_as(program.data).float()
        final_module_outputs = torch.cat(final_module_outputs, 0)
        return final_module_outputs

    def forward(self, x, program,save_activations = False ):
        N = x.size(0)
        assert N == len(program)

        feats = self.stem(x)

        if type(program) is list or type(program) is tuple:
            final_module_outputs = self._forward_modules_json(feats, program)
        elif type(program) is torch.Tensor and program.dim() == 2:
            final_module_outputs = self._forward_modules_ints(feats, program)
        elif torch.is_tensor(program) and program.dim() == 3:
            final_module_outputs = self._forward_modules_probs(feats, program)
        else:
            raise ValueError('Unrecognized program format')

        # After running modules for each input, concatenat the outputs from the
        # final module and run the classifier.
        out = self.classifier(final_module_outputs)
        return out
