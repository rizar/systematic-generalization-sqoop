#!/usr/bin/env python3

import math
import numpy
import pprint
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

from vr.models.layers import init_modules, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem
import vr.programs
from vr.models.tfilmed_net import ConcatFiLMedResBlock

from vr.models.filmed_net import FiLM, FiLMedResBlock, coord_map

class RTFiLMedNet(nn.Module):
  def __init__(self, vocab, feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               stem_kernel_size=3,
               stem_subsample_layers=None,
               stem_stride=1,
               stem_padding=None,
               num_modules=4,

               tree_type_for_RTfilm='complete_binary',
               treeArities=None,
               share_module_weight_at_depth=0,

               module_num_layers=1,
               module_dim=128,
               module_residual=True,
               module_intermediate_batchnorm=False,
               module_batchnorm=False,
               module_batchnorm_affine=False,
               module_dropout=0,
               module_input_proj=1,
               module_kernel_size=3,
               classifier_proj_dim=512,
               classifier_downsample='maxpool2',
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               condition_method='bn-film',
               condition_pattern=[],
               use_gamma=True,
               use_beta=True,
               use_coords=1,
               debug_every=float('inf'),
               print_verbose_every=float('inf'),
               verbose=True,
               ):
    super(RTFiLMedNet, self).__init__()

    num_answers = len(vocab['answer_idx_to_token'])

    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False

    self.num_modules = num_modules

    self.tree_type_for_RTfilm = tree_type_for_RTfilm
    self.treeArities = treeArities
    self.share_module_weight_at_depth = share_module_weight_at_depth == 1

    self.module_num_layers = module_num_layers
    self.module_batchnorm = module_batchnorm
    self.module_dim = module_dim
    self.condition_method = condition_method
    self.use_gamma = use_gamma
    self.use_beta = use_beta
    self.use_coords_freq = use_coords
    self.debug_every = debug_every
    self.print_verbose_every = print_verbose_every

    # Initialize helper variables
    self.stem_use_coords = (stem_stride == 1) and (self.use_coords_freq > 0)
    self.condition_pattern = condition_pattern

    self.extra_channel_freq = self.use_coords_freq
    #self.block = FiLMedResBlock
    self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
    self.fwd_count = 0
    self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
    if self.debug_every <= -1:
      self.print_verbose_every = 1

    stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
    self.stem = build_stem(
      stem_feature_dim, module_dim,
      num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
      kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding,
      subsample_layers=stem_subsample_layers)
    tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
    module_H = tmp.size(2)
    module_W = tmp.size(3)

    self.stem_coords = coord_map((feature_dim[1], feature_dim[2]))
    self.coords = coord_map((module_H, module_W))
    self.default_weight = Variable(torch.ones(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)
    self.default_bias = Variable(torch.zeros(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)

    # Initialize stem
    #stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
    #self.stem = build_stem(stem_feature_dim, module_dim,
    #                       num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
    #                       kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding,
    #                       subsample_layers=stem_subsample_layers)

    # Initialize Tfilmed network body
    self.function_modules = {}
    if self.share_module_weight_at_depth:
      self.depth_to_arity = {}
    self.vocab = vocab

    def generateModules(i, j):
      if i >= len(self.treeArities): return -1
      art = self.treeArities[i]
      if len(self.condition_pattern) == 0:
        with_cond = [self.condition_method != 'concat'] * (2*self.module_num_layers)
      else:
        with_cond = []
        if len(self.condition_pattern) > (2*i): with_cond.append(self.condition_pattern[2*i] > 0)
        else: with_cond.append(self.condition_pattern[-1] > 0)
        if len(self.condition_pattern) > (2*i+1): with_cond.append(self.condition_pattern[2*i+1] > 0)
        else: with_cond.append(self.condition_pattern[-1] > 0)
        with_cond = with_cond * self.module_num_layers

      if not self.share_module_weight_at_depth or j not in self.depth_to_arity:
        if art == 1 or art == 0:
          mod = FiLMedResBlock(module_dim, with_residual=module_residual,
                         with_intermediate_batchnorm=module_intermediate_batchnorm, with_batchnorm=module_batchnorm,
                         with_cond=with_cond,
                         dropout=module_dropout,
                         num_extra_channels=self.num_extra_channels,
                         extra_channel_freq=self.extra_channel_freq,
                         with_input_proj=module_input_proj,
                         num_cond_maps=self.num_cond_maps,
                         kernel_size=module_kernel_size,
                         batchnorm_affine=module_batchnorm_affine,
                         num_layers=self.module_num_layers,
                         condition_method=condition_method,
                         debug_every=self.debug_every)
        else:
          mod = ConcatFiLMedResBlock(art, module_dim, with_residual=module_residual,
                         with_intermediate_batchnorm=module_intermediate_batchnorm, with_batchnorm=module_batchnorm,
                         with_cond=with_cond,
                         dropout=module_dropout,
                         num_extra_channels=self.num_extra_channels,
                         extra_channel_freq=self.extra_channel_freq,
                         with_input_proj=module_input_proj,
                         num_cond_maps=self.num_cond_maps,
                         kernel_size=module_kernel_size,
                         batchnorm_affine=module_batchnorm_affine,
                         num_layers=self.module_num_layers,
                         condition_method=condition_method,
                         debug_every=self.debug_every)

        if not self.share_module_weight_at_depth:
          ikey = str(i) + '-' + str(j) + '-' + str(art)
          self.function_modules[i] = mod
        else:
          ikey = str(j) + '-' + str(art)
          self.function_modules[j] = mod
        self.add_module(ikey, mod)

      # ensuring every node at the same depth has the same arity
      if self.share_module_weight_at_depth:
        if j not in self.depth_to_arity:
          self.depth_to_arity[j] = art
        else:
          if art != self.depth_to_arity[j]: raise Exception("Nodes at the same depth need to have the same arity.")

      if art == 0: return i+1
      idx = i+1
      dpt = j+1
      for _ in range(art):
        idx = generateModules(idx, dpt)

      return idx


    generateModules(0, 1)

    # Initialize output classifier
    self.classifier = build_classifier(module_dim + self.num_extra_channels, module_H, module_W,
                                       num_answers, classifier_fc_layers, classifier_proj_dim,
                                       classifier_downsample, with_batchnorm=classifier_batchnorm,
                                       dropout=classifier_dropout)

    init_modules(self.modules())

  def _forward_modules(self, feats, gammas, betas, cond_maps, batch_coords, save_activations, j, ijd):

    if j >= len(self.treeArities): return None, j+1
    fn_art = self.treeArities[j]
    fn_dept = ijd

    if not self.share_module_weight_at_depth:
      module = self.function_modules[j]
    else:
      module = self.function_modules[fn_dept]

    idx = j + 1
    dpt = ijd + 1

    if fn_art == 0:
      module_inputs = feats
    else:
      module_inputs = []
      while len(module_inputs) < fn_art:
        cur_input, idx = self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, save_activations, idx, dpt)
        module_inputs.append(cur_input)
      if len(module_inputs) == 1: module_inputs = module_inputs[0]

    bcoords = batch_coords
    if self.condition_method == 'concat':
      icond_maps = cond_maps[:,j,:]
      icond_maps = icond_maps.unsqueeze(2).unsqueeze(3).expand(icond_maps.size() + feats.size()[-2:])
      module_output = module(module_inputs, extra_channels=bcoords, cond_maps=icond_maps)
    else:
      igammas = gammas[:,j,:]
      ibetas = betas[:,j,:]
      module_output = module(module_inputs, igammas, ibetas, bcoords)

    if save_activations:
      self.module_outputs.append(module_output)

    return module_output, idx

  def forward(self, x, film, save_activations=False):
    # Initialize forward pass and externally viewable activations
    self.fwd_count += 1
    if save_activations:
      self.feats = None
      self.module_outputs = []
      self.cf_input = None

    if self.debug_every <= -2:
      pdb.set_trace()

    # Prepare Tfilm layers
    cond_maps = None
    gammas = None
    betas = None
    if self.condition_method == 'concat':
      # Use parameters usually used to condition via FiLM instead to condition via concatenation
      cond_params = film[:,:,:2*self.module_dim]
      cond_maps = cond_params #.unsqueeze(3).unsqueeze(4).expand(cond_params.size() + x.size()[-2:])
    else:
      gammas, betas = torch.split(film[:,:,:2*self.module_dim], self.module_dim, dim=-1)
      if not self.use_gamma:
        gammas = self.default_weight.expand_as(gammas)
      if not self.use_beta:
        betas = self.default_bias.expand_as(betas)

    # Propagate up image features CNN
    stem_batch_coords = None
    batch_coords = None
    if self.use_coords_freq > 0:
      stem_batch_coords = self.stem_coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.stem_coords.size())))
      batch_coords = self.coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.coords.size())))
    if self.stem_use_coords:
      x = torch.cat([x, stem_batch_coords], 1)
    feats = self.stem(x)
    if save_activations:
      self.feats = feats
    #N, _, H, W = feats.size()

    final_module_output, _ = self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, save_activations, 0, 1)

    # Store for future computation
    #if save_activations:
    #  self.module_outputs.append(layer_output)

    if self.debug_every <= -2:
      pdb.set_trace()

    # Run the final classifier over the resultant, post-modulated features.
    if self.use_coords_freq > 0:
      final_module_output = torch.cat([final_module_output, batch_coords], 1)
    if save_activations:
      self.cf_input = final_module_output
    out = self.classifier(final_module_output)

    if ((self.fwd_count % self.debug_every) == 0) or (self.debug_every <= -1):
      pdb.set_trace()
    return out

class ResidualBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_batchnorm=True):
    if out_dim is None:
      out_dim = in_dim
    super(ResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
    self.with_batchnorm = with_batchnorm
    if with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim)
      self.bn2 = nn.BatchNorm2d(out_dim)
    self.with_residual = with_residual
    if in_dim == out_dim or not with_residual:
      self.proj = None
    else:
      self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

  def forward(self, x):
    if self.with_batchnorm:
      out = F.relu(self.bn1(self.conv1(x)))
      out = self.bn2(self.conv2(out))
    else:
      out = self.conv2(F.relu(self.conv1(x)))
    res = x if self.proj is None else self.proj(x)
    if self.with_residual:
      out = F.relu(res + out)
    else:
      out = F.relu(out)
    return out


class ConcatBlock(nn.Module):
  def __init__(self, num_input, dim, with_residual=True, with_batchnorm=True):
    super(ConcatBlock, self).__init__()
    self.proj = nn.Conv2d(num_input * dim, dim, kernel_size=1, padding=0)
    self.res_block = ResidualBlock(dim, with_residual=with_residual,
                        with_batchnorm=with_batchnorm)

  def forward(self, x):
    out = torch.cat(x, 1) # Concatentate along depth
    out = F.relu(self.proj(out))
    out = self.res_block(out)
    return out