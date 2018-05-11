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
from torch.nn.init import kaiming_normal, kaiming_uniform, xavier_uniform, xavier_normal, constant

from vr.models.layers import init_modules, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem
import vr.programs

from vr.models.tfilmed_net import ConcatFiLMedResBlock

from vr.models.filmed_net import FiLM, FiLMedResBlock, coord_map

class NMNFiLMedNet2(nn.Module):
  def __init__(self, vocab, feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               stem_kernel_size=3,
               stem_subsample_layers=None,
               stem_stride=1,
               stem_padding=None,
               
               sharing_patterns=[0,1],

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
    super(NMNFiLMedNet2, self).__init__()

    num_answers = len(vocab['answer_idx_to_token'])

    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False
    
    self.sharing_patterns = sharing_patterns

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

    self.prepare_condition_pattern()

    self.extra_channel_freq = self.use_coords_freq
    #self.block = FiLMedResBlock
    self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
    self.fwd_count = 0
    self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
    if self.debug_every <= -1:
      self.print_verbose_every = 1

    # Initialize stem
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
    self.function_modules_num_inputs = {}
    self.vocab = vocab
    self.fn_str_2_filmId = {}
    for fn_str in vocab['program_token_to_idx']:
      num_inputs = vocab['program_token_arity'][fn_str]
      if fn_str == 'scene': num_inputs = 1
      self.function_modules_num_inputs[fn_str] = num_inputs
      
      if self.sharing_patterns[1] == 1:
        self.fn_str_2_filmId[str(num_inputs)] = num_inputs-1
      else:
        self.fn_str_2_filmId[fn_str] = len(self.fn_str_2_filmId)
      
      if num_inputs == 1:
        if self.sharing_patterns[0] == 1 and 1 in self.function_modules:
          mod = self.function_modules['1']
          stored_name = '1'
          newModule = False
        else:
          mod = FiLMedResBlock(module_dim, with_residual=module_residual,
                          with_intermediate_batchnorm=module_intermediate_batchnorm, with_batchnorm=module_batchnorm,
                          with_cond=self.condition_pattern,
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
          stored_name = '1' if self.sharing_patterns[0] == 1 else fn_str
          newModule = True
      elif num_inputs == 2:
        if self.sharing_patterns[0] == 1 and 2 in self.function_modules:
          mod = self.function_modules['2']
          stored_name = '2'
          newModule = False
        else:
          mod = ConcatFiLMedResBlock(2, module_dim, with_residual=module_residual,
                          with_intermediate_batchnorm=module_intermediate_batchnorm, with_batchnorm=module_batchnorm,
                          with_cond=self.condition_pattern,
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
          stored_name = '2' if self.sharing_patterns[0] == 1 else fn_str
          newModule = True
      
      if newModule:
        self.add_module(stored_name, mod)
        self.function_modules[stored_name] = mod
      
    # Define film coefficient parameters
    self.declare_film_coefficients()

    # Initialize output classifier
    self.classifier = build_classifier(module_dim + self.num_extra_channels, module_H, module_W,
                                       num_answers, classifier_fc_layers, classifier_proj_dim,
                                       classifier_downsample, with_batchnorm=classifier_batchnorm,
                                       dropout=classifier_dropout)

    init_modules(self.modules())
  
  def declare_film_coefficients(self):
    if self.condition_method == 'concat' or self.use_gamma:
      self.gammas = nn.Parameter(torch.Tensor(len(self.fn_str_2_filmId), self.module_dim))
      xavier_uniform(self.gammas)
    else:
      self.gammas = None
    if self.condition_method == 'concat' or self.use_beta:
      self.betas = nn.Parameter(torch.Tensor(len(self.fn_str_2_filmId), self.module_dim))
      xavier_uniform(self.betas)
    else:
      self.betas = None

  def prepare_condition_pattern(self):
    if len(self.condition_pattern) == 0:
      self.condition_pattern = [self.condition_method != 'concat'] * (2*self.module_num_layers)
    else:
      val1 = self.condition_pattern[0]
      val2 = self.condition_pattern[1] if len(self.condition_pattern) >= 2 else 0
      self.condition_pattern = [val1 > 0, val2 > 0] * self.module_num_layers

  def _forward_modules(self, feats, gammas, betas, cond_maps, batch_coords, program, save_activations, i, j):
    used_fn_j = True
    if j < program.size(1):
      fn_idx = program.data[i, j]
      fn_str = self.vocab['program_idx_to_token'][fn_idx]
    else:
      used_fn_j = False
      fn_str = 'scene'
    if fn_str == '<NULL>':
      used_fn_j = False
      fn_str = 'scene'
    elif fn_str == '<START>':
      used_fn_j = False
      return self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, program, save_activations, i, j + 1)
    if used_fn_j:
      self.used_fns[i, j] = 1

    j += 1
    
    num_inputs = self.function_modules_num_inputs[fn_str]
    if fn_str == 'scence': num_inputs = 1
    
    if self.sharing_patterns[1] == 1:
      query_id = str(num_inputs)
    else:
      query_id = fn_str
    assert query_id in self.fn_str_2_filmId
    filmId = self.fn_str_2_filmId[query_id]
    
    if self.sharing_patterns[0] == 1:
      query_id = str(num_inputs)
    else:
      query_id = fn_str
    assert query_id in self.function_modules
    module = self.function_modules[query_id]
    if fn_str == 'scene':
      module_inputs = feats[i:i+1]
    else:
      module_inputs = []
      
      while len(module_inputs) < num_inputs:
        cur_input, j = self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, program, save_activations, i, j)
        module_inputs.append(cur_input)
      if len(module_inputs) == 1: module_inputs = module_inputs[0]
    
    midx = filmId
    bcoords = batch_coords[i:i+1] if batch_coords is not None else None
    if self.condition_method == 'concat':
      icond_maps = cond_maps[:,midx,:,:,:]
      module_output = module(module_inputs, extra_channels=bcoords, cond_maps=icond_maps)
    else:
      igammas = gammas[:,midx,:]
      ibetas =  betas[:,midx,:]
      module_output = module(module_inputs, igammas, ibetas, bcoords)
    if save_activations:
      self.module_outputs.append(module_output)
    return module_output, j

  def forward(self, x, program, save_activations=False):
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
      cond_params = torch.cat([self.gammas, self.betas], 1).unsqueeze(0)
      cond_maps = cond_params.unsqueeze(3).unsqueeze(4).expand(cond_params.size() + x.size()[-2:])
    else:
      #gammas, betas = torch.split(film[:,:,:2*self.module_dim], self.module_dim, dim=-1)
      if not self.use_gamma:
        gammas = self.default_weight #.expand_as(gammas)
      else:
        gammas = self.gammas.unsqueeze(0)
      if not self.use_beta:
        betas = self.default_bias #.expand_as(betas)
      else:
        betas = self.betas.unsqueeze(0)

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
    N, _, H, W = feats.size()
    
    self.used_fns = torch.Tensor(program.size()).fill_(0)
    final_module_output = []
    for i in range(N):
      cur_output, _ = self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, program, save_activations, i, 0)
      final_module_output.append(cur_output)
    self.used_fns = self.used_fns.type_as(program.data).float()
    final_module_output = torch.cat(final_module_output, 0)

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
