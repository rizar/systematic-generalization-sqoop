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

from vr.models.filmed_net import FiLM

class TFiLMedNet(nn.Module):
  def __init__(self, vocab, feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               stem_kernel_size=3,
               stem_stride=1,
               stem_padding=None,
               num_modules=4,

               max_program_module_arity=2,
               max_program_tree_depth=5,

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
    super(TFiLMedNet, self).__init__()

    num_answers = len(vocab['answer_idx_to_token'])

    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False

    self.num_modules = num_modules

    self.max_program_module_arity = max_program_module_arity
    self.max_program_tree_depth = max_program_tree_depth

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
    self.first_condition_pattern = []

    self.prepare_condition_pattern()

    self.extra_channel_freq = self.use_coords_freq
    #self.block = FiLMedResBlock
    self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
    self.fwd_count = 0
    self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
    if self.debug_every <= -1:
      self.print_verbose_every = 1
    module_H = feature_dim[1] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
    module_W = feature_dim[2] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
    self.coords = coord_map((module_H, module_W))
    self.default_weight = Variable(torch.ones(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)
    self.default_bias = Variable(torch.zeros(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)

    # Initialize stem
    stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
    self.stem = build_stem(stem_feature_dim, module_dim,
                           num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding)

    # Initialize Tfilmed network body
    self.function_modules = {}
    self.function_modules_num_inputs = {}
    self.vocab = vocab
    for fn_str in vocab['program_token_to_idx']:
      num_inputs = vocab['program_token_arity'][fn_str]
      self.function_modules_num_inputs[fn_str] = num_inputs
    #for fn_num in range(self.num_modules):

    mod = TfilmedResBlock(module_dim, with_residual=module_residual,
                       with_intermediate_batchnorm=module_intermediate_batchnorm, with_batchnorm=module_batchnorm,
                       with_cond=self.first_condition_pattern,
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
    #mod = ResidualBlock(module_dim, with_residual=module_residual, with_batchnorm=module_batchnorm)
    self.add_module('0', mod)
    self.function_modules['0'] = mod

    for dep in range(self.max_program_tree_depth):
      for art in range(self.max_program_module_arity):
        with_cond = self.condition_pattern[dep][art]
        if art == 0:
          mod = TfilmedResBlock(module_dim, with_residual=module_residual,
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
          #mod = ResidualBlock(module_dim, with_residual=module_residual, with_batchnorm=module_batchnorm)
        else:
          mod = ConCatTfilmBlock(art+1, module_dim, with_residual=module_residual,
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
          #mod = ConcatBlock(art+1, module_dim, with_residual=module_residual, with_batchnorm=module_batchnorm)
        ikey = str(dep+1)+'-'+str(art+1)
        self.add_module(ikey, mod)
        self.function_modules[ikey] = mod

    # Initialize output classifier
    self.classifier = build_classifier(module_dim + self.num_extra_channels, module_H, module_W,
                                       num_answers, classifier_fc_layers, classifier_proj_dim,
                                       classifier_downsample, with_batchnorm=classifier_batchnorm,
                                       dropout=classifier_dropout)

    init_modules(self.modules())

  def prepare_condition_pattern(self):
    if len(self.condition_pattern) == 0:
      self.first_condition_pattern = [self.condition_method != 'concat'] * (2*self.module_num_layers)
      outCond = []
      for i in range(self.max_program_tree_depth):
        idepth = []
        for j in range(self.max_program_module_arity):
          ijarity = [self.condition_method != 'concat'] * (2*self.module_num_layers)
          idepth.append(ijarity)
        outCond.append(idepth)
      self.condition_pattern = outCond
    else:
      ijc = 0
      if len(self.condition_pattern) > ijc: val1 = self.condition_pattern[ijc]
      else: val1 = self.condition_pattern[-1]
      ijc += 1
      if len(self.condition_pattern) > ijc: val2 = self.condition_pattern[ijc]
      else: val2 = self.condition_pattern[-1]
      self.first_condition_pattern = [val1 > 0, val2 > 0] * self.module_num_layers
      outCond = []
      for i in range(self.max_program_tree_depth):
        idepth = []
        for j in range(self.max_program_module_arity):
          ijc += 1
          if len(self.condition_pattern) > ijc: val1 = self.condition_pattern[ijc]
          else: val1 = self.condition_pattern[-1]
          ijc += 1
          if len(self.condition_pattern) > ijc: val2 = self.condition_pattern[ijc]
          else: val2 = self.condition_pattern[-1]
          idepth.append([val1 > 0, val2 > 0] * self.module_num_layers)
        outCond.append(idepth)
      self.condition_pattern = outCond

  def _forward_modules(self, feats, gammas, betas, cond_maps, batch_coords, program, program_arity, save_activations, i, j, ijd):
    #used_fn_j = True
    if j < program.size(1):
      fn_idx = program.data[i, j]
      fn_str = self.vocab['program_idx_to_token'][fn_idx]
      fn_art = program_arity.data[i,j]
      fn_dept = ijd
    else:
      #used_fn_j = False
      #fn_str = 'scene'
      fn_art = -1
      fn_dept = -1

    if fn_str == '<START>':
      #used_fn_j = False
      return self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, program, program_arity, save_activations, i, j + 1, ijd)
    if fn_art < 0 or fn_str == '<END>': return feats[i:i+1], j+1

    #if used_fn_j:
    #  self.used_fns[i, j] = 1
    j += 1

    if fn_art == 0:
      module = self.function_modules['0']
      module_inputs = feats[i:i+1]

    else:
      module_key = str(fn_dept) + '-' + str(fn_art)
      if module_key not in self.function_modules:
        print('Cannot find module: ' + module_key)
        exit()
      module = self.function_modules[module_key]

      module_inputs = []
      while len(module_inputs) < fn_art:
        cur_input, j = self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, program, program_arity, save_activations, i, j, ijd+1)
        module_inputs.append(cur_input)
      if len(module_inputs) == 1: module_inputs = module_inputs[0]

    midx = 0 if fn_art == 0 else (fn_dept-1)*self.max_program_module_arity+fn_art
    bcoords = batch_coords[i:i+1] if batch_coords is not None else None
    if self.condition_method == 'concat':
      icond_maps = cond_maps[i:i+1,0,:] if fn_art == 0 else cond_maps[i:i+1,midx,:]
      icond_maps = icond_maps.unsqueeze(2).unsqueeze(3).expand(icond_maps.size() + feats.size()[-2:])
      module_output = module(module_inputs, extra_channels=bcoords, cond_maps=icond_maps)
    else:
      igammas = gammas[i:i+1,0,:] if fn_art == 0 else gammas[i:i+1,midx,:]
      ibetas = betas[i:i+1,0,:] if fn_art == 0 else betas[i:i+1,midx,:]
      module_output = module(module_inputs, igammas, ibetas, bcoords)
    if save_activations:
      self.module_outputs.append(module_output)
    return module_output, j

  def computeArity(self, program):
    progDat = program.data
    arity = numpy.zeros(progDat.shape).astype('int32')
    for i in range(progDat.shape[0]):
      ended = False
      for j in range(progDat.shape[1]):
        fn_idx = progDat[i, j]
        fn_str = self.vocab['program_idx_to_token'][fn_idx]
        if fn_str == '<END>': ended = True
        value = self.function_modules_num_inputs[fn_str]
        if ended or fn_str == '<START>': value = -1
        if value >= 0 and fn_str == '<NULL>': value = 0
        arity[i,j] = value
    return arity

  def forward(self, x, film, program, save_activations=False):
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
    batch_coords = None
    if self.use_coords_freq > 0:
      batch_coords = self.coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.coords.size())))
    if self.stem_use_coords:
      x = torch.cat([x, batch_coords], 1)
    feats = self.stem(x)
    if save_activations:
      self.feats = feats
    N, _, H, W = feats.size()

    program_arity = self.computeArity(program)
    final_module_output = []
    for i in range(N):
      cur_output, _ = self._forward_modules(feats, gammas, betas, cond_maps, batch_coords, program, program_arity, save_activations, i, 0, 1)
      final_module_output.append(cur_output)
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

class ConCatTfilmBlock(nn.Module):
  def __init__(self, num_input, in_dim, out_dim=None, with_residual=True, with_intermediate_batchnorm=False, with_batchnorm=True,
               with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
               with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
               num_layers=1, condition_method='bn-film', debug_every=float('inf')):
      super(ConCatTfilmBlock, self).__init__()
      self.proj = nn.Conv2d(num_input * in_dim, in_dim, kernel_size=1, padding=0)
      self.tfilmedResBlock = TfilmedResBlock(in_dim=in_dim, out_dim=out_dim, with_residual=with_residual,
               with_intermediate_batchnorm=with_intermediate_batchnorm, with_batchnorm=with_batchnorm,
               with_cond=with_cond, dropout=dropout, num_extra_channels=num_extra_channels, extra_channel_freq=extra_channel_freq,
               with_input_proj=with_input_proj, num_cond_maps=num_cond_maps, kernel_size=kernel_size, batchnorm_affine=batchnorm_affine,
               num_layers=num_layers, condition_method=condition_method, debug_every=debug_every)

  def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):
    out = torch.cat(x, 1) # Concatentate along depth
    out = F.relu(self.proj(out))
    out = self.tfilmedResBlock(out, gammas=gammas, betas=betas, extra_channels=extra_channels, cond_maps=cond_maps)
    return out

class TfilmedResBlock(nn.Module):
  def __init__(self, in_dim, out_dim=None, with_residual=True, with_intermediate_batchnorm=False, with_batchnorm=True,
               with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
               with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
               num_layers=1, condition_method='bn-film', debug_every=float('inf')):
    if out_dim is None:
      out_dim = in_dim
    super(TfilmedResBlock, self).__init__()
    self.with_residual = with_residual
    self.with_intermediate_batchnorm = with_intermediate_batchnorm
    self.with_batchnorm = with_batchnorm
    self.with_cond = with_cond
    self.dropout = dropout
    self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
    self.with_input_proj = with_input_proj  # Kernel size of input projection
    self.num_cond_maps = num_cond_maps
    self.kernel_size = kernel_size
    self.batchnorm_affine = batchnorm_affine
    self.num_layers = num_layers
    self.condition_method = condition_method
    self.debug_every = debug_every

    if self.with_input_proj % 2 == 0:
      raise(NotImplementedError)
    if self.kernel_size % 2 == 0:
      raise(NotImplementedError)
    if self.num_layers >= 2:
      raise(NotImplementedError)

    if self.condition_method == 'block-input-film' and self.with_cond[0]:
      self.film = FiLM()
    if self.with_input_proj:
      self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                  in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

    self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps +
                           (num_extra_channels if self.extra_channel_freq >= 2 else 0),
                            out_dim, kernel_size=self.kernel_size,
                            padding=self.kernel_size // 2)
    if self.condition_method == 'conv-film' and self.with_cond[0]:
      self.film = FiLM()
    if self.with_intermediate_batchnorm:
      self.bn0 = nn.BatchNorm2d(in_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
    if self.with_batchnorm:
      self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
    if self.condition_method == 'bn-film' and self.with_cond[0]:
      self.film = FiLM()
    if dropout > 0:
      self.drop = nn.Dropout2d(p=self.dropout)
    if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
         and self.with_cond[0]):
      self.film = FiLM()

    init_modules(self.modules())

  def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):
    if self.debug_every <= -2:
      pdb.set_trace()

    if self.condition_method == 'block-input-film' and self.with_cond[0]:
      x = self.film(x, gammas, betas)

    # ResBlock input projection
    if self.with_input_proj:
      if extra_channels is not None and self.extra_channel_freq >= 1:
        x = torch.cat([x, extra_channels], 1)
      x = self.input_proj(x)
      if self.with_intermediate_batchnorm:
        x = self.bn0(x)
      x = F.relu(x)
    out = x

    # ResBlock body
    if cond_maps is not None:
      out = torch.cat([out, cond_maps], 1)
    if extra_channels is not None and self.extra_channel_freq >= 2:
      out = torch.cat([out, extra_channels], 1)
    out = self.conv1(out)
    if self.condition_method == 'conv-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    if self.with_batchnorm:
      out = self.bn1(out)
    if self.condition_method == 'bn-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    if self.dropout > 0:
      out = self.drop(out)
    out = F.relu(out)
    if self.condition_method == 'relu-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)

    # ResBlock remainder
    if self.with_residual:
      out = x + out
    if self.condition_method == 'block-output-film' and self.with_cond[0]:
      out = self.film(out, gammas, betas)
    return out


def coord_map(shape, start=-1, end=1):
  """
  Gives, a 2d shape tuple, returns two mxn coordinate maps,
  Ranging min-max in the x and y directions, respectively.
  """
  m, n = shape
  x_coord_row = torch.linspace(start, end, steps=n).type(torch.cuda.FloatTensor)
  y_coord_row = torch.linspace(start, end, steps=m).type(torch.cuda.FloatTensor)
  x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
  y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
  return Variable(torch.cat([x_coords, y_coords], 0))
