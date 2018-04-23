#!/usr/bin/env python3

import numpy as np
import math
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models
import math
from torch.nn.init import kaiming_normal, kaiming_uniform, xavier_uniform, xavier_normal, constant

from vr.models.layers import build_classifier, build_stem
import vr.programs

class MAC(nn.Module):
  """Implementation of the Compositional Attention Networks from: https://openreview.net/pdf?id=S1Euwz-Rb"""
  def __init__(self, vocab, feature_dim,
               stem_num_layers,
               stem_batchnorm,
               stem_kernel_size,
               stem_subsample_layers,
               stem_stride,
               stem_padding,
               num_modules,
               module_dim,
               question_embedding_dropout,
               stem_dropout,
               memory_dropout,
               read_dropout,
               use_prior_control_in_control_unit,
               use_self_attention,
               use_memory_gate,
               classifier_fc_layers,
               classifier_batchnorm,
               classifier_dropout,
               use_coords,
               debug_every=float('inf'),
               print_verbose_every=float('inf'),
               verbose=True,
               ):
    super(MAC, self).__init__()

    num_answers = len(vocab['answer_idx_to_token'])

    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False

    self.num_modules = num_modules

    self.question_embedding_dropout = question_embedding_dropout
    self.memory_dropout = memory_dropout
    self.read_dropout = read_dropout

    self.module_dim = module_dim

    self.use_self_attention = use_self_attention == 1
    self.use_memory_gate = use_memory_gate == 1

    self.use_coords_freq = use_coords
    self.debug_every = debug_every
    self.print_verbose_every = print_verbose_every

    # Initialize helper variables
    self.stem_use_coords = self.use_coords_freq
    self.extra_channel_freq = self.use_coords_freq

    self.fwd_count = 0
    self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
    if self.debug_every <= -1:
      self.print_verbose_every = 1

    # Initialize stem
    stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
    self.stem = build_stem(stem_feature_dim, module_dim,
                           num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding,
                           subsample_layers=stem_subsample_layers, acceptEvenKernel=True)


    #Define units
    self.inputUnits = []
    for i in range(self.num_modules):
      mod = InputUnit(module_dim)
      self.add_module('InputUnit' + str(i+1), mod)
      self.inputUnits.append(mod)

    mod = ControlUnit(module_dim, use_prior_control_in_control_unit=use_prior_control_in_control_unit)
    self.add_module('ControlUnit', mod)
    self.controlUnit = mod

    mod = ReadUnit(module_dim, self.read_dropout)
    self.add_module('ReadUnit', mod)
    self.readUnit = mod

    mod = WriteUnit(module_dim,
                    use_self_attention=self.use_self_attention,
                    use_memory_gate=self.use_memory_gate)
    self.add_module('WriteUnit', mod)
    self.writeUnit = mod

    #parameters for initial memory and control vectors
    self.init_memory = nn.Parameter(torch.randn(module_dim).cuda())

    #first transformation of question embeddings
    self.init_question_transformer = nn.Linear(self.module_dim, self.module_dim)
    self.init_question_non_linear = nn.Tanh()

    self.vocab = vocab

    self.question_embedding_dropout_module = nn.Dropout(p=self.question_embedding_dropout)

    # Initialize output classifier
    self.classifier = OutputUnit(module_dim, classifier_fc_layers, num_answers,
                                 with_batchnorm=classifier_batchnorm, dropout=classifier_dropout)

    init_modules(self.modules())

  def forward(self, x, ques, isTest=False, save_activations=False):
    # Initialize forward pass and externally viewable activations
    self.fwd_count += 1
    if save_activations:
      self.feats = None
      self.control_outputs = []
      self.memory_outputs = []
      self.cf_input = None

    q_context, q_rep, q_mask = ques

    original_q_rep = q_rep

    q_rep = self.question_embedding_dropout_module(q_rep)

    init_control = q_rep

    q_rep = self.init_question_non_linear(self.init_question_transformer(q_rep))

    stem_batch_coords = None
    if self.use_coords_freq > 0:
      stem_coords = coord_map((x.size(2), x.size(3)))
      stem_batch_coords = stem_coords.unsqueeze(0).expand(
          torch.Size((x.size(0), *stem_coords.size())))
    if self.stem_use_coords:
      x = torch.cat([x, stem_batch_coords], 1)
    feats = self.stem(x)
    if save_activations:
      self.feats = feats
    N, _, H, W = feats.size()

    control_storage = Variable(torch.zeros(N, 1+self.num_modules, self.module_dim)).type(torch.cuda.FloatTensor)
    memory_storage = Variable(torch.zeros(N, 1+self.num_modules, self.module_dim)).type(torch.cuda.FloatTensor)

    control_storage[:,0,:] = init_control
    memory_storage[:,0,:] = self.init_memory.expand(N, self.module_dim)

    if self.memory_dropout > 0. and not isTest:
      dropout_mask_memory = Variable(torch.Tensor(N, self.module_dim).fill_(
        self.memory_dropout).bernoulli_()).type(torch.cuda.FloatTensor)
    else:
      dropout_mask_memory = None

    for fn_num in range(self.num_modules):
      inputUnit = self.inputUnits[fn_num] if isinstance(self.inputUnits, list) else self.inputUnits
      controlUnit = self.controlUnit[fn_num] if isinstance(self.controlUnit, list) else self.controlUnit
      readUnit = self.readUnit[fn_num] if isinstance(self.readUnit, list) else self.readUnit
      writeUnit = self.writeUnit[fn_num] if isinstance(self.writeUnit, list) else self.writeUnit

      #compute question representation specific to this cell
      q_rep_i = inputUnit(q_rep) # N x d

      #compute control at the current step
      control_i = controlUnit(control_storage[:,fn_num,:], q_rep_i, q_context, q_mask)
      if save_activations:
        self.control_outputs.append(control_i)
      control_updated = control_storage.clone()
      control_updated[:,(fn_num+1),:] = control_updated[:,(fn_num+1),:] + control_i
      control_storage = control_updated

      #compute read at the current step
      read_i = readUnit(memory_storage[:,fn_num,:], control_updated[:,(fn_num+1),:], feats,
                        memory_dropout=self.memory_dropout, dropout_mask_memory=dropout_mask_memory, isTest=isTest)

      #compute write memeory at the current step
      memory_i = writeUnit(memory_storage, control_storage, read_i, fn_num+1)

      if save_activations:
        self.memory_outputs.append(memory_i)

      if fn_num == (self.num_modules - 1):
        final_module_output = memory_i
      else:
        memory_updated = memory_storage.clone()
        memory_updated[:,(fn_num+1),:] = memory_updated[:,(fn_num+1),:] + memory_i
        memory_storage = memory_updated

    if save_activations:
      self.cf_input = final_module_output

    out = self.classifier(final_module_output, original_q_rep, isTest=isTest)

    return out

class OutputUnit(nn.Module):
  def __init__(self, module_dim, hidden_units, num_outputs, with_batchnorm=False, dropout=0.0):
    super(OutputUnit, self).__init__()

    self.dropout = dropout

    self.question_transformer = nn.Linear(module_dim, module_dim)

    input_dim = 2*module_dim
    hidden_units = [input_dim] + [h for h in hidden_units] + [num_outputs]
    self.linears = []
    self.batchnorms = []
    for i, (nin, nout) in enumerate(zip(hidden_units, hidden_units[1:])):
      mod = nn.Linear(nin, nout)
      self.add_module('MAC_LinearFC' + str(i), mod)
      self.linears.append(mod)
      mod = nn.BatchNorm1d(nin) if with_batchnorm else None
      if mod is not None: self.add_module('MAC_BatchNormFC' + str(i), mod)
      self.batchnorms.append(mod)

    self.non_linear = nn.ReLU()
    self.dropout_module = nn.Dropout(p=self.dropout)

    init_modules(self.modules())

  def forward(self, final_memory, original_q_rep, isTest=False):

    transformed_question = self.question_transformer(original_q_rep)
    features = torch.cat([final_memory, transformed_question], 1)

    for i, (linear, batchnorm) in enumerate(zip(self.linears, self.batchnorms)):
      if batchnorm is not None:
        features = batchnorm(features)
      features = self.dropout_module(features)
      features = linear(features)
      if i < len(self.linears) - 1:
        features = self.non_linear(features)

    return features

class WriteUnit(nn.Module):
  def __init__(self, common_dim, use_self_attention=False, use_memory_gate=False):
    super(WriteUnit, self).__init__()
    self.common_dim = common_dim
    self.use_self_attention = use_self_attention
    self.use_memory_gate = use_memory_gate

    self.control_memory_transfomer = nn.Linear(2 * common_dim, common_dim) #Eq (w1)

    if use_self_attention:
      self.current_control_transformer = nn.Linear(common_dim, common_dim)

      self.control_transformer = nn.Linear(common_dim, 1) #Eq (w2.1)
      self.acc_memory_transformer = nn.Linear(common_dim, common_dim, bias=False)
      self.pre_memory_transformer = nn.Linear(common_dim, common_dim) #Eq (w2.3)

    if use_memory_gate:
      self.gated_control_transformer = nn.Linear(common_dim, 1) #Eq (w3.1)
      self.non_linear = nn.Sigmoid()

    init_modules(self.modules())

  def forward(self, memories, controls, current_read, idx):
    #memories (N x num_cell x d), controls (N x num_cell x d), current_read (N x d), idx (int starting from 1)

    #Eq (w1)
    res_memory = self.control_memory_transfomer( torch.cat([current_read, memories[:,idx-1,:]], 1) ) #N x d

    if self.use_self_attention:
      current_control = controls[:,idx,:] # N x d
      current_control = self.current_control_transformer(current_control) # N x d in code
      if idx > 1:
        #Eq (w2.1)
        previous_controls = controls[:,1:idx,:] # N x (idx-1) x d
        cscores = previous_controls * current_control.unsqueeze(1) # N x (idx-1) x d
        cscores = self.control_transformer(cscores).squeeze(2) # N x (idx -1)
        cscores = torch.exp(cscores - cscores.max(1, keepdim=True)[0]) # N x (idx -1)
        cscores = cscores / cscores.sum(1, keepdim=True) # N x (idx -1)

        #Eq (w2.2)
        previous_memories = memories[:,1:idx,:] #N x (idx-1) x d
        acc_memory = (previous_memories * cscores.unsqueeze(2)).sum(1) # N x d

        #Eq (w2.3)
        res_memory = self.acc_memory_transformer(acc_memory) + self.pre_memory_transformer(res_memory)
      else:
        #Eq (w2.3) as there is no m_i^{sa} in this case
        res_memory = self.pre_memory_transformer(res_memory)

    if self.use_memory_gate:
      #Eq (w3.1)
      gated_control = self.gated_control_transformer(controls[:,idx,:]) #N x 1

      #Eq (w3.2)
      gated_control = self.non_linear(gated_control-1)
      res_memory = memories[:,idx-1,:] * gated_control + res_memory * (1. - gated_control)

    return res_memory


class ReadUnit(nn.Module):
  def __init__(self, common_dim, read_dropout=0.):
    super(ReadUnit, self).__init__()
    self.common_dim = common_dim
    self.read_dropout = read_dropout

    #Eq (r1)
    self.pre_memory_transformer = nn.Linear(common_dim, common_dim)
    self.image_element_transformer = nn.Linear(common_dim, common_dim)

    #Eq (r2)
    self.intermediate_transformer = nn.Linear(2 * common_dim, common_dim)

    #Eq (r3.1)
    self.read_attention_transformer = nn.Linear(common_dim, 1)

    self.non_linear = nn.ReLU()
    self.read_dropout_module = nn.Dropout(p=self.read_dropout)

    init_modules(self.modules())

  def forward(self, pre_memory, current_control, image,
              memory_dropout=0., dropout_mask_memory=None, isTest=False):

    #pre_memory(Nxd), current_control(Nxd), image(NxdxHxW)

    image = image.transpose(1,2).transpose(2,3) #NXHxWxd
    trans_image = image

    if not isTest and memory_dropout > 0.:
      assert dropout_mask_memory is not None
      pre_memory = (pre_memory / (1. - memory_dropout)) * dropout_mask_memory

    pre_memory = self.read_dropout_module(pre_memory)
    trans_image = self.read_dropout_module(trans_image)

    #Eq (r1)
    trans_pre_memory = self.pre_memory_transformer(pre_memory) #Nxd
    trans_image = self.image_element_transformer(trans_image) #NxHxWxd image
    trans_pre_memory = trans_pre_memory.unsqueeze(1).unsqueeze(2).expand(trans_image.size()) #NxHxWxd
    intermediate = trans_pre_memory * trans_image #NxHxWxd

    #Eq (r2)
    #trans_intermediate = self.intermediate_transformer(torch.cat([intermediate, image], 3)) #NxHxWxd
    trans_intermediate = self.intermediate_transformer(torch.cat([intermediate, trans_image], 3)) #NxHxWxd
    trans_intermediate = self.non_linear(trans_intermediate)

    #Eq (r3.1)
    trans_current_control = current_control.unsqueeze(1).unsqueeze(2).expand(trans_intermediate.size()) #NxHxWxd
    intermediate_score = trans_current_control * trans_intermediate

    intermediate_score = self.non_linear(intermediate_score)

    intermediate_score = self.read_dropout_module(intermediate_score)

    scores = self.read_attention_transformer(intermediate_score).squeeze(3) #NxHxWx1 -> NxHxW

    #Eq (r3.2): softmax
    rscores = scores.view(scores.shape[0], -1) #N x (H*W)
    rscores = torch.exp(rscores - rscores.max(1, keepdim=True)[0])
    rscores = rscores / rscores.sum(1, keepdim=True)
    scores = rscores.view(scores.shape) #NxHxW

    #Eq (r3.3)
    readrep = image * scores.unsqueeze(3)
    readrep = readrep.view(readrep.shape[0], -1, readrep.shape[-1]) #N x (H*W) x d
    readrep = readrep.sum(1) #N x d

    return readrep

class ControlUnit(nn.Module):
  def __init__(self, common_dim, use_prior_control_in_control_unit=False):
    super(ControlUnit, self).__init__()
    self.common_dim = common_dim
    self.use_prior_control_in_control_unit = use_prior_control_in_control_unit

    if use_prior_control_in_control_unit:
      self.control_question_transformer = nn.Linear(2 * common_dim, common_dim) #Eq (c1)

    self.score_transformer = nn.Linear(common_dim, 1) # Eq (c2.1)

    init_modules(self.modules())

  def forward(self, pre_control, question, context, mask):

    #pre_control (Nxd), question (Nxd), context(NxLxd), mask(NxL)

    #Eq (c1)
    if self.use_prior_control_in_control_unit:
      control_question = self.control_question_transformer(torch.cat([pre_control, question], 1)) # N x d
    else:
      control_question = question # N x d

    #Eq (c2.1)
    scores = self.score_transformer(context * control_question.unsqueeze(1)).squeeze(2)  #NxLxd -> NxLx1 -> NxL

    #Eq (c2.2) : softmax
    scores = torch.exp(scores - scores.max(1, keepdim=True)[0]) * mask #mask help to eliminate null tokens
    scores = scores / scores.sum(1, keepdim=True) #NxL

    #Eq (c2.3)
    control = (context * scores.unsqueeze(2)).sum(1) #Nxd

    return control

class InputUnit(nn.Module):
  def __init__(self, common_dim):
    super(InputUnit, self).__init__()
    self.common_dim = common_dim
    self.question_transformer = nn.Linear(common_dim, common_dim)

    init_modules(self.modules())

  def forward(self, question):
    return self.question_transformer(question) #Section 2.1

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

def sincos_coord_map(shape, p_h=64., p_w=64.):
  m, n = shape
  x_coords = torch.zeros(m,n)
  y_coords = torch.zeros(m,n)

  for i in range(m):
    for j in range(n):
      icoord = i if i % 2 == 0 else i-1
      jcoord = j if j % 2 == 0 else j-1
      x_coords[i, j] = math.sin(1.0 * i / (10000. ** (1.0 * jcoord / p_h)))
      y_coords[i, j] = math.cos(1.0 * j / (10000. ** (1.0 * icoord / p_w)))

  x_coords = x_coords.type(torch.cuda.FloatTensor).unsqueeze(0)
  y_coords = y_coords.type(torch.cuda.FloatTensor).unsqueeze(0)

  return Variable(torch.cat([x_coords, y_coords], 0))


def init_modules(modules, init='uniform'):
  if init.lower() == 'normal':
    init_params = xavier_normal
  elif init.lower() == 'uniform':
    init_params = xavier_uniform
  else:
    return
  for m in modules:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      init_params(m.weight)
      if m.bias is not None: constant(m.bias, 0.)
