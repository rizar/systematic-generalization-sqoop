#!/usr/bin/env python3

import math
import pprint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

from vr.models.layers import init_modules #, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem
import vr.programs

class MAC(nn.Module):
  """Implementation of the Compositional Attention Networks from: https://openreview.net/pdf?id=S1Euwz-Rb"""
  def __init__(self, vocab, feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               stem_kernel_size=3,
               stem_stride=1,
               stem_padding=None,
               num_modules=12,
               
               #module_num_layers=1,
               module_dim=128,
               #module_residual=True,
               #module_intermediate_batchnorm=False,
               #module_batchnorm=False,
               #module_batchnorm_affine=False,
               #module_dropout=0,
               #module_input_proj=1,
               #module_kernel_size=3,
               
               #the boolean variables to decide wehther to share params betweens the MAC cells in the model for
               #the input units, control units, read units and write units respectively
               sharing_params_patterns=(0,1,0,0),
               use_self_attention=1, 
               use_memory_gate=1,
               
               #classifier_proj_dim=512,
               #classifier_downsample='maxpool2',
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               #condition_method='bn-film',
               #condition_pattern=[],
               #use_gamma=True,
               #use_beta=True,
               use_coords=1,
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
    #self.module_num_layers = module_num_layers
    #self.module_batchnorm = module_batchnorm
    self.module_dim = module_dim
    #self.condition_method = condition_method
    #self.use_gamma = use_gamma
    #self.use_beta = use_beta
    
    self.sharing_params_patterns = [True if p == 1 else 0 for p in sharing_params_patterns]
    self.use_self_attention = use_self_attention == 1
    self.use_memory_gate = use_memory_gate == 1
    
    self.use_coords_freq = use_coords
    self.debug_every = debug_every
    self.print_verbose_every = print_verbose_every

    # Initialize helper variables
    self.stem_use_coords = (stem_stride == 1) and (self.use_coords_freq > 0)
    
    '''
    self.condition_pattern = condition_pattern
    if len(condition_pattern) == 0:
      self.condition_pattern = []
      for i in range(self.module_num_layers * self.num_modules):
        self.condition_pattern.append(self.condition_method != 'concat')
    else:
      self.condition_pattern = [i > 0 for i in self.condition_pattern]
    '''
      
    self.extra_channel_freq = self.use_coords_freq
    
    '''
    self.block = FiLMedResBlock
    self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
    '''
    
    self.fwd_count = 0
    self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
    if self.debug_every <= -1:
      self.print_verbose_every = 1
    module_H = feature_dim[1] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
    module_W = feature_dim[2] // (stem_stride ** stem_num_layers)  # Rough calc: work for main cases
    self.coords = coord_map((module_H, module_W))
    
    #self.default_weight = Variable(torch.ones(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)
    #self.default_bias = Variable(torch.zeros(1, 1, self.module_dim)).type(torch.cuda.FloatTensor)

    # Initialize stem
    stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
    self.stem = build_stem(stem_feature_dim, module_dim,
                           num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding)
    
    
    #Define units
    if self.sharing_params_patterns[0]:
      mod = InputUnit(module_dim)
      self.add_module('InputUnit', mod)
      self.InputUnits = mod
    else:
      self.InputUnits = []
      for i in range(self.num_modules):
        mod = InputUnit(module_dim)
        self.add_module('InputUnit' + str(i+1), mod)
        self.InputUnits.append(mod)
    
    if self.sharing_params_patterns[1]:
      mod = ControlUnit(module_dim)
      self.add_module('ControlUnit', mod)
      self.ControlUnits = mod
    else:
      self.ControlUnits = []
      for i in range(self.num_modules):
        mod = ControlUnit(module_dim)
        self.add_module('ControlUnit' + str(i+1), mod)
        self.ControlUnits.append(mod)
    
    if self.sharing_params_patterns[2]:
      mod = ReadUnit(module_dim)
      self.add_module('ReadUnit', mod)
      self.ReadUnits = mod
    else:
      self.ReadUnits = []
      for i in range(self.num_modules):
        mod = ReadUnit(module_dim)
        self.add_module('ReadUnit' + str(i+1), mod)
        self.ReadUnits.append(mod)
    
    if self.sharing_params_patterns[3]:
      mod = WriteUnit(module_dim, 
                      use_self_attention=self.use_self_attention,
                      use_memory_gate=self.use_memory_gate)
      self.add_module('WriteUnit', mod)
      self.WriteUnits = mod
    else:
      self.WriteUnits = []
      for i in range(self.num_modules):
        mod = WriteUnit(module_dim, 
                        use_self_attention=self.use_self_attention,
                        use_memory_gate=self.use_memory_gate)
        self.add_module('WriteUnit' + str(i+1), mod)
        self.WriteUnits.append(mod)
    
    #parameters for initial memory and control vectors
    self.init_memory = nn.Parameter(torch.zeros(module_dim).cuda())
    self.init_control = nn.Parameter(torch.zeros(module_dim).cuda())

    self.vocab = vocab

    # Initialize output classifier
    self.classifier = OutputUnit(3*module_dim, classifier_fc_layers, num_answers,
                                 with_batchnorm=classifier_batchnorm, dropout=classifier_dropout)

    init_modules(self.modules())

  def forward(self, x, ques, save_activations=False):
    # Initialize forward pass and externally viewable activations
    self.fwd_count += 1
    if save_activations:
      self.feats = None
      self.control_outputs = []
      self.memory_outputs = []
      self.cf_input = None

    '''
    if self.debug_every <= -2:
      pdb.set_trace()
    '''
    
    q_context, q_rep, q_mask = ques

    batch_coords = None
    if self.use_coords_freq > 0:
      batch_coords = self.coords.unsqueeze(0).expand(torch.Size((x.size(0), *self.coords.size())))
    if self.stem_use_coords:
      x = torch.cat([x, batch_coords], 1)
    feats = self.stem(x)
    if save_activations:
      self.feats = feats
    N, _, H, W = feats.size()
    
    control_storage = Variable(torch.zeros(N, 1+self.num_modules, self.module_dim)).type(torch.cuda.FloatTensor)
    memory_storage = Variable(torch.zeros(N, 1+self.num_modules, self.module_dim)).type(torch.cuda.FloatTensor)
    
    control_storage[:,0,:] = self.init_control.expand(N, self.module_dim)
    memory_storage[:,0,:] = self.init_memory.expand(N, self.module_dim)
    
    for fn_num in range(self.num_modules):
      inputUnit = self.InputUnits[fn_num] if isinstance(self.InputUnits, list) else self.InputUnits
      controlUnit = self.ControlUnits[fn_num] if isinstance(self.ControlUnits, list) else self.ControlUnits
      readUnit = self.ReadUnits[fn_num] if isinstance(self.ReadUnits, list) else self.ReadUnits
      writeUnit = self.WriteUnits[fn_num] if isinstance(self.WriteUnits, list) else self.WriteUnits
      
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
      read_i = readUnit(memory_storage[:,fn_num,:], control_updated[:,(fn_num+1),:], feats)
      
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

    # Run the final classifier over the resultant, post-modulated features.
    '''
    if self.use_coords_freq > 0:
      final_module_output = torch.cat([final_module_output, batch_coords], 1)
    '''
    final_module_output = torch.cat([q_rep, final_module_output], 1)
    
    if save_activations:
      self.cf_input = final_module_output
    out = self.classifier(final_module_output)
    
    '''
    if ((self.fwd_count % self.debug_every) == 0) or (self.debug_every <= -1):
      pdb.set_trace()
    '''
    return out

class OutputUnit(nn.Module):
  def __init__(self, input_dim, hidden_units, num_outputs, with_batchnorm=False, dropout=0.0):
    super(OutputUnit, self).__init__()
    hidden_units = [input_dim] + [h for h in hidden_units]
    
    layers = []
    for nin, nout in zip(hidden_units, hidden_units[1:]):
      layers.append(nn.Linear(nin, nout))
      if with_batchnorm:
        layers.append(nn.BatchNorm1d(nout))
      layers.append(nn.ELU(inplace=True)) #ReLU
      if dropout > 0:
        layers.append(nn.Dropout(p=dropout))
    
    layers.append(nn.Linear(hidden_units[-1], num_outputs))
    
    self.layers = nn.Sequential(*layers)
    
    init_modules(self.modules())
  
  def forward(self, x):
    return self.layers(x)

class WriteUnit(nn.Module):
  def __init__(self, common_dim, use_self_attention=False, use_memory_gate=False):
    super(WriteUnit, self).__init__()
    self.common_dim = common_dim
    self.use_self_attention = use_self_attention
    self.use_memory_gate = use_memory_gate
    
    self.control_memory_transfomer = nn.Linear(2 * common_dim, common_dim) #Eq (w1)
    
    if use_self_attention:
      self.control_transformer = nn.Linear(common_dim, 1) #Eq (w2.1)
      self.acc_memory_transformer = nn.Linear(common_dim, common_dim, bias=False)
      self.pre_memory_transformer = nn.Linear(common_dim, common_dim) #Eq (w2.3)
    
    if use_memory_gate:
      self.gated_control_transformer = nn.Linear(common_dim, 1) #Eq (w3.1)
      #self.gated_control_transformer.bias.data.fill_(-1)
      self.non_linear = nn.Sigmoid()
    
    init_modules(self.modules())
    
  def forward(self, memories, controls, current_read, idx):
    #memories (N x num_cell x d), controls (N x num_cell x d), current_read (N x d), idx (int starting from 1)
    
    #Eq (w1)
    res_memory = self.control_memory_transfomer( torch.cat([current_read, memories[:,idx-1,:]], 1) ) #N x d
    
    if self.use_self_attention:
      current_control = controls[:,idx,:] # N x d
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
  def __init__(self, common_dim):
    super(ReadUnit, self).__init__()
    self.common_dim = common_dim
    
    #Eq (r1)
    self.pre_memory_transformer = nn.Linear(common_dim, common_dim)
    self.image_element_transformer = nn.Linear(common_dim, common_dim)
    
    #Eq (r2)
    self.intermediate_transformer = nn.Linear(2 * common_dim, common_dim)
    
    #Eq (r3.1)
    self.read_attention_transformer = nn.Linear(common_dim, 1)
    
    init_modules(self.modules())
    
  def forward(self, pre_memory, current_control, image):
      
    #pre_memory(Nxd), current_control(Nxd), image(NxdxHxW)
      
    image = image.transpose(1,2).transpose(2,3) #NXHxWxd
      
    #Eq (r1)
    trans_image = self.image_element_transformer(image) #NxHxWxd
    trans_pre_memory = self.pre_memory_transformer(pre_memory) #Nxd
    trans_pre_memory = trans_pre_memory.unsqueeze(1).unsqueeze(2).expand(trans_image.size()) #NxHxWxd
    intermediate = trans_pre_memory * trans_image #NxHxWxd
      
    #Eq (r2)
    trans_intermediate = self.intermediate_transformer(torch.cat([intermediate, image], 3)) #NxHxWxd
      
    #Eq (r3.1)
    trans_current_control = current_control.unsqueeze(1).unsqueeze(2).expand(trans_intermediate.size()) #NxHxWxd
    scores = self.read_attention_transformer(trans_current_control * trans_intermediate).squeeze(3) #NxHxWx1 -> NxHxW
      
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
  def __init__(self, common_dim):
    super(ControlUnit, self).__init__()
    self.common_dim = common_dim
    
    self.control_question_transformer = nn.Linear(2 * common_dim, common_dim) #Eq (c1)
    
    self.score_transformer = nn.Linear(common_dim, 1) # Eq (c2.1)
    
    init_modules(self.modules())
    
  def forward(self, pre_control, question, context, mask):
      
    #pre_control (Nxd), question (Nxd), context(NxLxd), mask(NxL)
      
    # N x d
    control_question = self.control_question_transformer(torch.cat([pre_control, question], 1)) #Eq (c1)
      
    #Eq (c2.1)
    scores = self.score_transformer(context * control_question.unsqueeze(1)).squeeze(2)  #NxLxd -> NxLx1 -> NxL
      
    #Eq (c2.2) : softmax
    scores = torch.exp(scores - scores.max(1, keepdim=True)[0]) * mask #mask help to elimiate null tokens
    scores = scores / scores.sum(1, keepdim=True) #NxL
      
    #Eq (c2.3)
    control = (context * scores.unsqueeze(2)).sum(1) #Nxd
      
    return control

class InputUnit(nn.Module):
  def __init__(self, common_dim):
    super(InputUnit, self).__init__()
    self.common_dim = common_dim
    self.question_transformer = nn.Linear(2 * common_dim, common_dim)
    
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
