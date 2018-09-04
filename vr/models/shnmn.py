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

NUM_QUESTION_TOKENS=8
def _softmax(vec):
  '''
  Take a vector (say dimension D) and softmax it to produce soft weights
  '''
  max_weight = torch.max(vec)
  unnormalized_weights =  torch.exp(vec-max_weight) 
  return unnormalized_weights / torch.sum(unnormalized_weights)


class ConvFunc(nn.Module):
  def __init__(self, dim, kernel_size):
    super(ConvFunc, self).__init__()
    self.proj = nn.Conv2d(2*dim, dim, kernel_size=1, padding = 0)
    self.dim = dim
    self.kernel_size = kernel_size

  def forward(self, question_rep, lhs_rep, rhs_rep):
    cnn_weight_dim = self.dim*self.dim*self.kernel_size*self.kernel_size
    cnn_bias_dim = self.dim
    if question_rep.size(1) != cnn_weight_dim + cnn_bias_dim: raise ValueError
    cnn_weight = question_rep[:, : cnn_weight_dim]
    cnn_bias = question_rep[:, cnn_weight_dim : ]

    cnn_out_total = []
    bs = question_rep.size(0)
    cnn_inp = self.proj(torch.cat([lhs_rep, rhs_rep], 1))

    for i in range(bs):
      cnn_weight_curr = cnn_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
      cnn_bias_curr   = cnn_bias[i]
      cnn_out_total.append(F.conv2d(cnn_inp[[i]], cnn_weight_curr, bias = cnn_bias_curr, padding = self.kernel_size // 2))

    return torch.cat(cnn_out_total)

class SHNMN(nn.Module):
  def __init__(self, vocab, feature_dim, module_dim, 
      module_kernel_size, stem_dim, stem_num_layers, 
      stem_subsample_layers, stem_kernel_size, stem_padding, 
      stem_batchnorm, classifier_fc_layers, 
      classifier_proj_dim, classifier_downsample,classifier_batchnorm, 
      num_modules, **kwargs):
    super().__init__()
    self.num_modules = num_modules
    # alphas and taus from Overleaf Doc.
    self.alpha = nn.Parameter(torch.Tensor(num_modules, NUM_QUESTION_TOKENS))
    xavier_uniform(self.alpha)
    self.tau_0   = nn.Parameter(torch.Tensor(num_modules, num_modules)) # weights for left  child
    self.tau_1   = nn.Parameter(torch.Tensor(num_modules, num_modules)) # weights for right child
    xavier_uniform(self.tau_0)
    xavier_uniform(self.tau_1)
    self.question_embeddings = nn.Embedding(len(vocab['question_idx_to_token']), module_dim + (module_dim*module_dim*module_kernel_size*module_kernel_size))
    # stem for processing the image into a 3D tensor
    self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
               num_layers=stem_num_layers,
               subsample_layers=stem_subsample_layers,
               kernel_size=stem_kernel_size,
               padding=stem_padding,
               with_batchnorm=stem_batchnorm)

    tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
    module_H = tmp.size(2)
    module_W = tmp.size(3)
    num_answers = len(vocab['answer_idx_to_token'])
    self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
              classifier_fc_layers,
              classifier_proj_dim,
              classifier_downsample,
              with_batchnorm=classifier_batchnorm) 

    self.func = ConvFunc(module_dim, module_kernel_size)
  
  def forward(self, image, question):
    question = self.question_embeddings(question)
    h_prev = self.stem(image).unsqueeze(1) # B x1 x C x H x W
    for i in range(self.num_modules):
      question_rep = torch.sum( _softmax(self.alpha[i]).view(1,-1,1)*question, dim=1) #(B,D)
      lhs_rep = torch.sum(_softmax(self.tau_0[i, :(i+1)]).view(1, (i+1), 1, 1, 1)*h_prev, dim=1) # B x C x H x W  
      rhs_rep = torch.sum(_softmax(self.tau_1[i, :(i+1)]).view(1, (i+1), 1, 1, 1)*h_prev, dim=1) # B x C x H x W
      h_i = self.func(question_rep, lhs_rep, rhs_rep) # B x C x H x W

      h_prev = torch.cat([h_prev, h_i.unsqueeze(1)], dim = 1)

    return self.classifier(h_prev[:, -1, :, :, :]) 
