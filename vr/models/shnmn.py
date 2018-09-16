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


def _random_tau(num_modules):
  tau_0 = torch.zeros(num_modules, num_modules+1)
  tau_1 = torch.zeros(num_modules, num_modules+1)
  xavier_uniform(tau_0)
  xavier_uniform(tau_1)
  return tau_0, tau_1


def _chain_tau():
  tau_0 = torch.zeros(3, 4)
  tau_1 = torch.zeros(3, 4)
  tau_0[0][1] = tau_1[0][0] = 10 #1st block - lhs inp img, rhs inp sentinel
  tau_0[1][2] = tau_1[1][0] = 10 #2nd block - lhs inp 1st block, rhs inp sentinel
  tau_0[2][3] = tau_1[2][0] = 10 #3rd block - lhs inp 2nd block, rhs inp sentinel
  return tau_0, tau_1


def _tree_tau():
  tau_0 = torch.zeros(3, 4)
  tau_1 = torch.zeros(3, 4)
  tau_0[0][1] = tau_1[0][0] = 10 #1st block - lhs inp img, rhs inp sentinel
  tau_0[1][1] = tau_1[1][0] = 10 #2st block - lhs inp img, rhs inp sentinel
  tau_0[2][2] = tau_1[2][3] = 10 #3rd block - lhs inp 1st block, rhs inp 2nd block
  return tau_0, tau_1


def _random_alpha(num_modules):
  alpha = torch.Tensor(num_modules, NUM_QUESTION_TOKENS)
  xavier_uniform(alpha)
  return alpha


def _good_alpha():
  alpha = torch.zeros(3, NUM_QUESTION_TOKENS)
  alpha[0][4] = 10 # LHS
  alpha[1][7] = 10 # RHS
  alpha[2][5] = 10 # relation
  return alpha


def _shnmn_func(question, img, num_modules, alpha, tau_0, tau_1, func):
  sentinel = torch.zeros_like(img) # B x 1 x C x H x W
  h_prev = torch.cat([sentinel, img], dim=1) # B x 2 x C x H x W

  for i in range(num_modules):
    alpha_curr = F.softmax(alpha[i])
    tau_0_curr = F.softmax(tau_0[i, :(i+2)])
    tau_1_curr = F.softmax(tau_1[i, :(i+2)])

    question_rep = torch.sum(alpha_curr.view(1,-1,1)*question, dim=1) #(B,D)
    # B x C x H x W
    lhs_rep = torch.sum(tau_0_curr.view(1, (i+2), 1, 1, 1)*h_prev, dim=1)
    # B x C x H x W
    rhs_rep = torch.sum(tau_1_curr.view(1, (i+2), 1, 1, 1)*h_prev, dim=1)
    h_i = func(question_rep, lhs_rep, rhs_rep) # B x C x H x W

    h_prev = torch.cat([h_prev, h_i.unsqueeze(1)], dim=1)

  return h_prev[:, -1, :, :, :]


class ConvFunc():
  def __init__(self, dim, kernel_size):
    self.dim = dim
    self.kernel_size = kernel_size

  def __call__(self, question_rep, lhs_rep, rhs_rep):
    cnn_weight_dim = self.dim*self.dim*self.kernel_size*self.kernel_size
    cnn_bias_dim = self.dim
    proj_cnn_weight_dim = 2*self.dim*self.dim
    proj_cnn_bias_dim = self.dim
    if (question_rep.size(1) !=
          proj_cnn_weight_dim + proj_cnn_bias_dim
          + cnn_weight_dim + cnn_bias_dim):
      raise ValueError

    # pick out CNN and projection CNN weights/biases
    cnn_weight = question_rep[:, : cnn_weight_dim]
    cnn_bias = question_rep[:, cnn_weight_dim : cnn_weight_dim + cnn_bias_dim]
    proj_weight = question_rep[:, cnn_weight_dim+cnn_bias_dim :
                              cnn_weight_dim+cnn_bias_dim+proj_cnn_weight_dim]
    proj_bias   = question_rep[:, cnn_weight_dim+cnn_bias_dim+proj_cnn_weight_dim:]

    cnn_out_total = []
    bs = question_rep.size(0)

    for i in range(bs):
      cnn_weight_curr = cnn_weight[i].view(self.dim, self.dim, self.kernel_size, self.kernel_size)
      cnn_bias_curr   = cnn_bias[i]
      proj_weight_curr = proj_weight[i].view(self.dim, 2*self.dim, 1, 1)
      proj_bias_curr = proj_bias[i]

      cnn_inp = F.conv2d(torch.cat([lhs_rep[[i]], rhs_rep[[i]]], 1),
                         proj_weight_curr,
                         bias=proj_bias_curr, padding=0)
      cnn_out_total.append(F.relu(F.conv2d(
        cnn_inp, cnn_weight_curr, bias=cnn_bias_curr, padding=self.kernel_size // 2)))

    return torch.cat(cnn_out_total)


class SHNMN(nn.Module):
  def __init__(self, vocab, feature_dim, module_dim,
      module_kernel_size, stem_dim, stem_num_layers,
      stem_subsample_layers, stem_kernel_size, stem_padding,
      stem_batchnorm, classifier_fc_layers,
      classifier_proj_dim, classifier_downsample,classifier_batchnorm,
      num_modules, hard_code_alpha=False, hard_code_tau=False,
      tau_init='random', alpha_init='random',
      model_type='soft', model_bernoulli=0.5,**kwargs):
    super().__init__()
    self.num_modules = num_modules
    # alphas and taus from Overleaf Doc.
    self.hard_code_alpha = hard_code_alpha
    self.hard_code_tau = hard_code_tau

    # create alphas
    if alpha_init == 'random':
      alpha = _random_alpha(num_modules)
    else:
      alpha = _good_alpha()

    if hard_code_alpha:
      self.alpha = Variable(alpha)
      if torch.cuda.is_available():
        self.alpha = self.alpha.cuda()
    else:
      self.alpha = nn.Parameter(alpha)

    # create taus
    if tau_init == 'tree':
      tau_0, tau_1 = _tree_tau()
      print("initializing with tree.")
    elif tau_init == 'chain':
      tau_0, tau_1 = _chain_tau()
      print("initializing with chain")
    else:
      tau_0, tau_1 = _random_tau(num_modules)

    if hard_code_tau:
      assert(tau_init in ['chain', 'tree'])
      self.tau_0 = Variable(tau_0)
      self.tau_1 = Variable(tau_1)
      if torch.cuda.is_available():
          self.tau_0 = self.tau_0.cuda()
          self.tau_1 = self.tau_1.cuda()
    else:
      self.tau_0   = nn.Parameter(tau_0)
      self.tau_1   = nn.Parameter(tau_1)

    embedding_dim_1 = module_dim + (module_dim*module_dim*module_kernel_size*module_kernel_size)
    embedding_dim_2 = module_dim + (2*module_dim*module_dim)

    self.question_embeddings_1 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_1)
    self.question_embeddings_2 = nn.Embedding(len(vocab['question_idx_to_token']),embedding_dim_2)

    stdv_1 = 1. / math.sqrt(module_dim*module_kernel_size*module_kernel_size)
    stdv_2 = 1. / math.sqrt(2*module_dim)

    self.question_embeddings_1.weight.data.uniform_(-stdv_1, stdv_1)
    self.question_embeddings_2.weight.data.uniform_(-stdv_2, stdv_2)


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
    self.model_type = model_type
    self.model_bernoulli = nn.Parameter(torch.Tensor([model_bernoulli]))


  def forward_hard(self, image, question):
    question = torch.cat([self.question_embeddings_1(question), self.question_embeddings_2(question)],dim=-1)
    stemmed_img = self.stem(image).unsqueeze(1) # B x 1 x C x H x W

    chain_tau_0, chain_tau_1 = _chain_tau()
    if torch.cuda.is_available():
        chain_tau_0 = chain_tau_0.cuda()
        chain_tau_1 = chain_tau_1.cuda()
    h_final_chain = _shnmn_func(question, stemmed_img,
                    self.num_modules, self.alpha,
                    Variable(chain_tau_0), Variable(chain_tau_1), self.func)
    tree_tau_0, tree_tau_1 = _tree_tau()
    if torch.cuda.is_available():
        tree_tau_0 = tree_tau_0.cuda()
        tree_tau_1 = tree_tau_1.cuda()
    h_final_tree  = _shnmn_func(question, stemmed_img,
                    self.num_modules, self.alpha,
                    Variable(tree_tau_0), Variable(tree_tau_1), self.func)

    prob = F.sigmoid(self.model_bernoulli[0])
    logits_tree  = self.classifier(h_final_tree)
    logits_chain = self.classifier(h_final_chain)
    return prob*logits_tree + (1.0 - prob)*logits_chain


  def forward_soft(self, image, question):
    question = torch.cat([self.question_embeddings_1(question), self.question_embeddings_2(question)],dim=-1)
    stemmed_img = self.stem(image).unsqueeze(1) # B x 1 x C x H x W

    h_final = _shnmn_func(question, stemmed_img, self.num_modules, self.alpha, self.tau_0, self.tau_1, self.func)
    return self.classifier(h_final)

  def forward(self, image, question):
    if self.model_type == 'hard': return self.forward_hard(image, question)
    else: return self.forward_soft(image, question)
