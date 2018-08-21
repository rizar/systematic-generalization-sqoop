#!/usr/bin/env python3

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vr.models.layers import init_modules, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem


class RelationNet(nn.Module):
  def __init__(self,
               vocab,
               feature_dim=(3, 64, 64),
               stem_num_layers=2,
               stem_batchnorm=True,
               stem_kernel_size=3,
               stem_stride=1,
               stem_padding=None,
               stem_feature_dim=24,
               module_num_layers=1,
               module_dim=128,
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               rnn_hidden_dim=128,
               # unused
               stem_subsample_layers=[],
               module_input_proj=None,
               module_residual=None,
               module_kernel_size=None,
               module_batchnorm=None,
               classifier_proj_dim=None,
               classifier_downsample=None,
               debug_every=float('inf'),
               print_verbose_every=float('inf'),
               verbose=True):
    super().__init__()

    # initialize stem
    self.stem = build_stem(feature_dim[0],
                           stem_feature_dim,
                           num_layers=stem_num_layers,
                           with_batchnorm=stem_batchnorm,
                           kernel_size=stem_kernel_size,
                           stride=stem_stride,
                           padding=stem_padding,
                           subsample_layers=stem_subsample_layers)
    tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
    module_H = tmp.size(2)
    module_W = tmp.size(3)

    # initialize relation model
    # (output of stem + 2 coordinates) * 2 objects + question vector
    relation_modules = [nn.Linear((stem_feature_dim + 2)*2 + rnn_hidden_dim, module_dim)]
    for _ in range(module_num_layers - 1):
      relation_modules.append(nn.Linear(module_dim, module_dim))
    self.relation = nn.Sequential(*relation_modules)

    # initialize coordinates to be appended to "objects"
    # can be switched to using torch instead of np after 0.4.1
    x = np.arange(module_H)
    y = np.arange(module_W)
    xv, yv = np.meshgrid(x,y)
    grid = np.stack([xv,yv], axis=2)
    flat_grid = grid.reshape(-1, 2)
    self.coordinates = Variable(torch.from_numpy(flat_grid)).float().cuda()

    # indices to slice a H*W x H*W matrix to remove relation between object and itself
    indices = flat_grid[flat_grid[:,0] != flat_grid[:,1]]
    self.slice_indices = Variable(torch.LongTensor(indices)).cuda()

    # initialize classifier (f_theta)
    num_answers = len(vocab['answer_idx_to_token'])
    self.classifier = build_classifier(module_dim,
                                      1,
                                      1,
                                      num_answers,
                                      classifier_fc_layers,
                                      classifier_proj_dim,
                                      classifier_downsample,
                                      classifier_batchnorm,
                                      classifier_dropout)

    init_modules(self.modules())

  def forward(self, image, question):
    # convert image to features (aka objects)
    features = self.stem(image)
    N, F, H, W = features.size()

    # conctenate coordinates to features
    features_flat = features.view(N, F, H*W).permute(0,2,1)               # N x H*W x F
    batch_coords = self.coordinates.unsqueeze(0).repeat(N, 1, 1)
    features_coords = torch.cat([features_flat, batch_coords], dim=2) # N x H*W x F+2

    # make matrix of all possible permuations of 2 features
    x_i = torch.unsqueeze(features_coords, 1)   # N x 1 x H*W x F+2
    x_i = x_i.repeat(1, H*W, 1,1)               # N x H*W x H*W x F+2
    x_j = torch.unsqueeze(features_coords, 2)   # N x H*W x 1 x F+2
    x_j = x_j.repeat(1, 1, H*W, 1)              # N x H*W x H*W x F+2
    feature_pairs = torch.cat([x_i, x_j], dim=3)  # N x H*W x H*W x 2*(F+2)

    # we don't want relations between an object and itself
    # slice to remove the eye of the matrix
    # feature_pairs = feature_pairs[self.slice_indices]  # N x H*(W-1) x 2*(F+2)
    N, _, _, F2 = feature_pairs.size()
    feature_pairs = feature_pairs.view(N, -1, F2)

    # concatenate question to feature pair
    _, ques, _ = question     # (N x Q)
    ques = ques.unsqueeze(1).repeat(1, feature_pairs.size(1), 1)
    feature_pairs_ques = torch.cat([feature_pairs, ques], dim=2)  # N x H*(W-1) x 2*(F+2)+Q

    # pass through model (g_theta)
    relations = self.relation(feature_pairs_ques)   # N x H*(W-1) x module_dim

    # sum across relations
    relations = torch.sum(relations, dim=1)         # N x module_dim

    # pass through classifier (f_theta)
    out = self.classifier(relations)

    return out
