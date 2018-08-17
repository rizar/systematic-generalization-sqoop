#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vr.models.layers import init_modules, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem


class RelationNet(nn.Module):
  def __init__(self,
               vocab,
               batch_size,
               feature_dim=(3, 64, 64),
               stem_num_layers=2,
               stem_batchnorm=True,
               stem_kernel_size=3,
               stem_subsample_layers=None,
               stem_stride=1,
               stem_padding=None,
               stem_feature_dim=24,
               module_num_layers=1,
               module_dim=128,
               classifier_proj_dim=512,
               classifier_downsample=None,
               classifier_fc_layers=(1024,),
               classifier_batchnorm=False,
               classifier_dropout=0,
               rnn_hidden_dim=128,
               debug_every=float('inf'),
               print_verbose_every=float('inf'),
               verbose=True):
    super().__init__()

    # initialize stem
    self.stem = build_stem(feature_dim,
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
      relation_modules.extend(nn.Linear(module_dim, module_dim))
    self.relation = nn.Sequential(*relation_modules)

    # initialize coordinates to be appended to "objects"
    # can be switched to using torch instead of np after 0.4.1
    x = np.arange(module_H)
    y = np.arange(module_W)
    xv, yv = np.meshgrid(x,y)
    grid = np.stack([x,y], axis=2)
    flat_grid = np.reshape(-1, 2)
    batch_grid = np.tile(flat_grid, (batch_size, 1, 1))
    self.coordinates = torch.from_numpy(batch_grid).cuda()

    # indices to slice a H*W x H*W matrix of permutations
    size = module_H * module_W
    indices = list(zip(*itertools.combinations(range(size), 2)))
    self.comb_indices = torch.LongTensor(indices).cuda()

    # initialize classifier (f_theta)
    num_answers = len(vocab['answer_idx_to_token'])
    self.clasifier = build_classifier(module_dim,
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

    # conctenate coordinates to feature dimension
    features_flat = features.view(N, F, H*W).permute(0,2,1)               # N x H*W x F
    features_coords = torch.cat([features_flat, self.coordinates], dim=2) # N x H*W x F+2

    # make matrix of all possible permuations of 2 features
    x_i = torch.unsqueeze(features_coords, 1)   # N x 1 x H*W x F+2
    x_i = x_i.repeat(1, H*W, 1,1)               # N x H*W x H*W x F+2
    x_j = torch.unsqueeze(x_flat,2)             # N x H*W x 1 x F+2
    x_j = x_j.repeat(1, 1, H*W, 1)              # N x H*W x H*W x F+2
    feature_perms = torch.cat([xi, xj], dim=3)  # N x H*W x H*W x 2*(F+2)

    # we don't want permutations (e.g. relation(0,1) and relation(1,0))
    # slice to get all possible combinations without replacement
    feature_pairs = feature_perms[self.comb_indices]  # N x (H*W choose 2) x 2*(F+2)
    num_pairs = feature_pairs.size[1]

    # concatenate question to feature pair
    _, ques, _ = question     # (N x Q)
    ques = ques.unsqueeze(1).expand(1, num_pairs, 1)
    feature_pairs_ques = torch.cat([feature_pairs, ques], dim=2)  # N x (H*W choose 2) x 2*(F+2)+Q

    # pass through model (g_theta)
    relations = self.relation(feature_pairs_ques)   # N x (H*W choose 2) x module_dim

    # sum across relations
    relations = torch.sum(relations, dim=1)         # N x module_dim

    if self.debug_every <= -2:
      pdb.set_trace()

    # pass through classifier (f_theta)
    out = self.classifier(relations)

    if ((self.fwd_count % self.debug_every) == 0) or (self.debug_every <= -1):
      pdb.set_trace()

    return out
