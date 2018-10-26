#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.autograd import Variable

from vr.models.layers import (build_classifier,
                              build_stem,
                              init_modules)


class ConvLSTM(nn.Module):
    def __init__(self,
                 vocab,
                 feature_dim=[3, 64, 64],
                 stem_dim=128,
                 module_dim=128,
                 stem_num_layers=2,
                 stem_batchnorm=True,
                 stem_kernel_size=3,
                 stem_stride=1,
                 stem_padding=None,
                 stem_feature_dim=24,
                 stem_subsample_layers=None,
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 rnn_hidden_dim=128,
                 **kwargs):
        super().__init__()

        # initialize stem
        self.stem = build_stem(feature_dim[0],
                               stem_dim,
                               module_dim,
                               num_layers=stem_num_layers,
                               with_batchnorm=stem_batchnorm,
                               kernel_size=stem_kernel_size,
                               stride=stem_stride,
                               padding=stem_padding,
                               subsample_layers=stem_subsample_layers)
        tmp = self.stem(Variable(torch.zeros([1] + feature_dim)))
        _, F, H, W = tmp.size()

        # initialize classifier
        # TODO(mnoukhov): fix this for >1 layer RNN
        question_dim = rnn_hidden_dim
        image_dim = F*H*W
        num_answers = len(vocab['answer_idx_to_token'])
        self.classifier = build_classifier(image_dim + question_dim,
                                           1,
                                           1,
                                           num_answers,
                                           classifier_fc_layers,
                                           None,
                                           None,
                                           classifier_batchnorm,
                                           classifier_dropout)

        init_modules(self.modules())

    def forward(self, image, question):
        # convert image to features
        img_feats = self.stem(image)                      # N x F x H x W
        img_feats = img_feats.view(img_feats.size(0), -1) # N x F*H*W

        # get hidden state from question
        _, q_feats, _ = question                          # N x Q

        # concatenate feats
        feats = torch.cat([img_feats, q_feats], dim=1)   # N x F*H*W+Q

        # pass through classifier
        out = self.classifier(feats)

        return out
