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

class SHNMN(nn.Module):
    def __init__(self,feature_dim, stem_dim, module_dim, stem_num_layers, stem_subsample_layers, stem_kernel_size, stem_padding, stem_batchnorm, num_answers, classifier_fc_layers, classifier_proj_dim, classifier_downsample, classifier_batchnorm, classifier_dropout, num_modules):
        super().__init__()
        self.num_modules = num_modules
        # alphas and taus from Overleaf Doc.
        self.alpha = nn.Parameter(torch.Tensor(num_modules, num_question_tokens))
        xavier_uniform(self.alpha)
        self.tau_0   = nn.Parameter(torch.Tensor(num_modules, num_modules))
        self.tau_1   = nn.Parameter(torch.Tensor(num_modules, num_modules))
        xavier_uniform(self.tau_0)
        xavier_uniform(self.tau_1)
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
        self.classifier = build_classifier(module_dim, module_H, module_W, num_answers,
                          classifier_fc_layers,
                          classifier_proj_dim,
                          classifier_downsample,
                          with_batchnorm=classifier_batchnorm
                          dropout=classifier_dropout) 

        self.func = None # Biggest TODO
    
    def foward(self, image, question):
        h_prev = [self.stem(image)]
        for i in range(self.num_modules):
            question_rep = None
            lhs_rep = None
            rhs_rep = None
            h_i = self.func(question_rep, lhs_rep, rhs_rep)

            h_prev.append(h_i)

        return self.classifier(h_prev[-1]) 
