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


def _softmax(vec):
    '''
    Take a vector (say dimension D) and softmax it to produce soft weights
    '''
    max_weight = torch.max(vec)
    unnormalized_weights =  torch.exp(vec-max_weight) 
    return unnormalized_weights / torch.sum(unnormalized_weights)

class SHNMN(nn.Module):
    def __init__(self,feature_dim, stem_dim, module_dim, stem_num_layers, 
            stem_subsample_layers, stem_kernel_size, stem_padding, 
            stem_batchnorm, num_answers, classifier_fc_layers, 
            classifier_proj_dim, classifier_downsample,classifier_batchnorm, 
            classifier_dropout, num_modules):
        super().__init__()
        self.num_modules = num_modules
        # alphas and taus from Overleaf Doc.
        self.alpha = nn.Parameter(torch.Tensor(num_modules, num_question_tokens))
        xavier_uniform(self.alpha)
        self.tau_0   = nn.Parameter(torch.Tensor(num_modules, num_modules)) # weights for left  child
        self.tau_1   = nn.Parameter(torch.Tensor(num_modules, num_modules)) # weights for right child
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
        h_prev = self.stem(image).unsqueeze(1) # B x1 x H x W x C
        for i in range(self.num_modules):
            question_rep = torch.sum( _softmax(self.alphas[i]).view(1,-1,1)*question, dim=1) #(B,D)
            lhs_rep = torch.sum(_softmax(self.tau_0[i, :(i+1)]).view(1, (i+1), 1, 1, 1)*h_prev, dim=1) # B x H x W x C  
            rhs_rep = torch.sum(_softmax(self.tau_0[i, :(i+1)]).view(1, (i+1), 1, 1, 1)*h_prev, dim=1) # B x H x W x C
            # use a hypernet that takes the question and returns a 3 x 3 x 16 x 16 convolution filter bank? 
            h_i = self.func(question_rep, lhs_rep, rhs_rep)

            h_prev = torch.stack([h_prev, h_i.unsqueeze(1)], dim = 1)

        return self.classifier(h_prev[:, -1, :, :, :]) 
