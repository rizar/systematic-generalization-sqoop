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
    def __init__(self,feature_dim, stem_dim, module_dim, stem_num_layers, stem_subsample_layers, stem_kernel_size, stem_padding, stem_batchnorm,  num_modules):
        super().__init__()
        self.num_modules = num_modules
        # alphas and taus from Overleaf Doc.
        self.alpha = nn.Parameter(torch.Tensor(num_modules, num_question_tokens))
        xavier_uniform(self.alpha)
        self.tau   = nn.Parameter(torch.Tensor(num_modules, num_modules))
        xavier_uniform(self.tau)
        # stem for processing the image into a 3D tensor
        self.stem = build_stem(feature_dim[0], stem_dim, module_dim,
                           num_layers=stem_num_layers,
                           subsample_layers=stem_subsample_layers,
                           kernel_size=stem_kernel_size,
                           padding=stem_padding,
                           with_batchnorm=stem_batchnorm)
        
    
    def foward(self, image, question):
        image = self.stem(image)
        for i in range(self.num_modules):
            pass
