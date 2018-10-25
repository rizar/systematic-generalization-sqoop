#!/usr/bin/env python3

import math
import pprint
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models

from vr.models.layers import init_modules, GlobalAveragePool, Flatten
from vr.models.layers import build_classifier, build_stem
import vr.programs


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
    'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        gammas = gammas.unsqueeze(2).unsqueeze(3).expand_as(x)
        betas = betas.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gammas * x) + betas


class FiLMedNet(nn.Module):
    def __init__(self, vocab, feature_dim=(1024, 14, 14),
                 stem_num_layers=2,
                 stem_batchnorm=False,
                 stem_kernel_size=3,
                 stem_subsample_layers=None,
                 stem_stride=1,
                 stem_padding=None,
                 stem_dim=64,
                 num_modules=4,
                 module_num_layers=1,
                 module_dim=128,
                 module_residual=True,
                 module_intermediate_batchnorm=False,
                 module_batchnorm=False,
                 module_batchnorm_affine=False,
                 module_dropout=0,
                 module_input_proj=1,
                 module_kernel_size=3,
                 classifier_proj_dim=512,
                 classifier_downsample='maxpool2',
                 classifier_fc_layers=(1024,),
                 classifier_batchnorm=False,
                 classifier_dropout=0,
                 condition_method='bn-film',
                 condition_pattern=[],
                 use_gamma=True,
                 use_beta=True,
                 use_coords=1,
                 debug_every=float('inf'),
                 print_verbose_every=float('inf'),
                 verbose=True,
                 ):
        super(FiLMedNet, self).__init__()

        num_answers = len(vocab['answer_idx_to_token'])

        self.stem_times = []
        self.module_times = []
        self.classifier_times = []
        self.timing = False

        self.num_modules = num_modules
        self.module_num_layers = module_num_layers
        self.module_batchnorm = module_batchnorm
        self.module_dim = module_dim
        self.condition_method = condition_method
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.use_coords_freq = use_coords
        self.debug_every = debug_every
        self.print_verbose_every = print_verbose_every

        # Initialize helper variables
        self.stem_use_coords = (stem_stride == 1) and (self.use_coords_freq > 0)
        self.condition_pattern = condition_pattern
        if len(condition_pattern) == 0:
            self.condition_pattern = []
            for i in range(self.module_num_layers * self.num_modules):
                self.condition_pattern.append(self.condition_method != 'concat')
        else:
            self.condition_pattern = [i > 0 for i in self.condition_pattern]
        self.extra_channel_freq = self.use_coords_freq
        self.block = FiLMedResBlock
        self.num_cond_maps = 2 * self.module_dim if self.condition_method == 'concat' else 0
        self.fwd_count = 0
        self.num_extra_channels = 2 if self.use_coords_freq > 0 else 0
        if self.debug_every <= -1:
            self.print_verbose_every = 1

        # Initialize stem
        stem_feature_dim = feature_dim[0] + self.stem_use_coords * self.num_extra_channels
        self.stem = build_stem(
            stem_feature_dim, stem_dim, module_dim,
          num_layers=stem_num_layers, with_batchnorm=stem_batchnorm,
          kernel_size=stem_kernel_size, stride=stem_stride, padding=stem_padding,
          subsample_layers=stem_subsample_layers)
        tmp = self.stem(Variable(torch.zeros([1, feature_dim[0], feature_dim[1], feature_dim[2]])))
        module_H = tmp.size(2)
        module_W = tmp.size(3)

        self.stem_coords = coord_map((feature_dim[1], feature_dim[2]))
        self.coords = coord_map((module_H, module_W))
        self.default_weight = torch.ones(1, 1, self.module_dim).to(device)
        self.default_bias = torch.zeros(1, 1, self.module_dim).to(device)

        # Initialize FiLMed network body
        self.function_modules = {}
        self.vocab = vocab
        for fn_num in range(self.num_modules):
            with_cond = self.condition_pattern[self.module_num_layers * fn_num:
                                                self.module_num_layers * (fn_num + 1)]
            mod = self.block(module_dim, with_residual=module_residual,
                             with_intermediate_batchnorm=module_intermediate_batchnorm, with_batchnorm=module_batchnorm,
                             with_cond=with_cond,
                             dropout=module_dropout,
                             num_extra_channels=self.num_extra_channels,
                             extra_channel_freq=self.extra_channel_freq,
                             with_input_proj=module_input_proj,
                             num_cond_maps=self.num_cond_maps,
                             kernel_size=module_kernel_size,
                             batchnorm_affine=module_batchnorm_affine,
                             num_layers=self.module_num_layers,
                             condition_method=condition_method,
                             debug_every=self.debug_every)
            self.add_module(str(fn_num), mod)
            self.function_modules[fn_num] = mod

        # Initialize output classifier
        self.classifier = build_classifier(module_dim + self.num_extra_channels, module_H, module_W,
                                           num_answers, classifier_fc_layers, classifier_proj_dim,
                                           classifier_downsample, with_batchnorm=classifier_batchnorm,
                                           dropout=classifier_dropout)

        init_modules(self.modules())

    def forward(self, x, film, save_activations=False):
        # Initialize forward pass and externally viewable activations
        self.fwd_count += 1
        if save_activations:
            self.feats = None
            self.module_outputs = []
            self.cf_input = None

        if self.debug_every <= -2:
            pdb.set_trace()

        # Prepare FiLM layers
        gammas = None
        betas = None
        if self.condition_method == 'concat':
            # Use parameters usually used to condition via FiLM instead to condition via concatenation
            cond_params = film[:,:,:2*self.module_dim]
            cond_maps = cond_params.unsqueeze(3).unsqueeze(4).expand(cond_params.size() + x.size()[-2:])
        else:
            gammas, betas = torch.split(film[:,:,:2*self.module_dim], self.module_dim, dim=-1)
            if not self.use_gamma:
                gammas = self.default_weight.expand_as(gammas)
            if not self.use_beta:
                betas = self.default_bias.expand_as(betas)

        # Propagate up image features CNN
        stem_batch_coords = None
        batch_coods = None
        if self.use_coords_freq > 0:
            stem_batch_coords = self.stem_coords.unsqueeze(0).expand(
                torch.Size((x.size(0), *self.stem_coords.size())))
            batch_coords = self.coords.unsqueeze(0).expand(
                torch.Size((x.size(0), *self.coords.size())))
        if self.stem_use_coords:
            x = torch.cat([x, stem_batch_coords], 1)
        feats = self.stem(x)
        if save_activations:
            self.feats = feats
        N, _, H, W = feats.size()

        # Propagate up the network from low-to-high numbered blocks
        module_inputs = torch.zeros(feats.size()).unsqueeze(1).expand(
            N, self.num_modules, self.module_dim, H, W).to(device)
        module_inputs[:,0] = feats
        for fn_num in range(self.num_modules):
            if self.condition_method == 'concat':
                layer_output = self.function_modules[fn_num](module_inputs[:,fn_num],
                  extra_channels=batch_coords, cond_maps=cond_maps[:,fn_num])
            else:
                layer_output = self.function_modules[fn_num](module_inputs[:,fn_num],
                  gammas[:,fn_num,:], betas[:,fn_num,:], batch_coords)

            # Store for future computation
            if save_activations:
                self.module_outputs.append(layer_output)
            if fn_num == (self.num_modules - 1):
                final_module_output = layer_output
            else:
                module_inputs_updated = module_inputs.clone()
                module_inputs_updated[:,fn_num+1] = module_inputs_updated[:,fn_num+1] + layer_output
                module_inputs = module_inputs_updated

        if self.debug_every <= -2:
            pdb.set_trace()

        # Run the final classifier over the resultant, post-modulated features.
        if self.use_coords_freq > 0:
            final_module_output = torch.cat([final_module_output, batch_coords], 1)
        if save_activations:
            self.cf_input = final_module_output
        out = self.classifier(final_module_output)

        if ((self.fwd_count % self.debug_every) == 0) or (self.debug_every <= -1):
            pdb.set_trace()
        return out


class FiLMedResBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, with_residual=True, with_intermediate_batchnorm=False, with_batchnorm=True,
                 with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                 with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
                 num_layers=1, condition_method='bn-film', debug_every=float('inf')):
        if out_dim is None:
            out_dim = in_dim
        super(FiLMedResBlock, self).__init__()
        self.with_residual = with_residual
        self.with_intermediate_batchnorm = with_intermediate_batchnorm
        self.with_batchnorm = with_batchnorm
        self.with_cond = with_cond
        self.dropout = dropout
        self.extra_channel_freq = 0 if num_extra_channels == 0 else extra_channel_freq
        self.with_input_proj = with_input_proj  # Kernel size of input projection
        self.num_cond_maps = num_cond_maps
        self.kernel_size = kernel_size
        self.batchnorm_affine = batchnorm_affine
        self.num_layers = num_layers
        self.condition_method = condition_method
        self.debug_every = debug_every

        if self.kernel_size % 2 == 0:
            raise(NotImplementedError)
        if self.num_layers >= 2:
            raise(NotImplementedError)

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_input_proj:
            self.input_proj = nn.Conv2d(in_dim + (num_extra_channels if self.extra_channel_freq >= 1 else 0),
                                        in_dim, kernel_size=self.with_input_proj, padding=self.with_input_proj // 2)

        self.conv1 = nn.Conv2d(in_dim + self.num_cond_maps +
                               (num_extra_channels if self.extra_channel_freq >= 2 else 0),
                                out_dim, kernel_size=self.kernel_size,
                                padding=self.kernel_size // 2)
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            self.film = FiLM()
        if self.with_intermediate_batchnorm:
            self.bn0 = nn.BatchNorm2d(in_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
        if self.with_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_dim, affine=((not self.with_cond[0]) or self.batchnorm_affine))
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            self.film = FiLM()
        if dropout > 0:
            self.drop = nn.Dropout2d(p=self.dropout)
        if ((self.condition_method == 'relu-film' or self.condition_method == 'block-output-film')
             and self.with_cond[0]):
            self.film = FiLM()

        init_modules(self.modules())

    def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):
        if self.debug_every <= -2:
            pdb.set_trace()

        if self.condition_method == 'block-input-film' and self.with_cond[0]:
            x = self.film(x, gammas, betas)

        # ResBlock input projection
        if self.with_input_proj:
            if extra_channels is not None and self.extra_channel_freq >= 1:
                x = torch.cat([x, extra_channels], 1)
            x = self.input_proj(x)
            if self.with_intermediate_batchnorm:
                x = self.bn0(x)
            x = F.relu(x)
        out = x

        # ResBlock body
        if cond_maps is not None:
            out = torch.cat([out, cond_maps], 1)
        if extra_channels is not None and self.extra_channel_freq >= 2:
            out = torch.cat([out, extra_channels], 1)
        out = self.conv1(out)
        if self.condition_method == 'conv-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        if self.with_batchnorm:
            out = self.bn1(out)
        if self.condition_method == 'bn-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        if self.dropout > 0:
            out = self.drop(out)
        out = F.relu(out)
        if self.condition_method == 'relu-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)

        # ResBlock remainder
        if self.with_residual:
            out = x + out
        if self.condition_method == 'block-output-film' and self.with_cond[0]:
            out = self.film(out, gammas, betas)
        return out


class ConcatFiLMedResBlock(nn.Module):
    def __init__(self, num_input, in_dim, out_dim=None, with_residual=True, with_intermediate_batchnorm=False, with_batchnorm=True,
                 with_cond=[False], dropout=0, num_extra_channels=0, extra_channel_freq=1,
                 with_input_proj=0, num_cond_maps=0, kernel_size=3, batchnorm_affine=False,
                 num_layers=1, condition_method='bn-film', debug_every=float('inf')):
        super(ConcatFiLMedResBlock, self).__init__()
        self.proj = nn.Conv2d(num_input * in_dim, in_dim, kernel_size=1, padding=0)
        self.tfilmedResBlock = FiLMedResBlock(in_dim=in_dim, out_dim=out_dim, with_residual=with_residual,
            with_intermediate_batchnorm=with_intermediate_batchnorm, with_batchnorm=with_batchnorm,
            with_cond=with_cond, dropout=dropout, num_extra_channels=num_extra_channels, extra_channel_freq=extra_channel_freq,
            with_input_proj=with_input_proj, num_cond_maps=num_cond_maps, kernel_size=kernel_size, batchnorm_affine=batchnorm_affine,
            num_layers=num_layers, condition_method=condition_method, debug_every=debug_every)

    def forward(self, x, gammas=None, betas=None, extra_channels=None, cond_maps=None):
        out = torch.cat(x, 1) # Concatentate along depth
        out = F.relu(self.proj(out))
        out = self.tfilmedResBlock(out, gammas=gammas, betas=betas, extra_channels=extra_channels, cond_maps=cond_maps)
        return out


def coord_map(shape, start=-1, end=1):
    """
    Gives, a 2d shape tuple, returns two mxn coordinate maps,
    Ranging min-max in the x and y directions, respectively.
    """
    m, n = shape
    x_coord_row = torch.linspace(start, end, steps=n).to(device)
    y_coord_row = torch.linspace(start, end, steps=m).to(device)
    x_coords = x_coord_row.unsqueeze(0).expand(torch.Size((m, n))).unsqueeze(0)
    y_coords = y_coord_row.unsqueeze(1).expand(torch.Size((m, n))).unsqueeze(0)
    return Variable(torch.cat([x_coords, y_coords], 0))
