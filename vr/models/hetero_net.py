"""Heterogenous ModuleNet as done originally in Hu et al

TODO:
- batchnorm?
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

from vr.models.layers import build_stem
from vr.models.module_net import ModuleNet


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Find(nn.Module):
    # Input:
    #   image_feat_grid: [N, D_im, H, W]
    #   text_param: [N, D_txt]
    # Output:
    #   image_att: [N, 1, H, W]
    def __init__(self, image_dim, text_dim, map_dim=500):
        super().__init__()
        self.conv1 = nn.Conv2d(image_dim, map_dim, 1)
        self.embed = nn.Embedding(text_dim, map_dim)
        self.conv2 = nn.Conv2d(map_dim, 1, 1)
        self.map_dim = map_dim

    def forward(self, text, images):
        image_mapped = self.conv1(images)
        text_mapped = self.embed(text).view(-1, self.map_dim, 1, 1)

        mult_norm = F.normalize(image_mapped * text_mapped, p=2, dim=1)
        return self.conv2(mult_norm)


class Transform(nn.Module):
    # Input:
    #   image_att: [N, 1, H, W]
    #   text: [N, D_txt]
    # Output:
    #   image_att: [N, 1, H, W]
    def __init__(self, text_dim, map_dim=500, kernel_size=3):
        super().__init__()
        if kernel_size % 2 == 0:
            raise NotImplementedError()
        self.conv1 = nn.Conv2d(1, map_dim, kernel_size, padding=kernel_size // 2)
        self.embed = nn.Embedding(text_dim, map_dim)
        self.conv2 = nn.Conv2d(map_dim, 1, 1)
        self.map_dim = map_dim

    def forward(self, text, image_att):
        image_att_mapped = self.conv1(image_att)
        text_mapped = self.embed(text).view(-1, self.map_dim, 1, 1)

        mult_norm = F.normalize(image_att_mapped * text_mapped, p=2, dim=1)
        return self.conv2(mult_norm)


class And(nn.Module):
    # Input:
    #   att_grid_0: [N, 1, H, W]
    #   att_grid_1: [N, 1, H, W]
    # Output:
    #   att_grid_and: [N, 1, H, W]
    def forward(self, att1, att2):
        return torch.min(att1, att2)


class Answer(nn.Module):
    # Input:
    #   att_grid: [N, 1, H, W]
    # Output:
    #   answer_scores: [N, self.num_answers]
    def __init__(self, num_answers):
        super().__init__()
        self.linear = nn.Linear(3, num_answers)

    def forward(self, att):
        att_min = att.min(dim=-1)[0].min(dim=-1)[0]
        att_max = att.max(dim=-1)[0].max(dim=-1)[0]
        att_mean = att.mean(dim=-1).mean(dim=-1)
        att_reduced = torch.cat((att_min, att_mean, att_max), dim=1)

        return self.linear(att_reduced)


class HeteroModuleNet(ModuleNet):
    def __init__(self,
                 vocab,
                 feature_dim,
                 stem_num_layers,
                 stem_kernel_size,
                 stem_stride,
                 stem_padding,
                 stem_batchnorm,
                 module_dim,
                 module_batchnorm,
                 verbose=True):
        super(ModuleNet, self).__init__()

        self.program_idx_to_token = vocab['program_idx_to_token']
        self.answer_to_idx = vocab['answer_idx_to_token']
        self.text_token_to_idx = vocab['text_token_to_idx']
        self.program_token_to_module_text = vocab['program_token_to_module_text']
        self.name_to_module = {
            'and': And(),
          'answer': lambda x: x,
          'find': Find(module_dim, len(self.text_token_to_idx)),
          'transform': Transform(len(self.text_token_to_idx)),
        }
        self.name_to_num_inputs = {
            'and': 2,
          'answer': 1,
          'find': 1,
          'transform': 1,
        }

        input_C, input_H, input_W = feature_dim
        self.stem = build_stem(input_C,
                               module_dim,
                               num_layers=stem_num_layers,
                               kernel_size=stem_kernel_size,
                               stride=stem_stride,
                               padding=stem_padding,
                               with_batchnorm=stem_batchnorm)

        self.classifier = Answer(len(self.answer_to_idx))

        if verbose:
            print('Here is my stem:')
            print(self.stem)
            print('Here is my classifier:')
            print(self.classifier)

        for name, module in self.name_to_module.items():
            if name != 'answer':
                self.add_module(name, module)

        self.save_module_outputs = False

    def _forward_modules_ints_helper(self, feats, program, i, j):
        if j >= program.size(1):
            raise IndexError('malformed program, reached index', j)

        fn_idx = program.data[i, j]
        fn_str = self.program_idx_to_token[fn_idx]

        if fn_str == '<START>':
            return self._forward_modules_ints_helper(feats, program, i, j + 1)
        elif fn_str in ['<NULL>', '<END>']:
            raise IndexError('reached area out of program ', fn_str)

        j += 1
        if fn_str == 'scene':
            output = feats[i].unsqueeze(0)
        else:
            module_name, text_token = self.program_token_to_module_text[fn_str]
            module = self.name_to_module[module_name]
            num_inputs = self.name_to_num_inputs[module_name]
            module_inputs = []

            if text_token is not None:
                # very ugly
                input_text = torch.LongTensor([self.text_token_to_idx[text_token]]).unsqueeze(0)
                if program.is_cuda:
                    input_text = input_text.to(device)
                module_inputs.append(Variable(input_text))

            for _ in range(num_inputs):
                module_input, j = self._forward_modules_ints_helper(feats, program, i, j)
                module_inputs.append(module_input)

            output = module(*module_inputs)

        return output, j
