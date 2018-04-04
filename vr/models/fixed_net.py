import torch
import torch.nn as nn

from vr.models.layers import build_stem
from vr.models.module_net import ModuleNet

# TODO(mnoukhov)
# batchnorm?

class Find(nn.Module):
  # Input:
  #   image_feat_grid: [N, D_im, H, W]
  #   text_param: [N, D_txt]
  # Output:
  #   image_att: [N, 1, H, W]
  def __init__(self, image_dim, text_dim, map_dim=500):
    super().__init__()
    self.conv2d = nn.Conv2d(image_dim, map_dim, 1)
    self.linear = nn.Linear(text_dim, map_dim)
    self.conv3d = nn.Conv3d(map_dim, 1, 1)

  def forward(self, images, text):
    image_mapped = self.conv2d(images)

    text_mapped = self.linear(text)
    text_reshape = text_mapped.unsqueeze(2).unsqueeze(2)

    mult_norm = torch.norm(image_mapped * text_reshape, p=2, dim=1)
    out = self.conv3d(mult_norm)
    return out


class Transform(nn.Module):
  # Input:
  #   image_att: [N, 1, H, W]
  #   text: [N, D_txt]
  # Output:
  #   image_att: [N, 1, H, W]
  def __init__(self, text_dim, map_dim=500, kernel_size=3):
    super().__init__()
    self.conv2d = nn.Conv2d(1, map_dim, kernel_size)
    self.linear = nn.Linear(text_dim, map_dim)
    self.conv3d = nn.Conv3d(map_dim, 1, 1)

  def forward(self, image_att, text):
    image_att_mapped = self.conv2d(image_att)

    text_mapped = self.linear(text)
    text_reshape = text_mapped.unsqueeze(2).unsqueeze(2)

    mult_norm = torch.norm(image_att_mapped * text_reshape, p=2, dim=1)
    out = self.conv3d(mult_norm)
    return out


class And(nn.Module):
  # Input:
  #   att_grid_0: [N, 1, H, W]
  #   att_grid_1: [N, 1, H, W]
  # Output:
  #   att_grid_and: [N, 1, H, W]
  def forward(self, att1, att2):
    min_, _ = torch.min(att1, att2)
    return min_


class Answer(nn.Module):
  # Input:
  #   att_grid: [N, 1, H, W]
  # Output:
  #   answer_scores: [N, self.num_answers]
  def __init__(self, num_answers):
    super().__init__()
    self.linear = nn.Linear(3, num_answers)

  def forward(self, att):
    att_min, _ = att.min(dim=3).min(dim=2)
    att_mean, _ = att.mean(dim=3).mean(dim=2)
    att_max, _ = att.max(dim=3).max(dim=2)
    att_reduced = torch.cat((att_min, att_mean, att_max), dim=1)

    out = self.linear(att_reduced)
    return out


class FixedModuleNet(ModuleNet):
  def __init__(self,
               vocab,
               feature_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               module_dim=128,
               module_batchnorm=False,
               verbose=True):
    super(ModuleNet, self).__init__()

    input_C, input_H, input_W = feature_dim
    self.stem = build_stem(input_C,
                           module_dim,
                           num_layers=stem_num_layers,
                           with_batchnorm=stem_batchnorm)
    if verbose:
      print('Here is my stem:')
      print(self.stem)

    self.vocab = vocab
    self.num_answer = len(vocab['answer_idx_to_token'])
    self.text_dim = len(vocab['text_token_to_idx'])

    self.name_to_module = {
      'and': And(),
      'answer': Answer(self.num_answer),
      'find': Find(module_dim, self.text_dim),
      'transform': Transform(self.text_dim),
    }
    self.name_to_num_inputs = {
      'and': 2,
      'answer': 1,
      'find': 1,
      'transform': 1,
    }
    for name, module in self.name_to_module.items():
      self.add_module(name, module)

    self.save_module_outputs = False

  def _forward_modules_ints_helper(self, feats, program, i, j):
    if j < program.size(1):
      fn_idx = program.data[i, j]
      fn_str = self.vocab['program_idx_to_token'][fn_idx]
    else:
      raise IndexError('malformed program')

    if fn_str == '<START>':
      used_fn_j = False
      return self._forward_modules_ints_helper(feats, program, i, j + 1)
    elif fn_str in ['<NULL>', '<END>']:
      used_fn_j = False
      raise ValueError('reached area out of program')

    self.used_fns[i, j] = 1

    j += 1
    if fn_str == 'scene':
      output = feats[i].unsqueeze(0)
    else:
      module_name, input_text = self.vocab['program_token_to_module_text'][fn_str]
      module = self.name_to_module[module_name]
      num_inputs = self.name_to_num_inputs[module_name]
      module_inputs = []
      for _ in range(num_inputs):
        module_input, j = self._forward_modules_ints_helper(feats, program, i, j)
        module_inputs.append(module_input)

      output = module(*module_inputs)

    return output, j
