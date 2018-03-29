import torch
import torch.nn as nn

from vr.models.module_net import ModuleNet

# TODO(mnoukhov)
# text arg to idx
# module to module name?
# passing text as input to modules

class Find(nn.Module):
  # Input:
  #   image_feat_grid: [N, D_im, H, W]
  #   text_param: [N, D_txt]
  # Output:
  #   image_att: [N, 1, H, W]
  def __init__(self, image_dim, text_dim, map_dim=500):
    super().__init__()
    self.conv2d = nn.Conv2d(image_dim, self.map_dim, 1)
    self.linear = nn.Linear(text_dim, self.map_dim)
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
               input_dim=(1024, 14, 14),
               stem_num_layers=2,
               stem_batchnorm=False,
               image_dim=128,
               batchnorm=False,
               verbose=True):
    super(ModuleNet, self).__init__()

    input_C, input_H, input_W = input_dim
    self.stem = build_stem(input_C,
                            image_dim,
                            num_layers=stem_num_layers,
                            with_batchnorm=stem_batchnorm)
    if verbose:
      print('Here is my stem:')
      print(self.stem)


    if verbose:
      print('Here is my classifier:')
      print(self.classifier)

    self.stem_times = []
    self.module_times = []
    self.classifier_times = []
    self.timing = False

    self.vocab = vocab
    self.num_answer = len(vocab['answer_idx_to_token'])
    self.text_dim = len(vocab['text_arg_to_idx'])

    self.function_modules = {
      'and': And(),
      'answer': Answer(self.num_answer),
      'find': Find(image_dim, self.text_dim),
      'transform': Transform(self.text_dim),
    }

    # hard-coding the number of image/att inputs
    self.function_modules_num_inputs = {
      'and': 2,
      'answer': 1,
      'find': 1,
      'transform': 1,
    }

    for name, module in function_modules:
      self.add_module(fn_str, mod)

    self.save_module_outputs = False

  def _forward_modules_ints_helper(self, feats, program, i, j):
    used_fn_j = True

    if j < program.size(1):
      fn_idx = program.data[i, j]
      fn_str = self.vocab['program_idx_to_token'][fn_idx]
    else:
      used_fn_j = False
      fn_str = 'scene'

    if fn_str == '<NULL>':
      used_fn_j = False
      fn_str = 'scene'
    elif fn_str == '<START>':
      used_fn_j = False
      return self._forward_modules_ints_helper(feats, program, i, j + 1)

    if used_fn_j:
      self.used_fns[i, j] = 1

    j += 1
    module = self.function_modules[fn_str]

    if fn_str == 'scene':
      module_inputs = [feats[i:i+1]]
    else:
      module_name = self.
      num_inputs = self.function_modules_num_inputs[module_name]
      module_inputs = []
      while len(module_inputs) < num_inputs:
        cur_input, j = self._forward_modules_ints_helper(feats, program, i, j)
        module_inputs.append(cur_input)

    module_output = module(*module_inputs)
    return module_output, j

  def _forward_modules_ints(self, feats, program):
    """
    feats: FloatTensor of shape (N, C, H, W) giving features for each image
    program: LongTensor of shape (N, L) giving a prefix-encoded program for
      each image.
    """
    N = feats.size(0)
    final_module_outputs = []
    self.used_fns = torch.Tensor(program.size()).fill_(0)
    for i in range(N):
      cur_output, _ = self._forward_modules_ints_helper(feats, program, i, 0)
      final_module_outputs.append(cur_output)
    self.used_fns = self.used_fns.type_as(program.data).float()
    final_module_outputs = torch.cat(final_module_outputs, 0)
    return final_module_outputs

  def forward(self, x, program):
    N = x.size(0)
    assert N == len(program)

    feats = self.stem(x)

    if type(program) is list or type(program) is tuple:
      final_module_outputs = self._forward_modules_json(feats, program)
    elif type(program) is Variable and program.dim() == 2:
      final_module_outputs = self._forward_modules_ints(feats, program)
    elif torch.is_tensor(program) and program.dim() == 3:
      final_module_outputs = self._forward_modules_probs(feats, program)
    else:
      raise ValueError('Unrecognized program format')

    # After running modules for each input, concatenat the outputs from the
    # final module and run the classifier.
    out = self.classifier(final_module_outputs)
    return out
