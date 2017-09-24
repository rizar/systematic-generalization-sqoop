# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ipdb as pdb
import json
import random
import shutil
from termcolor import colored
import time
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import numpy as np
import h5py
from scipy.misc import imread, imresize, imsave

import vr.utils as utils
import vr.programs
from vr.data import ClevrDataset, ClevrDataLoader
from vr.preprocess import tokenize, encode


parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default='models/best.pt')
parser.add_argument('--execution_engine', default='models/best.pt')
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--model_type', default='FiLM')
parser.add_argument('--debug_every', default=float('inf'), type=float)
parser.add_argument('--use_gpu', default=1, type=int)

# For running on a preprocessed dataset
parser.add_argument('--input_question_h5', default=None)
parser.add_argument('--input_features_h5', default=None)

# This will override the vocab stored in the checkpoint;
# we need this to run CLEVR models on human data
parser.add_argument('--vocab_json', default=None)

# For running on a single example
parser.add_argument('--question', default=None)
parser.add_argument('--image', default='img/CLEVR_val_000017.png')
parser.add_argument('--cnn_model', default='resnet101')
parser.add_argument('--cnn_model_stage', default=3, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--enforce_clevr_vocab', default=1, type=int)

parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_samples', default=None, type=int)
parser.add_argument('--num_last_words_shuffled', default=0, type=int)  # -1 for all shuffled
parser.add_argument('--family_split_file', default=None)

parser.add_argument('--sample_argmax', type=int, default=1)
parser.add_argument('--temperature', default=1.0, type=float)

# FiLM models only
parser.add_argument('--gamma_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp', 'relu', 'softplus'])
parser.add_argument('--gamma_scale', default=1, type=float)
parser.add_argument('--gamma_shift', default=0, type=float)
parser.add_argument('--gammas_from', default=None)  # Load gammas from file
parser.add_argument('--beta_option', default='linear',
  choices=['linear', 'sigmoid', 'tanh', 'exp', 'relu', 'softplus'])
parser.add_argument('--beta_scale', default=1, type=float)
parser.add_argument('--beta_shift', default=0, type=float)
parser.add_argument('--betas_from', default=None)  # Load betas from file

# If this is passed, then save all predictions to this file
parser.add_argument('--output_h5', default=None)
parser.add_argument('--output_preds', default=None)
parser.add_argument('--output_viz_dir', default='img/')
parser.add_argument('--output_program_stats_dir', default=None)

grads = {}
programs = {}  # NOTE: Useful for zero-shot program manipulation when in debug mode

def main(args):
  if args.debug_every <= 1:
    pdb.set_trace()
  model = None
  if args.baseline_model is not None:
    print('Loading baseline model from ', args.baseline_model)
    model, _ = utils.load_baseline(args.baseline_model)
    if args.vocab_json is not None:
      new_vocab = utils.load_vocab(args.vocab_json)
      model.rnn.expand_vocab(new_vocab['question_token_to_idx'])
  elif args.program_generator is not None and args.execution_engine is not None:
    pg, _ = utils.load_program_generator(args.program_generator, args.model_type)
    ee, _ = utils.load_execution_engine(
      args.execution_engine, verbose=False, model_type=args.model_type)
    if args.vocab_json is not None:
      new_vocab = utils.load_vocab(args.vocab_json)
      pg.expand_encoder_vocab(new_vocab['question_token_to_idx'])
    model = (pg, ee)
  else:
    print('Must give either --baseline_model or --program_generator and --execution_engine')
    return

  dtype = torch.FloatTensor
  if args.use_gpu == 1:
    dtype = torch.cuda.FloatTensor
  if args.question is not None and args.image is not None:
    run_single_example(args, model, dtype, args.question)
  # Interactive mode
  elif args.image is not None and args.input_question_h5 is None and args.input_features_h5 is None:
    feats_var = extract_image_features(args, dtype)
    print(colored('Ask me something!', 'cyan'))
    while True:
      # Get user question
      question_raw = input(">>> ")
      run_single_example(args, model, dtype, question_raw, feats_var)
  else:
    vocab = load_vocab(args)
    loader_kwargs = {
      'question_h5': args.input_question_h5,
      'feature_h5': args.input_features_h5,
      'vocab': vocab,
      'batch_size': args.batch_size,
    }
    if args.num_samples is not None and args.num_samples > 0:
      loader_kwargs['max_samples'] = args.num_samples
    if args.family_split_file is not None:
      with open(args.family_split_file, 'r') as f:
        loader_kwargs['question_families'] = json.load(f)
    with ClevrDataLoader(**loader_kwargs) as loader:
      run_batch(args, model, dtype, loader)


def extract_image_features(args, dtype):
  # Build the CNN to use for feature extraction
  print('Extracting image features...')
  cnn = build_cnn(args, dtype)

  # Load and preprocess the image
  img_size = (args.image_height, args.image_width)
  img = imread(args.image, mode='RGB')
  img = imresize(img, img_size, interp='bicubic')
  img = img.transpose(2, 0, 1)[None]
  mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
  std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
  img = (img.astype(np.float32) / 255.0 - mean) / std

  # Use CNN to extract features for the image
  img_var = Variable(torch.FloatTensor(img).type(dtype), volatile=False, requires_grad=True)
  feats_var = cnn(img_var)
  return feats_var


def run_single_example(args, model, dtype, question_raw, feats_var=None):
  interactive = feats_var is not None
  if not interactive:
    feats_var = extract_image_features(args, dtype)

  # Tokenize the question
  vocab = load_vocab(args)
  question_tokens = tokenize(question_raw,
                      punct_to_keep=[';', ','],
                      punct_to_remove=['?', '.'])
  if args.enforce_clevr_vocab == 1:
    for word in question_tokens:
      if word not in vocab['question_token_to_idx']:
        print(colored('No one taught me what "%s" means :( Try me again!' % (word), 'magenta'))
        return
  question_encoded = encode(question_tokens,
                       vocab['question_token_to_idx'],
                       allow_unk=True)
  question_encoded = torch.LongTensor(question_encoded).view(1, -1)
  question_encoded = question_encoded.type(dtype).long()
  question_var = Variable(question_encoded, volatile=False)

  # Run the model
  scores = None
  predicted_program = None
  if type(model) is tuple:
    pg, ee = model
    pg.type(dtype)
    pg.eval()
    ee.type(dtype)
    ee.eval()
    if args.model_type == 'FiLM':
      predicted_program = pg(question_var)
    else:
      predicted_program = pg.reinforce_sample(
                            question_var,
                            temperature=args.temperature,
                            argmax=(args.sample_argmax == 1))
    programs[question_raw] = predicted_program
    if args.debug_every <= -1:
      pdb.set_trace()
    scores = ee(feats_var, predicted_program, save_activations=True)
  else:
    model.type(dtype)
    scores = model(question_var, feats_var)

  # Print results
  predicted_probs = scores.data.cpu()
  _, predicted_answer_idx = predicted_probs[0].max(dim=0)
  predicted_probs = F.softmax(Variable(predicted_probs[0])).data
  predicted_answer = vocab['answer_idx_to_token'][predicted_answer_idx[0]]

  answers_to_probs = {}
  for i in range(len(vocab['answer_idx_to_token'])):
    answers_to_probs[vocab['answer_idx_to_token'][i]] = predicted_probs[i]
  answers_to_probs_sorted = sorted(answers_to_probs.items(), key=lambda x: x[1])
  answers_to_probs_sorted.reverse()
  for i in range(len(answers_to_probs_sorted)):
    if answers_to_probs_sorted[i][1] >= 1e-3 and args.debug_every < float('inf'):
      print("%s: %.1f%%" % (answers_to_probs_sorted[i][0].capitalize(),
        100 * answers_to_probs_sorted[i][1]))

  if not interactive:
    print(colored('Question: "%s"' % question_raw, 'cyan'))
  print(colored(str(predicted_answer).capitalize(), 'magenta'))

  if interactive:
    return

  # Visualize Gradients w.r.t. output
  cf_conv = ee.classifier[0](ee.cf_input)
  cf_bn = ee.classifier[1](cf_conv)
  pre_pool = ee.classifier[2](cf_bn)
  pooled = ee.classifier[3](pre_pool)

  pre_pool_max_per_c = pre_pool.max(2)[0].max(3)[0].expand_as(pre_pool)
  pre_pool_masked = (pre_pool_max_per_c == pre_pool).float() * pre_pool
  pool_feat_locs = (pre_pool_masked > 0).float().sum(1)
  if args.debug_every <= 1:
    pdb.set_trace()

  if args.output_viz_dir != 'NA':
    viz_dir = args.output_viz_dir + question_raw + ' ' + predicted_answer
    if not os.path.isdir(viz_dir):
      os.mkdir(viz_dir)
    args.viz_dir = viz_dir
    print('Saving visualizations to ' + args.viz_dir)

    # Backprop w.r.t. sum of output scores - What affected prediction most?
    ee.feats.register_hook(save_grad('stem'))
    for i in range(ee.num_modules):
      ee.module_outputs[i].register_hook(save_grad('m' + str(i)))
    scores_sum = scores.sum()
    scores_sum.backward()

    # Visualizations!
    visualize(feats_var, args, 'resnet101')
    visualize(ee.feats, args, 'conv-stem')
    visualize(grads['stem'], args, 'grad-conv-stem')
    for i in range(ee.num_modules):
      visualize(ee.module_outputs[i], args, 'resblock' + str(i))
      visualize(grads['m' + str(i)], args, 'grad-resblock' + str(i))
    visualize(pre_pool, args, 'pre-pool')
    visualize(pool_feat_locs, args, 'pool-feature-locations')

  if (predicted_program is not None) and (args.model_type != 'FiLM'):
    print()
    print('Predicted program:')
    program = predicted_program.data.cpu()[0]
    num_inputs = 1
    for fn_idx in program:
      fn_str = vocab['program_idx_to_token'][fn_idx]
      num_inputs += vr.programs.get_num_inputs(fn_str) - 1
      print(fn_str)
      if num_inputs == 0:
        break


def run_our_model_batch(args, pg, ee, loader, dtype):
  pg.type(dtype)
  pg.eval()
  ee.type(dtype)
  ee.eval()

  all_scores, all_programs = [], []
  all_probs = []
  all_preds = []
  num_correct, num_samples = 0, 0

  loaded_gammas = None
  loaded_betas = None
  if args.gammas_from:
    print('Loading ')
    loaded_gammas = torch.load(args.gammas_from)
  if args.betas_from:
    print('Betas loaded!')
    loaded_betas = torch.load(args.betas_from)

  q_types = []
  film_params = []

  if args.num_last_words_shuffled == -1:
    print('All words of each question shuffled.')
  elif args.num_last_words_shuffled > 0:
    print('Last %d words of each question shuffled.' % args.num_last_words_shuffled)
  start = time.time()
  for batch in tqdm(loader):
    assert(not pg.training)
    assert(not ee.training)
    questions, images, feats, answers, programs, program_lists = batch

    if args.num_last_words_shuffled != 0:
      for i, question in enumerate(questions):
        # Search for <END> token to find question length
        q_end = get_index(question.numpy().tolist(), index=2, default=len(question))
        if args.num_last_words_shuffled > 0:
          q_end -= args.num_last_words_shuffled  # Leave last few words unshuffled
        if q_end < 2:
          q_end = 2
        question = question[1:q_end]
        random.shuffle(question)
        questions[i][1:q_end] = question

    if isinstance(questions, list):
      questions_var = Variable(questions[0].type(dtype).long(), volatile=True)
      q_types += [questions[1].cpu().numpy()]
    else:
      questions_var = Variable(questions.type(dtype).long(), volatile=True)
    feats_var = Variable(feats.type(dtype), volatile=True)
    if args.model_type == 'FiLM':
      programs_pred = pg(questions_var)
      # Examine effect of various conditioning modifications at test time!
      programs_pred = pg.modify_output(programs_pred, gamma_option=args.gamma_option,
                                       gamma_scale=args.gamma_scale, gamma_shift=args.gamma_shift,
                                       beta_option=args.beta_option, beta_scale=args.beta_scale,
                                       beta_shift=args.beta_shift)
      if args.gammas_from:
        programs_pred[:,:,:pg.module_dim] = loaded_gammas.expand_as(
          programs_pred[:,:,:pg.module_dim])
      if args.betas_from:
        programs_pred[:,:,pg.module_dim:2*pg.module_dim] = loaded_betas.expand_as(
          programs_pred[:,:,pg.module_dim:2*pg.module_dim])
    else:
      programs_pred = pg.reinforce_sample(
                        questions_var,
                        temperature=args.temperature,
                        argmax=(args.sample_argmax == 1))

    film_params += [programs_pred.cpu().data.numpy()]
    scores = ee(feats_var, programs_pred, save_activations=True)
    probs = F.softmax(scores)

    _, preds = scores.data.cpu().max(1)
    all_programs.append(programs_pred.data.cpu().clone())
    all_scores.append(scores.data.cpu().clone())
    all_probs.append(probs.data.cpu().clone())
    all_preds.append(preds.cpu().clone())
    if answers[0] is not None:
      num_correct += (preds == answers).sum()
    num_samples += preds.size(0)

  acc = float(num_correct) / num_samples
  print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
  print('%.2fs to evaluate' % (start - time.time()))
  all_programs = torch.cat(all_programs, 0)
  all_scores = torch.cat(all_scores, 0)
  all_probs = torch.cat(all_probs, 0)
  all_preds = torch.cat(all_preds, 0).squeeze().numpy()
  if args.output_h5 is not None:
    print('Writing output to "%s"' % args.output_h5)
    with h5py.File(args.output_h5, 'w') as fout:
      fout.create_dataset('scores', data=all_scores.numpy())
      fout.create_dataset('probs', data=all_probs.numpy())
      fout.create_dataset('predicted_programs', data=all_programs.numpy())

  # Save FiLM params
  np.save('film_params', np.vstack(film_params))
  if isinstance(questions, list):
    np.save('q_types', np.vstack(q_types))

  # Save FiLM param stats
  if args.output_program_stats_dir:
    if not os.path.isdir(args.output_program_stats_dir):
      os.mkdir(args.output_program_stats_dir)
    gammas = all_programs[:,:,:pg.module_dim]
    betas = all_programs[:,:,pg.module_dim:2*pg.module_dim]
    gamma_means = gammas.mean(0)
    torch.save(gamma_means, os.path.join(args.output_program_stats_dir, 'gamma_means'))
    beta_means = betas.mean(0)
    torch.save(beta_means, os.path.join(args.output_program_stats_dir, 'beta_means'))
    gamma_medians = gammas.median(0)[0]
    torch.save(gamma_medians, os.path.join(args.output_program_stats_dir, 'gamma_medians'))
    beta_medians = betas.median(0)[0]
    torch.save(beta_medians, os.path.join(args.output_program_stats_dir, 'beta_medians'))

    # Note: Takes O(10GB) space
    torch.save(gammas, os.path.join(args.output_program_stats_dir, 'gammas'))
    torch.save(betas, os.path.join(args.output_program_stats_dir, 'betas'))

  if args.output_preds is not None:
    vocab = load_vocab(args)
    all_preds_strings = []
    for i in range(len(all_preds)):
      all_preds_strings.append(vocab['answer_idx_to_token'][all_preds[i]])
    save_to_file(all_preds_strings, args.output_preds)

  if args.debug_every <= 1:
    pdb.set_trace()
  return


def visualize(features, args, file_name=None):
    """
    Converts a 4d map of features to alpha attention weights,
    According to their 2-Norm across dimensions 0 and 1.
    Then saves the input RGB image as an RGBA image using an upsampling of this attention map.
    """
    save_file = os.path.join(args.viz_dir, file_name)
    img_path = args.image

    # Scale map to [0, 1]
    f_map = (features ** 2).mean(0).mean(1).squeeze().sqrt()
    f_map_shifted = f_map - f_map.min().expand_as(f_map)
    f_map_scaled = f_map_shifted / f_map_shifted.max().expand_as(f_map_shifted)

    if save_file is None:
      print(f_map_scaled)
    else:
      # Read original image
      img = imread(img_path, mode='RGB')
      orig_img_size = img.shape

      # Convert to image format
      alpha = (255 * f_map_scaled).round()
      alpha4d = alpha.unsqueeze(0).unsqueeze(0)
      alpha_upsampled = torch.nn.functional.upsample_bilinear(
        alpha4d, size=torch.Size(orig_img_size)).squeeze(0).transpose(1, 0).transpose(1, 2)
      alpha_upsampled_np = alpha_upsampled.cpu().data.numpy()

      # Create and save visualization
      imga = np.concatenate([img, alpha_upsampled_np], axis=2)
      if save_file[-4:] != '.png': save_file += '.png'
      imsave(save_file, imga)

    return f_map_scaled


def build_cnn(args, dtype):
  if not hasattr(torchvision.models, args.cnn_model):
    raise ValueError('Invalid model "%s"' % args.cnn_model)
  if not 'resnet' in args.cnn_model:
    raise ValueError('Feature extraction only supports ResNets')
  whole_cnn = getattr(torchvision.models, args.cnn_model)(pretrained=True)
  layers = [
    whole_cnn.conv1,
    whole_cnn.bn1,
    whole_cnn.relu,
    whole_cnn.maxpool,
  ]
  for i in range(args.cnn_model_stage):
    name = 'layer%d' % (i + 1)
    layers.append(getattr(whole_cnn, name))
  cnn = torch.nn.Sequential(*layers)
  cnn.type(dtype)
  cnn.eval()
  return cnn


def run_batch(args, model, dtype, loader):
  if type(model) is tuple:
    pg, ee = model
    run_our_model_batch(args, pg, ee, loader, dtype)
  else:
    run_baseline_batch(args, model, loader, dtype)


def run_baseline_batch(args, model, loader, dtype):
  model.type(dtype)
  model.eval()

  all_scores, all_probs = [], []
  num_correct, num_samples = 0, 0
  for batch in loader:
    questions, images, feats, answers, programs, program_lists = batch

    questions_var = Variable(questions.type(dtype).long(), volatile=True)
    feats_var = Variable(feats.type(dtype), volatile=True)
    scores = model(questions_var, feats_var)
    probs = F.softmax(scores)

    _, preds = scores.data.cpu().max(1)
    all_scores.append(scores.data.cpu().clone())
    all_probs.append(probs.data.cpu().clone())

    num_correct += (preds == answers).sum()
    num_samples += preds.size(0)
    print('Ran %d samples' % num_samples)

  acc = float(num_correct) / num_samples
  print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))

  all_scores = torch.cat(all_scores, 0)
  all_probs = torch.cat(all_probs, 0)
  if args.output_h5 is not None:
    print('Writing output to %s' % args.output_h5)
    with h5py.File(args.output_h5, 'w') as fout:
      fout.create_dataset('scores', data=all_scores.numpy())
      fout.create_dataset('probs', data=all_probs.numpy())


def load_vocab(args):
  path = None
  if args.baseline_model is not None:
    path = args.baseline_model
  elif args.program_generator is not None:
    path = args.program_generator
  elif args.execution_engine is not None:
    path = args.execution_engine
  return utils.load_cpu(path)['vocab']


def save_grad(name):
  def hook(grad):
    grads[name] = grad
  return hook


def save_to_file(text, filename):
  with open(filename, mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(text))
    myfile.write('\n')


def get_index(l, index, default=-1):
  try:
    return l.index(index)
  except ValueError:
    return default


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
