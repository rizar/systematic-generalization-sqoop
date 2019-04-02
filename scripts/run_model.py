# Copyright 2019-present, Mila
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
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
from vr.models import *


parser = argparse.ArgumentParser()
parser.add_argument('--program_generator', default=None)
parser.add_argument('--execution_engine', default=None)
parser.add_argument('--baseline_model', default=None)
parser.add_argument('--debug_every', default=float('inf'), type=float)
parser.add_argument('--use_gpu', default=torch.cuda.is_available(), type=int)

# For running on a preprocessed dataset
parser.add_argument('--data_dir', default=None, type=str)
parser.add_argument('--part', default='val', type=str)

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
    if not args.program_generator:
        args.program_generator = args.execution_engine
    input_question_h5 = os.path.join(args.data_dir, '{}_questions.h5'.format(args.part))
    input_features_h5 = os.path.join(args.data_dir, '{}_features.h5'.format(args.part))

    model = None
    if args.baseline_model is not None:
        print('Loading baseline model from ', args.baseline_model)
        model, _ = utils.load_baseline(args.baseline_model)
        if args.vocab_json is not None:
            new_vocab = utils.load_vocab(args.vocab_json)
            model.rnn.expand_vocab(new_vocab['question_token_to_idx'])
    elif args.program_generator is not None and args.execution_engine is not None:
        pg, _ = utils.load_program_generator(args.program_generator)
        ee, _ = utils.load_execution_engine(
            args.execution_engine, verbose=False)
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
    else:
        vocab = load_vocab(args)
        loader_kwargs = {
          'question_h5': input_question_h5,
          'feature_h5': input_features_h5,
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


def run_our_model_batch(args, pg, ee, loader, dtype):
    if pg:
        pg.type(dtype)
        pg.eval()
    ee.type(dtype)
    ee.eval()

    all_scores = []
    all_programs = []
    all_correct = []
    all_probs = []
    all_preds = []
    all_film_scores = []
    all_read_scores = []
    all_control_scores = []
    all_connections = []
    all_vib_costs = []
    num_correct, num_samples = 0, 0

    q_types = []

    start = time.time()
    for batch in tqdm(loader):
        assert(not pg or not pg.training)
        assert(not ee.training)
        questions, images, feats, answers, programs, program_lists = batch

        if isinstance(questions, list):
            questions_var = questions[0].type(dtype).long()
            q_types += [questions[1].cpu().numpy()]
        else:
            questions_var = questions.type(dtype).long()
        feats_var = feats.type(dtype)
        if pg:
            programs_pred = pg(questions_var)
        else:
            programs_pred = programs

        kwargs = ({'save_activations': True}
                  if isinstance(ee, (FiLMedNet, ModuleNet, MAC))
                  else {})
        pos_args = [feats_var]
        if isinstance(ee, SHNMN):
            pos_args.append(questions_var)
        else:
            pos_args.append(programs_pred)
        scores = ee(*pos_args, **kwargs)
        probs = F.softmax(scores, dim=1)

        #loss = torch.nn.CrossEntropyLoss()(scores, answers.cuda())
        #loss.backward()

        #for i, output in enumerate(ee.stem.outputs):
        #  print('module_{}:'.format(i), output.mean().item(),
        #        ((output ** 2).mean() ** 0.5).item(),
        #        output.min().item(),
        #        output.max().item())

        _, preds = scores.data.cpu().max(1)
        # all_programs.append(programs_pred.data.cpu().clone())
        all_scores.append(scores.data.cpu().clone())
        all_probs.append(probs.data.cpu().clone())
        all_preds.append(preds.cpu().clone())
        all_correct.append(preds == answers)
        if isinstance(pg, FiLMGen) and pg.scores is not None:
            all_film_scores.append(pg.scores.data.cpu().clone())
        if isinstance(ee, MAC):
            all_control_scores.append(ee.control_scores.data.cpu().clone())
            all_read_scores.append(ee.read_scores.data.cpu().clone())
        if hasattr(ee, 'vib_costs'):
            all_vib_costs.append(ee.vib_costs.data.cpu().clone())
        if hasattr(ee, 'connections') and ee.connections:
            all_connections.append(torch.cat([conn.unsqueeze(1) for conn in ee.connections], 1).data.cpu().clone())
        if answers[0] is not None:
            num_correct += (preds == answers).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d = %.2f correct' % (num_correct, num_samples, 100 * acc))
    print('%.2fs to evaluate' % (start - time.time()))
    if all_control_scores:
        max_len = max(cs.size(2) for cs in all_control_scores)
        for i in range(len(all_control_scores)):
            tmp = torch.zeros(
                (all_control_scores[i].size(0), all_control_scores[i].size(1), max_len))
            tmp[:, :, :all_control_scores[i].size(2)] = all_control_scores[i]
            all_control_scores[i] = tmp

    output_path = ('output_' + args.execution_engine[:-3] + ".h5"
                   if not args.output_h5
                   else args.output_h5)

    print('Writing output to "%s"' % output_path)
    with h5py.File(output_path, 'w') as fout:
        fout.create_dataset('scores', data=torch.cat(all_scores, 0).numpy())
        fout.create_dataset('probs', data=torch.cat(all_probs, 0).numpy())
        fout.create_dataset('correct', data=torch.cat(all_correct, 0).numpy())
        if all_film_scores:
            fout.create_dataset('film_scores', data=torch.cat(all_film_scores, 1).numpy())
        if all_vib_costs:
            fout.create_dataset('vib_costs', data=torch.cat(all_vib_costs, 0).numpy())
        if all_read_scores:
            fout.create_dataset('read_scores', data=torch.cat(all_read_scores, 0).numpy())
        if all_control_scores:
            fout.create_dataset('control_scores', data=torch.cat(all_control_scores, 0).numpy())
        if all_connections:
            fout.create_dataset('connections', data=torch.cat(all_connections, 0).numpy())

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
    f_map = (features ** 2).mean(0, keepdim=True).mean(1, keepdim=True).squeeze().sqrt()
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
