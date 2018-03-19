import h5py
import sys
import json
import numpy
import time
import timeit
from vr.data import ClevrDataset
from vr.utils import load_vocab
import cProfile, pstats, io

f = h5py.File(sys.argv[1])
num = int(sys.argv[2]) if len(sys.argv) > 1 else 10
vocab = load_vocab('vocab.json')
program_vocab = vocab['program_idx_to_token']
question_vocab = vocab['question_idx_to_token']
programs = None
if 'programs' in f:
  programs = f['programs']
questions = f['questions']
if 'answers' in f:
  answers = f['answers']
for i in range(num):
  if programs:
    prog = programs[i]
    print(" ".join('"' + program_vocab[prog[j]] + '"' for j in range(len(prog))))
  quest = questions[i]
  print(" ".join(question_vocab[quest[j]] for j in range(len(quest))))
  if 'answers' in f:
    print(answers[i])
