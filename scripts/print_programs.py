import h5py
import sys
import json
import numpy
import time
import timeit
from vr.data import ClevrDataset
from vr.utils import load_vocab
import cProfile, pstats, io


def print_program_tree(program, prefix):
    token = program_vocab[program[0]]
    cur_arity = arity[token]
    print("{}{} {}".format(prefix, token, str(cur_arity)))
    if cur_arity == 0:
        return 1
    if cur_arity == 1:
        return 1 + print_program_tree(program[1:], prefix + "  ")
    if cur_arity == 2:
        right_subtree = 1 + print_program_tree(program[1:], prefix + "  ")
        return right_subtree + print_program_tree(program[right_subtree:], prefix + "  ")
    raise ValueError()


f = h5py.File(sys.argv[1])
num = int(sys.argv[2]) if len(sys.argv) > 1 else 10
vocab = load_vocab('vocab.json')
arity = vocab['program_token_arity']
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
        print_program_tree(programs[i], "")
    quest = questions[i]
    print(" ".join(question_vocab[quest[j]] for j in range(len(quest))))
    if 'answers' in f:
        print(answers[i])
