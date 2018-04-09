import h5py
import json
import numpy
import time
import timeit
from vr.data import ClevrDataset
from vr.utils import load_vocab
import cProfile, pstats, io

features = h5py.File('data/train_features.h5')
questions = h5py.File('data/train_questions.h5')
vocab = load_vocab('data/vocab.json')
dataset = ClevrDataset(questions, features, vocab)
def batch():
  for i in numpy.random.choice(700000, 1):
    dataset[i]
pr = cProfile.Profile()
pr.enable()
print(timeit.timeit(batch, number=100) / 100)
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
ps.print_stats()
print(s.getvalue())



