import h5py
import numpy
import os
import argparse
import json

def create_vocab(questions):
  question_vocab = {'<NULL>': 0, '<START>': 1, '<END>': 2}
  for q in questions:
    for w in q:
      if not w in question_vocab:
        question_vocab[w] = len(question_vocab)
  return question_vocab

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--shapes_data', type=str,
                      help="Path to the SHAPES dataset")
  parser.add_argument('--size', type=str,
                      help="Which version of the training set to use")
  parser.add_argument('--programs', type=str, choices=['none', 'chain', 'tree'],
                      help="Which programs to put in the resulting HDF5")

  args = parser.parse_args()
  parts = ['train', 'val', 'test']
  part_prefixes = ['train.' + args.size, 'val', 'test']
  part_prefixes = [os.path.join(args.shapes_data, prefix)
                   for prefix in part_prefixes]

  for part, prefix in zip(parts, part_prefixes):
    image_path = prefix + '.input.npy'
    images = numpy.load(image_path)

    questions_path = prefix + '.query_str.txt'
    with open(questions_path) as src:
      questions = [str_.split() for str_ in src]
    max_question_len = max([len(q) for q in questions])
    question_vocab = create_vocab(questions)
    questions = [[question_vocab[w] for w in q] for q in questions]
    questions_arr = numpy.zeros((len(questions), max_question_len), dtype='int64')
    for row, q in zip(questions_arr, questions):
      row[:][:len(q)] = q

    # The parentheses in programs seem to denote the tree structure,
    # but since it is always linear, they can just be discarded
    programs_path = prefix + '.query'
    with open(programs_path) as src:
      programs = []
      for line_ in src:
        line_ = line_.replace('(', ' ').replace(')', ' ')
        programs.append([w for w in line_.split() if w])
    programs = [['<START>'] + program + ['<END>'] for program in programs]
    max_program_len = max([len(p) for p in programs])
    program_vocab = create_vocab(programs)
    programs = [[program_vocab[w] for w in p] for p in programs]
    programs_arr = numpy.zeros((len(programs), max_program_len), dtype='int64')
    for row, p in zip(programs_arr, programs):
      row[:][:len(p)] = p

    answers_path = prefix + '.output'
    with open(answers_path) as src:
      answers = [1 if w.strip() == 'true' else 0 for w in src]

    with h5py.File(part + '_features.h5', 'w') as dst:
      features = images.transpose(0, 3, 1, 2)
      features_dataset = dst.create_dataset(
        'features', (features.shape), dtype=numpy.float32)
      features_dataset[:] = features
    with h5py.File(part + '_questions.h5', 'w') as dst:
      questions_dataset = dst.create_dataset(
        'questions', (len(questions), max_question_len), dtype=numpy.int64)
      questions_dataset[:] = questions_arr
      if args.programs == 'chain':
        programs_dataset = dst.create_dataset(
          'programs', (len(programs), max_program_len), dtype=numpy.int64)
        programs_dataset[:] = programs_arr
      answers_dataset = dst.create_dataset(
        'answers', (len(questions),), dtype=numpy.int64)
      answers_dataset[:] = answers
      image_idxs_dataset = dst.create_dataset(
        'image_idxs', (len(questions),), dtype=numpy.int64)
      image_idxs_dataset[:] = range(len(questions))

    with open('vocab.json', 'w') as dst:
      json.dump({'question_token_to_idx': question_vocab,
                 'program_token_to_idx': program_vocab,
                 'answer_token_to_idx': {'false': 0, 'true': 1}},
                dst)

if __name__ == '__main__':
  main()
