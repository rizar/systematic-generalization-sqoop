import h5py
import numpy
import argparse
import json
import sexpdata

def create_vocab(questions):
  question_vocab = {'<NULL>': 0, '<START>': 1, '<END>': 2}
  for q in questions:
    for w in q:
      if not w in question_vocab:
        question_vocab[w] = len(question_vocab)
  return question_vocab

# An adaptation of the original code for parsing SHAPES queries

def extract_parse(p):
    if isinstance(p, sexpdata.Symbol):
        return p.value()
    elif isinstance(p, int):
        return str(p)
    elif isinstance(p, bool):
        return str(p).lower()
    elif isinstance(p, float):
        return str(p).lower()
    return tuple(extract_parse(q) for q in p)


def parse_tree(p):
    parsed = sexpdata.loads(p)
    extracted = extract_parse(parsed)
    return extracted


def layout_from_parsing(parse):
    if isinstance(parse, str):
        return ("_Find[{}]".format(parse),)
    head = parse[0]
    if len(parse) > 2:  # fuse multiple tokens with "_And"
        assert(len(parse)) == 3
        below = ("_And", layout_from_parsing(parse[1]),
                 layout_from_parsing(parse[2]))
    else:
        below = layout_from_parsing(parse[1])
    if head == "is":
        module = "_Answer"
    elif head in ["above", "below", "left_of", "right_of"]:
        module = "_Transform[{}]".format(head)
    return (module, below)


def flatten_layout(module_layout):
    # Postorder traversal to generate Reverse Polish Notation (RPN)
    if isinstance(module_layout, str):
        return [module_layout]
    PN = []
    module = module_layout[0]
    PN += [module]
    for m in module_layout[1:]:
        PN += flatten_layout(m)
    return PN


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

  for part, prefix in zip(parts, part_prefixes):
    image_path = "{}/shapes_dataset/{}.input.npy".format(args.shapes_data, prefix)
    images = numpy.load(image_path)

    questions_path = "{}/shapes_dataset/{}.query_str.txt".format(args.shapes_data, prefix)
    with open(questions_path) as src:
      questions = [str_.split() for str_ in src]
    max_question_len = max([len(q) for q in questions])
    if part == 'train':
      question_vocab = create_vocab(questions)
    questions = [[question_vocab[w] for w in q] for q in questions]
    questions_arr = numpy.zeros((len(questions), max_question_len), dtype='int64')
    for row, q in zip(questions_arr, questions):
      row[:][:len(q)] = q

    programs_path = "{}/shapes_dataset/{}.query".format(args.shapes_data, prefix)
    programs = []
    with open(programs_path) as src:
      for line in src:
        program = flatten_layout(layout_from_parsing(parse_tree(line)))
        buffer_ = []
        for token in program:
          buffer_.append(token)
          if token.startswith('_Find'):
            buffer_.append('scene')
        program = ['<START>'] + buffer_ + ['<END>']
        programs.append(program)
    max_program_len = max([len(p) for p in programs])
    if part == 'train':
      program_vocab = create_vocab(programs)
    programs = [[program_vocab[w] for w in p] for p in programs]
    programs_arr = numpy.zeros((len(programs), max_program_len), dtype='int64')
    for row, p in zip(programs_arr, programs):
      row[:][:len(p)] = p

    answers_path = "{}/shapes_dataset/{}.output".format(args.shapes_data, prefix)
    with open(answers_path) as src:
      answers = [1 if w.strip() == 'true' else 0 for w in src]

    with h5py.File(part + '_features.h5', 'w') as dst:
      features = images.transpose(0, 3, 1, 2) / 255.0
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
                'program_token_arity':
                  {name: (1 if name != 'scene' else 0) for name in program_vocab},
                'answer_token_to_idx':
                  {'false': 0, 'true': 1}},
              dst)

if __name__ == '__main__':
  main()
