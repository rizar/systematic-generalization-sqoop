import numpy
import pygame
import argparse
import h5py
import pygame
import json


COLOR2RGB = {'red': (255, 0, 0),
             'green': (0, 255, 0),
             'blue': (0, 0, 255),
             'yellow': (255, 255, 0),
             'cyan': (0, 255, 255),
             'purple': (128, 0, 128),
             'brown': (165, 42, 42),
             'gray': (128, 128, 128)}


SHAPES = ['square', 'circle', 'triangle']


def draw_object(shape, color, surf):
  width, height = surf.get_size()
  if width != height:
    raise ValueError("can only draw on square cells")
  if width < 8:
    raise ValueError("too small, can't draw")
  rgb = COLOR2RGB[color]
  if shape == 'square':
    pygame.draw.rect(surf, rgb, pygame.Rect(2, 2, width - 4, height - 4))
  elif shape == 'circle':
    pygame.draw.circle(surf, rgb, (width // 2, height // 2), width // 4 + 1)
  elif shape == 'triangle':
    polygon = [(width // 4, height // 4),
        (width // 2, height - height // 4),
        (width - width // 4, height // 4)]
    pygame.draw.polygon(surf, rgb, polygon)
  else:
    raise ValueError()


def get_object_bitmap(shape, color, size):
  surf = pygame.Surface((size, size))
  draw_object(shape, color, surf)
  return surf


def surf2array(surf):
  arr = pygame.surfarray.array3d(surf)
  arr = arr.transpose(1, 0, 2)
  arr = arr[::-1, :, :]
  return arr


class SceneGenerator:

  def __init__(self, grid_size, cell_size, num_objects, seed):
    self._grid_size = grid_size
    self._cell_size = cell_size
    self._num_objects = num_objects
    self._seed = seed
    self._rng = numpy.random.RandomState(seed)

  def __iter__(self):
    self._rng = numpy.random.RandomState(self._seed)
    return self

  def __next__(self):
    return self.generate_scene()

  def generate_scene(self):
    surface = pygame.Surface((self._grid_size * self._cell_size,
    self._grid_size * self._cell_size))

    objects = []
    positions = set()
    while len(objects) < self._num_objects:
      i = self._rng.randint(self._grid_size)
      j = self._rng.randint(self._grid_size)
      if (i, j) in positions:
        continue
      shape = self._rng.choice(SHAPES)
      color = self._rng.choice(list(COLOR2RGB.keys()))
      objects.append(((i, j), shape, color))
      positions.add((i, j))

    for (i, j), shape, color in objects:
      bitmap = get_object_bitmap(shape, color, self._cell_size)
      surface.blit(source=bitmap, dest=(self._cell_size * i, self._cell_size * j))

    return objects, surface


def generate_dataset(prefix, size, seed):
  sg = SceneGenerator(grid_size=5, cell_size=10,
                      num_objects=5, seed=1)

  # generate images
  scenes = []
  with h5py.File(prefix + '_features.h5', 'w') as dst:
    features_dataset = dst.create_dataset(
      'features', (size, 3, 50, 50), dtype=numpy.float32)
    for i, (scene, surface) in enumerate(sg):
      if i == size:
        break
      features_dataset[i] = surf2array(surface).transpose(2, 0, 1) / 255.0
      scenes.append(scene)

  max_question_len = 5
  max_program_len = 7

  question_words = (['<NULL>', '<START>', '<END>', 'is', 'there', 'a']
                    + sorted(list(COLOR2RGB))
                    + SHAPES)
  question_vocab = {word: i for i, word in enumerate(question_words)}

  program_words = (['<NULL>', '<START>', '<END>', 'scene', 'And']
                   + sorted(list(COLOR2RGB))
                   + SHAPES)
  program_vocab = {word: i for i, word in enumerate(program_words)}

  # generate questions
  with h5py.File(prefix + '_questions.h5', 'w') as dst:
    questions_dataset = dst.create_dataset(
      'questions', (size, max_question_len), dtype=numpy.int64)
    programs_dataset = dst.create_dataset(
      'programs', (size, max_program_len), dtype=numpy.int64)
    answers_dataset = dst.create_dataset(
      'answers', (size,), dtype=numpy.int64)
    image_idxs_dataset = dst.create_dataset(
      'image_idxs', (size,), dtype=numpy.int64)

    rng = numpy.random.RandomState(seed)
    for i, scene in enumerate(scenes):
      answer = rng.choice(2)
      if answer == 1:
        _, shape, color = scene[rng.randint(len(scene))]
      else:
        # sample a (shape, color) pair that is not present in the picture
        while True:
          shape = rng.choice(SHAPES)
          color = rng.choice(list(COLOR2RGB.keys()))
          found = any((shape, color) == (obj_shape, obj_color)
                      for _, obj_shape, obj_color in scene)
          if not found:
            break

      question = ["is", "there", "a"] + [color, shape]
      program = ['<START>', 'And', shape, 'scene', color, 'scene', '<END>']

      questions_dataset[i] = [question_vocab[w] for w in question]
      programs_dataset[i] = [program_vocab[w] for w in program]
      answers_dataset[i] = int(answer)
      image_idxs_dataset[i] = i

  def arity(token):
    if token == 'And':
      return 2
    elif token == 'scene':
      return 0
    else:
      return 1
  with open('vocab.json', 'w') as dst:
    json.dump({'question_token_to_idx': question_vocab,
               'program_token_to_idx': program_vocab,
               'program_token_arity':
                  {name: arity(name) for name in program_vocab},
                'answer_token_to_idx':
                  {'false': 0, 'true': 1}},
              dst)


def main():
  generate_dataset('train', args.train, 1)
  generate_dataset('val', args.val, 2)
  generate_dataset('test', args.test, 3)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', type=int, default=1000,
                      help="Size of the training set")
  parser.add_argument('--val', type=int, default=100,
                      help="Size of the development set")
  parser.add_argument('--test', type=int, default=100,
                      help="Size of the test set")
  args = parser.parse_args()
  main()
