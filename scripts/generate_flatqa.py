"""Generates a dataset.

The algorithm:
  GIVEN: a set of allowed object for the current dataset,
         the desired number of examples
  REPEAT:
    - decide if the answer will be yes or no
    - if yes
      - generate a random scene for which the answer can be yes using
        rejection sampling
      - select an allowed object for the generated scene
      - ask a question about it
    - if no
      - generate a random scene
      - generate an allowed (shape, color) pair for which the answer is no
        for the given scene using rejection sampling
      - ask the question about the selected (shape, color) objects

"""
import argparse
import collections
import io
import json
import logging
import math
import time
from functools import partial

import h5py
import numpy
from PIL import Image, ImageDraw


logger = logging.getLogger(__name__)
RELATIONS = ['left_of', 'right_of', 'above', 'below']
COLORS = ['red', 'green', 'blue', 'yellow', 'cyan',
          'purple', 'brown', 'gray']
SHAPES = ['square', 'triangle',  'cross', 'bar',
          'empty_square', 'empty_triangle']#, 'circle']
MIN_OBJECT_SIZE = 8


class Object(object):
  def __init__(self, size, angle, pos=None, shape=None, color=None):
    self.size = size
    self.angle = angle
    angle_rad = angle / 180 * math.pi
    self.rotated_size =  math.ceil(size * (abs(math.sin(angle_rad)) + abs(math.cos(angle_rad))))
    self.pos = pos
    self.shape = shape
    self.color = color

  def overlap(self, other):
    min_dist = (self.rotated_size + other.rotated_size) // 2 + 1
    return (abs(self.pos[0] - other.pos[0]) < min_dist and
            abs(self.pos[1] - other.pos[1]) < min_dist)

  def relate(self, rel, other):
    if rel == 'left_of':
      return self.pos[0] < other.pos[0]
    if rel == 'right_of':
      return self.pos[0] > other.pos[0]
    if rel == 'above':
      return self.pos[1] > other.pos[1]
    if rel == 'below':
      return self.pos[1] < other.pos[1]
    raise ValueError(rel)


class CustomJSONEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Object):
      return {'size': obj.size,
              'rotated_size': obj.rotated_size,
              'angle': obj.angle,
              'pos': obj.pos,
              'shape': obj.shape,
              'color': obj.color}
    else:
      return super().default(obj)


def draw_object(draw, obj):
  if obj.size < MIN_OBJECT_SIZE:
    raise ValueError("too small, can't draw")

  img = Image.new('RGBA', (obj.size, obj.size))
  draw = ImageDraw.Draw(img)
  width, height = (obj.size, obj.size)

  if obj.shape == 'square':
    draw.rectangle([(0, 0), (width, height)], fill=obj.color)
  elif obj.shape == 'circle':
    draw.ellipse([(0, 0), (width, height)], fill=obj.color)
  elif obj.shape == 'triangle':
    polygon = [(0, 0),
               (width // 2, height - 1),
               (width - 1, 0)]
    draw.polygon(polygon, fill=obj.color)
  elif obj.shape == 'empty_triangle':
    polygon = [(0, 0),
               (width // 2, height - 1),
               (width - 1, 0)]
    draw.polygon(polygon, outline=obj.color)
  elif obj.shape == 'empty_square':
    draw.rectangle([(0, 0), (width-1, height-1)], outline=obj.color)
  elif obj.shape == 'cross':
    draw.rectangle([(width // 3, 0), (2 * width // 3, height)], fill=obj.color)
    draw.rectangle([(0, height // 3), (width, 2 * height // 3)], fill=obj.color)
  elif obj.shape == 'bar':
    draw.rectangle([(0, height // 3), (width, 2 * height // 3)], fill=obj.color)
  else:
    raise ValueError()

  return img.rotate(obj.angle, expand=True, resample=Image.LINEAR)


def draw_scene(objects):
  img = Image.new('RGB', (args.image_size, args.image_size))
  draw = ImageDraw.Draw(img)
  for obj in objects:
    obj_img = draw_object(draw, obj)
    obj_pos = (obj.pos[0] - obj_img.size[0] // 2,
               obj.pos[1] - obj_img.size[1] // 2)
    img.paste(obj_img, obj_pos, obj_img)

  return img


def get_random_spot(rng, objects):
  """Get a spot for a new object that does not overlap with existing ones."""
  # then, select the object size
  size = rng.randint(args.min_obj_size, args.max_obj_size + 1)
  angle = rng.randint(0, 360) if args.rotate else 0
  obj = Object(size, angle)

  min_center = obj.rotated_size // 2 + 1
  max_center = args.image_size - obj.rotated_size // 2 - 1
  for attempt in range(10):
    x = rng.randint(min_center, max_center)
    y = rng.randint(min_center, max_center)
    obj.pos = (x, y)

    # make sure there is no overlap between bounding squares
    if (any([abs(obj.pos[0] - other.pos[0]) < 5 for other in objects]) or
        any([abs(obj.pos[1] - other.pos[1]) < 5 for other in objects])):
      continue
    if any([obj.overlap(other) for other in objects]):
      continue
    return obj
  else:
    return None


def shape_module(shape):
  return "Shape[{}]".format(shape)


def binary_shape_module(shape):
  return "Shape2[{}]".format(shape)


def color_module(color):
  return "Color[{}]".format(color)


def binary_color_module(color):
  return "Color2[{}]".format(color)


def relation_module(relation):
  return "Relate[{}]".format(relation)


def unary_relation_module(relation):
  return "Relate1[{}]".format(relation)


def generate_scene(rng, sampler, objects=[], **kwargs):
  orig_objects = objects

  objects = list(orig_objects)
  place_failures = 0
  while len(objects) < args.num_objects:
    # first, select which object to draw by rejection sampling
    shape, color = sampler.sample_shape_color(purpose='generate', **kwargs)

    new_object = get_random_spot(rng, objects)
    if new_object is None:
      place_failures += 1
      if place_failures == 10:
        # reset generation
        objects = list(orig_objects)
        place_failures = 0
      continue

    new_object.shape = shape
    new_object.color = color
    objects.append(new_object)

  return objects


def generate_dataset(prefix, num_examples, seed, sampler, save_vocab=False):
  if args.level == 'shapecolor':
    max_question_len = 5
    max_program_len = 7
  elif args.level == 'relations':
    max_question_len = 8
    if args.program == 'best':
      max_program_len = 12
    elif args.program == 'noand':
      max_program_len = 8
    elif args.program == 'chain':
      max_program_len = 7
    elif args.program == 'chain_shortcut':
      max_program_len = 11

  question_words = (['<NULL>', '<START>', '<END>', 'is', 'there', 'a']
                    + sampler.colors + sampler.shapes + RELATIONS)
  question_vocab = {word: i for i, word in enumerate(question_words)}

  program_words = (['<NULL>', '<START>', '<END>', 'scene', 'And']
                   + [color_module(color) for color in sampler.colors]
                   + [shape_module(shape) for shape in sampler.shapes]
                   + [binary_color_module(color) for color in sampler.colors]
                   + [binary_shape_module(shape) for shape in sampler.shapes]
                   + [relation_module(rel) for rel in RELATIONS]
                   + [unary_relation_module(rel) for rel in RELATIONS])
  program_vocab = {word: i for i, word in enumerate(program_words)}

  answer_token_to_idx = {word: idx for idx, word in
                         enumerate(['false', 'true'])}
  module_token_to_idx = {word: idx for idx, word in
                         enumerate(['find', 'relate', 'and'])}
  program_token_to_module_text = {}
  for color in sampler.colors:
    program_token_to_module_text[color_module(color)] = ['find', color]
  for shape in sampler.shapes:
    program_token_to_module_text[shape_module(shape)] = ['find', shape]
  for rel in RELATIONS:
    program_token_to_module_text[relation_module(rel)] = ['relate', rel]
  program_token_to_module_text['And'] = ('and', 'null')
  for module in ['<START>', '<END>', '<NULL>']:
    program_token_to_module_text[module] = ('null', 'null')

  text_token_to_idx = {}
  for idx, word in enumerate(
      ['null'] + sampler.colors + sampler.shapes + RELATIONS):
    text_token_to_idx[word] = idx

  def arity(token):
    if (token == 'And' or token.startswith('Relate[') 
        or token.startswith('Color2[') or token.startswith('Shape2[')):
      return 2
    elif token == 'scene':
      return 0
    else:
      return 1
  program_token_arity = {name: arity(name) for name in program_vocab},

  rng = numpy.random.RandomState(seed)
  scenes = []

  # generate questions
  before = time.time()
  with h5py.File(prefix + '_questions.h5', 'w') as dst_questions, \
       h5py.File(prefix + '_features.h5', 'w') as dst_features:
    features_dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    features_dataset = dst_features.create_dataset(
      'features', (num_examples,), dtype=features_dtype)
    questions_dataset = dst_questions.create_dataset(
      'questions', (num_examples, max_question_len), dtype=numpy.int64)
    programs_dataset = dst_questions.create_dataset(
      'programs', (num_examples, max_program_len), dtype=numpy.int64)
    answers_dataset = dst_questions.create_dataset(
      'answers', (num_examples,), dtype=numpy.int64)
    image_idxs_dataset = dst_questions.create_dataset(
      'image_idxs', (num_examples,), dtype=numpy.int64)

    # we presample relations because some of them may be harder to sample
    # then others, and we want a balanced dataset
    presampled_relations = [sampler.sample_relation() for i in range(num_examples)]

    rng = numpy.random.RandomState(seed)
    i = 0
    while i < num_examples:
      if i and i % 1000 == 0:
        print(i)

      answer = i % 2

      if args.level == 'shapecolor': # a shape-color question
        shape, color = sampler.sample_shape_color(purpose='ask')
        if answer:
          # first, sample the witness object and place it
          obj = get_random_spot(rng, [])
          obj.shape = shape
          obj.color = color
          # complete the scene by sampling a random object
          scene = generate_scene(rng, sampler, objects=[obj])
        else:
          # sample a scene and check if the answer is no
          # if failed 10 times, try another question
          scene = generate_scene(rng, sampler)
          if any((shape, color) == (obj.shape, obj.color) for obj in scene):
            continue
        question = ["is", "there", "a"] + [color, shape]
        program = ['<START>', 'And', shape_module(shape), 'scene',
                  color_module(color), 'scene', '<END>']
      elif args.level == 'relations':
        rel = presampled_relations[i]
        if answer:
          # first, select two spots for which the relation holds
          obj1 = get_random_spot(rng, [])
          obj2 = get_random_spot(rng, [obj1])
          if not obj2 or not obj1.relate(rel, obj2):
            continue
          # second, select shapes and colors
          shape1, color1 = sampler.sample_shape_color(purpose='ask', relation=rel)
          obj1.shape = shape1
          obj1.color = color1
          shape2, color2 = sampler.sample_shape_color(purpose='ask', relation=rel)
          obj2.shape = shape2
          obj2.color = color2
          # lastly, fill the scene with other objects
          scene = generate_scene(rng, sampler, objects=[obj1, obj2], relation=rel)
        else:
          # first generate a scene
          scene = generate_scene(rng, sampler, relation=rel)

          # Choose a question for which the answer is false. Note, that
          # generating a completely random question might be a bad idea
          # because it will be to easy to detect as false. Hence:
          neg_question_type = rng.choice(['rel', 'lhs', 'rhs'])
          if neg_question_type == 'rel':
            # select two random objects and check that they are not related
            obj1 = scene[rng.randint(len(scene))]
            shape1, color1 = obj1.shape, obj1.color
            obj2 = scene[rng.randint(len(scene))]
            shape2, color2 = obj2.shape, obj2.color
          elif neg_question_type == 'lhs':
            # select an existing rhs
            shape1, color1 = sampler.sample_shape_color(purpose='ask', relation=rel)
            obj2 = scene[rng.randint(len(scene))]
            shape2, color2 = obj2.shape, obj2.color
          elif neg_question_type == 'rhs':
            # select an existing lhs
            obj1 = scene[rng.randint(len(scene))]
            shape1, color1 = obj1.shape, obj1.color
            shape2, color2 = sampler.sample_shape_color(purpose='ask', relation=rel)

          # verify if the answer to the question is False
          relation_holds = False
          for obj1 in scene:
            for obj2 in scene:
              if (obj1 != obj2 and obj1.relate(rel, obj2)
                  and obj1.shape == shape1 and obj1.color == color1
                  and obj2.shape == shape2 and obj2.color == color2):
                relation_holds = True
          if relation_holds:
            continue
        question = ["is", "there", "a",
                    color1, shape1, rel, color2, shape2]
        if args.program == 'best':
          program = ["<START>", relation_module(rel),
                      "And", shape_module(shape1), "scene", color_module(color1), "scene",
                      "And", shape_module(shape2), "scene", color_module(color2), "scene"]
        elif args.program == 'noand':
          program = ["<START>", relation_module(rel),
                     shape_module(shape1), color_module(color1), "scene",
                     shape_module(shape2), color_module(color2), "scene"]
        elif args.program == 'chain':
          program = ["<START>", 
                     shape_module(shape1), color_module(color1), 
                     unary_relation_module(rel),
                     shape_module(shape2), color_module(color2), 
                     "scene"]
        elif args.program == 'chain_shortcut':
          program = ["<START>", 
                     binary_shape_module(shape1), 'scene', binary_color_module(color1), 'scene', 
                     unary_relation_module(rel),
                     binary_shape_module(shape2), 'scene', binary_color_module(color2), 'scene', 'scene']

      scenes.append(scene)
      buffer_ = io.BytesIO()
      image = draw_scene(scene)
      image.save(buffer_, format='png')
      buffer_.seek(0)

      features_dataset[i] = numpy.frombuffer(buffer_.read(), dtype='uint8')
      questions_dataset[i] = [question_vocab[w] for w in question]
      programs_dataset[i] = [program_vocab[w] for w in program]
      answers_dataset[i] = int(answer)
      image_idxs_dataset[i] = i

      i += 1

  print("{} examples per second".format((time.time() - before) / num_examples))

  with open(prefix + '_scenes.json', 'w') as dst:
    json.dump(scenes, dst, indent=2, cls=CustomJSONEncoder)

  if save_vocab:
    with open('vocab.json', 'w') as dst:
      json.dump({'question_token_to_idx': question_vocab,
                 'program_token_to_idx': program_vocab,
                 'program_token_arity': {
                   name: arity(name) for name in program_vocab},
                 'answer_token_to_idx': answer_token_to_idx,
                 'program_token_to_module_text': program_token_to_module_text,
                 'module_token_to_idx': module_token_to_idx,
                 'text_token_to_idx': text_token_to_idx},
                dst, indent=2)


class Sampler:

  def __init__(self, test, seed, colors, shapes):
    self._test = test
    self._rng = numpy.random.RandomState(seed)
    self.colors = colors
    self.shapes = shapes

  def _choose(self, list_like):
    return list_like[self._rng.randint(len(list_like))]

  def _rejection_sample(self, restricted=[]):
    while True:
      shape = self._rng.choice(self.shapes)
      color = self._rng.choice(self.colors)
      if (shape, color) not in restricted:
        return shape, color

  def sample_relation(self, *args, **kwargs):
    return self._choose(RELATIONS)

  def sample_shape_color(self, *args, **kwargs):
    return self._rejection_sample()


class BelowRedSampler(Sampler):

  def sample_relation(self):
    if self._test:
      return 'below'
    else:
      return super().sample_relation()

  def sample_shape_color(self, purpose, relation):
    red_objects = [(shape, 'red') for shape in self.shapes]
    if self._test:
      # only 'below'
      return self._rejection_sample(red_objects)
    else:
      if relation == 'below':
        return self._choose(red_objects)
      return self._rejection_sample()


class BelowSquaresSampler(Sampler):
  """At training time the relation 'below' is only explained with squares.
  At test time we test the generalization of 'below' to other types of objects.
  """

  def sample_relation(self):
    if self._test:
      return 'below'
    else:
      return super().sample_relation()

  def sample_shape_color(self, purpose, relation):
    squares = [('square', color) for color in self.colors]
    if self._test:
      # only 'below'
      return self._rejection_sample(squares)
    else:
      if relation == 'below':
        return self._choose(squares)
      return self._rejection_sample()


class BelowExcludeDiagSampler(Sampler):

  def sample_relation(self):
    if self._test:
      return 'below'
    else:
      return super().sample_relation()

  def sample_shape_color(self, purpose, relation):
    diagonal = list(zip(self.shapes, self.colors))
    if self._test:
      if purpose == 'ask':
        return self._choose(diagonal)
      else:
        return self._rejection_sample()
    else:
      if relation == 'below':
        return self._rejection_sample(diagonal)
      return self._rejection_sample()


class HoldDiagSampler(Sampler):

  def sample_shape_color(self, purpose):
    diagonal = list(zip(self.shapes, self.colors))
    if self._test:
      if purpose == 'ask':
        return self._choose(diagonal)
      else:
        return self._rejection_sample()
    else:
      return self._rejection_sample(diagonal)


class HoldRedSquareSampler(Sampler):

  def sample_shape_color(self, purpose):
    if self._test:
      if purpose == 'ask':
        return ('square', 'red')
      else:
        return super().sample_shape_color()
    else:
      return self._rejection_sample([('square', 'red')])


class CoGenTSampler(Sampler):

  def __init__(self, test, seed, colors, shapes):
    if colors != COLORS:
      raise ValueError("can't do CoGenT with less than 8 colors")
    set1 = ['gray', 'blue', 'brown', 'yellow']
    set2 = ['red', 'green', 'purple', 'cyan']
    self._restrict = ([('square', color) for color in set1] +
                      [('triangle', color) for color in set2])
    self._restrict_test = ([('square', color) for color in set2] +
                            [('triangle', color) for color in set1])
    super().__init__(test, seed, colors, shapes)

  def sample_shape_color(self, purpose):
    if self._test:
      if purpose == 'ask':
        return self._choose(self._restrict)
      return self._rejection_sample(self._restrict_test)
    else:
      return self._rejection_sample(self._restrict)


def main():
  shapes = SHAPES[:args.num_shapes]
  colors = COLORS[:args.num_colors]

  sampler_class = Sampler

  if args.split == 'CoGenT':
    sampler_class = CoGenTSampler
  if args.split == 'HoldDiag':
    sampler_class = HoldDiagSampler
  if args.split == 'HoldRedSquare':
    sampler_class = HoldRedSquareSampler
  if args.split == 'BelowRed':
    sampler_class = BelowRedSampler
  if args.split == 'BelowHoldDiag':
    sampler_class = BelowExcludeDiagSampler
  if args.split == 'BelowSquares':
    sampler_class = BelowSquaresSampler

  train_sample = sampler_class(False, 1, colors, shapes)
  val_sample = sampler_class(True, 2, colors, shapes)
  test_sample = sampler_class(True, 3, colors, shapes)

  generate_dataset('train', args.train, 1, train_sample, save_vocab=True)
  generate_dataset('val', args.val, 2, val_sample)
  generate_dataset('test', args.test, 3, test_sample)


if __name__ == '__main__':
  level_splits = {'shapecolor': ['none', 'CoGenT', 'HoldDiag', 'HoldRedSquare'],
                  'relations': ['none', 'BelowRed', 'BelowSquares', 'BelowHoldDiag']}

  parser = argparse.ArgumentParser()
  parser.add_argument('--train', type=int, default=1000,
                      help="Size of the training set")
  parser.add_argument('--val', type=int, default=1000,
                      help="Size of the development set")
  parser.add_argument('--test', type=int, default=1000,
                      help="Size of the test set")
  parser.add_argument('--level', type=str, choices=('shapecolor', 'relations'), default='shapecolor')
  parser.add_argument('--program', type=str, choices=('best', 'noand', 'chain', 'chain_shortcut'))
  parser.add_argument('--num-shapes', type=int, default=len(SHAPES))
  parser.add_argument('--num-colors', type=int, default=len(COLORS))
  parser.add_argument('--num-objects', type=int, default=5)
  parser.add_argument('--image-size', type=int, default=64)
  parser.add_argument('--min-obj-size', type=int, default=10)
  parser.add_argument('--max-obj-size', type=int, default=15)
  parser.add_argument('--rotate', type=int, default=1)
  parser.add_argument('--split', type=str,
                      choices=level_splits['shapecolor'] + level_splits['relations'],
                      default='none',
                      help="The split to use")
  parser.add_argument('--restrict-scene', type=int, default=1,
                      help="Make sure that held-out objects do not appeat in the scene"
                      "during training")
  args = parser.parse_args()

  if not args.split in level_splits[args.level]:
    raise ValueError()
  with open('args.txt', 'w') as dst:
    print(args, file=dst)


  main()
