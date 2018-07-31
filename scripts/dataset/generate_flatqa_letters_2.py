import argparse
import collections
import io
import json
import logging
import math
import string
import time
import random
import sys
from functools import partial

import h5py
import numpy
from PIL import Image, ImageDraw, ImageFont


logger = logging.getLogger(__name__)
RELATIONS = ['left_of', 'right_of', 'above', 'below']
COLORS = ['red', 'green', 'blue', 'yellow', 'cyan',
          'purple', 'brown', 'gray']
SHAPES = list(string.ascii_uppercase) + ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']


# === Definition of modules for NMN === #
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



class Object(object):
  def __init__(self, fontsize, angle=0, pos=None, shape=None):
    self.font = ImageFont.truetype('FreeSans.ttf', fontsize)
    width, self.size = self.font.getsize('A')
    self.angle = angle
    angle_rad = angle / 180 * math.pi
    self.rotated_size =  math.ceil(self.size * (abs(math.sin(angle_rad)) + abs(math.cos(angle_rad))))
    self.pos = pos
    self.shape = shape

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

  def draw(self):
    img = Image.new('RGBA', (self.size, self.size))
    draw = ImageDraw.Draw(img)
    draw.text((0,0), self.shape, font=self.font, fill='green')

    #if self.angle != 0:
    #  img = img.rotate(self.angle, expand=True, resample=Image.LINEAR)

    return img

class CustomJSONEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, Object):
      return {'size': obj.size,
              'rotated_size': obj.rotated_size,
              'angle': obj.angle,
              'pos': obj.pos,
              'shape': obj.shape
              }
    else:
      return super().default(obj)


def draw_scene(objects):
  img = Image.new('RGB', (args.image_size, args.image_size))
  for obj in objects:
    obj_img = obj.draw()
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


def generate_scene(rng, sampler, objects=[], **kwargs):
  orig_objects = objects

  objects = list(orig_objects)
  place_failures = 0
  while len(objects) < args.num_objects:
    # first, select which object to draw by rejection sampling
    shape = sampler.sample_object(purpose='generate', **kwargs)

    new_object = get_random_spot(rng, objects)
    if new_object is None:
      place_failures += 1
      if place_failures == 10:
        # reset generation
        objects = list(orig_objects)
        place_failures = 0
      continue

    new_object.shape = shape
    objects.append(new_object)

  return objects




class Sampler:
  def __init__(self, test, seed, objects):
    self._test = test
    self._rng = numpy.random.RandomState(seed)
    self.objects = objects

  def _choose(self, list_like):
    return list_like[self._rng.randint(len(list_like))]

  def _rejection_sample(self, restricted=[]):
    while True:
      rand_object = self._rng.choice(self.objects)
      if rand_object not in restricted:
        return rand_object 

  def sample_relation(self, *args, **kwargs):
    return self._choose(RELATIONS)

  def sample_object(self, *args, **kwargs):
    return self._rejection_sample()



class _LongTailSampler(Sampler):
  def __init__(self, dist, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.object_probs = dist

  def sample_object(self, *args, **kwargs):
    if self._test:
        return self._rejection_sample()
    else:
        return self._rejection_sample(self.object_probs)

  def _rejection_sample(self, shape_probs=None):
    shape = self._rng.choice(self.objects, p=shape_probs)
    return shape


def LongTailSampler(long_tail_dist):
  return partial(_LongTailSampler, long_tail_dist)


def flatQA_gen(vocab):
  uniform_dist = [1.0 / len(vocab) ]*len(vocab)
  sampler_class = LongTailSampler(uniform_dist)

  train_sampler = sampler_class(False, 1, vocab)
  dev_sampler   = sampler_class(True,  2, vocab)
  test_sampler  = sampler_class(True,  3, vocab)

  train_pairs = []
  dev_pairs   = []
  test_pairs  = []  

  chosen = set([ (x,y) for x in vocab for y in vocab if x != y] )
  for x in vocab:
    ys = random.sample(vocab, args.rhs_variety)
    for y in ys:
      if x == y: continue
      chosen.remove((x,y))
      train_pairs += [(x,y)]*args.num_repeats

  left = list(chosen)
  print('number of zero shot pairs: %d' %len(left))
  # dev / test pairs are all unseen
  dev_slice = len(left) // 2
 

  for pair in left[ : dev_slice] :
    dev_pairs  += [pair]*args.num_repeats_eval

  for pair in left[ dev_slice : ] :
    test_pairs += [pair]*args.num_repeats_eval 

  


  # generate data vocabulary
  question_words = (['<NULL>', '<START>', '<END>', 'is', 'there', 'a', 'green'] + vocab + RELATIONS)
  question_vocab = {word: i for i, word in enumerate(question_words)}

  program_words = (['<NULL>', '<START>', '<END>', 'scene', 'And']
                   + [color_module('green')]
                   + [shape_module(shape) for shape in vocab]
                   + [binary_color_module('green') ]
                   + [binary_shape_module(shape) for shape in vocab]
                   + [relation_module(rel) for rel in RELATIONS]
                   + [unary_relation_module(rel) for rel in RELATIONS])
  program_vocab = {word: i for i, word in enumerate(program_words)}

  answer_token_to_idx = {word: idx for idx, word in
                         enumerate(['false', 'true'])}
  module_token_to_idx = {word: idx for idx, word in
                         enumerate(['find', 'relate', 'and'])}
  program_token_to_module_text = {}
  program_token_to_module_text[color_module('green')] = ['find', 'green']
  for shape in vocab:
    program_token_to_module_text[shape_module(shape)] = ['find', shape]
  for rel in RELATIONS:
    program_token_to_module_text[relation_module(rel)] = ['relate', rel]
  program_token_to_module_text['And'] = ('and', 'null')
  for module in ['<START>', '<END>', '<NULL>']:
    program_token_to_module_text[module] = ('null', 'null')

  text_token_to_idx = {}
  for idx, word in enumerate(
      ['null', 'green'] + vocab + RELATIONS):
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


  gen_data(train_pairs, train_sampler, 1, vocab, 'train', question_vocab, program_vocab)
  gen_data(dev_pairs, dev_sampler, 2, vocab, 'dev', question_vocab, program_vocab)
  gen_data(test_pairs, test_sampler, 3, vocab, 'test', question_vocab, program_vocab)



 
def gen_data(obj_pairs, sampler, seed, vocab, prefix, question_vocab, program_vocab):
  num_examples = len(obj_pairs)
  max_question_len = 8
  if args.program == 'best':
    max_program_len = 13
  elif args.program == 'noand':
    max_program_len = 9
  elif args.program == 'chain':
    max_program_len = 8
  elif args.program == 'chain_shortcut':
    max_program_len = 12

  presampled_relations = [sampler.sample_relation() for ex in obj_pairs] # pre-sample relations
  with h5py.File(prefix + '_questions.h5', 'w') as dst_questions, h5py.File(prefix + '_features.h5', 'w') as dst_features:
    features_dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
    features_dataset = dst_features.create_dataset('features', (num_examples,), dtype=features_dtype)
    questions_dataset = dst_questions.create_dataset('questions', (num_examples, max_question_len), dtype=numpy.int64)
    programs_dataset = dst_questions.create_dataset('programs', (num_examples, max_program_len), dtype=numpy.int64)
    answers_dataset = dst_questions.create_dataset('answers', (num_examples,), dtype=numpy.int64)
    image_idxs_dataset = dst_questions.create_dataset('image_idxs', (num_examples,), dtype=numpy.int64)

    i = 0
  
    # different seeds for train/dev/test
    rng = numpy.random.RandomState(seed)
    before = time.time()
    scenes = []
    while i < len(obj_pairs):
      scene, question, program, success = generate_imgAndQuestion(obj_pairs[i], sampler, rng, (i % 2) == 0, vocab, presampled_relations[i])
      if success:
        scenes.append(scene)
        buffer_ = io.BytesIO()
        image = draw_scene(scene)
        image.save(buffer_, format='png')
        buffer_.seek(0)
        features_dataset[i]   = numpy.frombuffer(buffer_.read(), dtype='uint8') 
        questions_dataset[i]  = [question_vocab[w] for w in question] 
        programs_dataset[i]   = [program_vocab[w] for w in program] 
        answers_dataset[i]    = int( (i%2) == 0) 
        image_idxs_dataset[i] = i
        print("\r>> Done with %d/%d examples" %(i+1, len(obj_pairs)), end = '')
        sys.stdout.flush() 
        i += 1

  print("{} seconds per example".format((time.time() - before) / len(obj_pairs) ))

  with open(prefix + '_scenes.json', 'w') as dst:
    json.dump(scenes, dst, indent=2, cls=CustomJSONEncoder)


def generate_imgAndQuestion(pair, sampler, rng, label, vocab, rel):
  # x rel y has value label where pair == (x, y) 

  max_question_len = 8
  if args.program == 'best':
    max_program_len = 13
  elif args.program == 'noand':
    max_program_len = 9
  elif args.program == 'chain':
    max_program_len = 8
  elif args.program == 'chain_shortcut':
    max_program_len = 12



  x,y = pair
  if label:
    obj1 = get_random_spot(rng, [])
    obj2 = get_random_spot(rng, [obj1])
    if not obj2 or not obj1.relate(rel, obj2): return None, None, None, False
    obj1.shape = x
    obj2.shape = y
    scene = generate_scene(rng, sampler, objects=[obj1, obj2], relation=rel)
  else:
    # first generate a scene
    scene = generate_scene(rng, sampler, relation=rel)
    # choose x,y,x', y' st. x r' y, x r y', x' r y holds true
    i, j, k, l = random.sample(range(len(scene)), 4)

    obj1 = scene[i] #x
    obj1.shape = x
    obj2 = scene[j] #y
    obj2.shape = y
    obj3 = scene[k] #x'
    obj4 = scene[l] #y'

    r_corrupted_1 = sampler.sample_relation()
    if r_corrupted_1 == rel or obj3.shape == x or obj4.shape == y: return None, None, None, False 
    elif not obj1.relate(r_corrupted_1, obj2): return None, None, None, False
    elif not obj1.relate(rel, obj4): return None, None, None, False
    elif not obj3.relate(rel, obj2): return None, None, None, False

  color1 = "green"
  color2 = "green"
  shape1 = x
  shape2 = y
  question = ["is", "there", "a", color1 , x, rel, color2, y] 
  if args.program == 'best':
    program = ["<START>", relation_module(rel),
         "And", shape_module(shape1), "scene", color_module(color1), "scene",
         "And", shape_module(shape2), "scene", color_module(color2), "scene",
         "<END>"]
  elif args.program == 'noand':
    program = ["<START>", relation_module(rel),
         shape_module(shape1), color_module(color1), "scene",
         shape_module(shape2), color_module(color2), "scene",
         "<END>"]
  elif args.program == 'chain':
    program = ["<START>",
         shape_module(shape1), color_module(color1),
         unary_relation_module(rel),
         shape_module(shape2), color_module(color2),
         "scene", "<END>"]
  elif args.program == 'chain_shortcut':
    program = ["<START>",
         binary_shape_module(shape1), 'scene',
         binary_color_module(color1), 'scene',
         unary_relation_module(rel),
         binary_shape_module(shape2), 'scene',
         binary_color_module(color2), 'scene',
         'scene', "<END>"]

  return scene, question, program, True


def main():
  flatQA_gen(SHAPES)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', type=int, default=1000,
                      help="Size of the training set")
  parser.add_argument('--val', type=int, default=1000,
                      help="Size of the development set")
  parser.add_argument('--test', type=int, default=1000,
                      help="Size of the test set")
  parser.add_argument('--program', type=str, choices=('best', 'noand', 'chain', 'chain_shortcut'), default='best')
  parser.add_argument('--num-shapes', type=int, default=len(SHAPES))
  parser.add_argument('--num-colors', type=int, default=len(COLORS))
  parser.add_argument('--num-objects', type=int, default=5)
  parser.add_argument('--rhs_variety', type=int, default=15)
  parser.add_argument('--num_repeats', type=int, default=1000)
  parser.add_argument('--num_repeats_eval', type=int, default=10)
  

  parser.add_argument('--image-size', type=int, default=64)
  parser.add_argument('--min-obj-size', type=int, default=10)
  parser.add_argument('--max-obj-size', type=int, default=15)
  parser.add_argument('--no-rotate', action='store_false', dest='rotate')
  args = parser.parse_args()

  args.level = 'relations'

  with open('args.txt', 'w') as dst:
    print(args, file=dst)

  main()
