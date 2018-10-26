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
import os
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
        self.font = FONT_OBJECTS[fontsize]
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

def get_random_spot(rng, objects, rel = None,  rel_holds = False, rel_obj = 0):
    """Get a spot for a new object that does not overlap with existing ones."""
    # then, select the object size
    size = rng.randint(args.min_obj_size, args.max_obj_size + 1)
    angle = rng.randint(0, 360) if args.rotate else 0
    obj = Object(size, angle)

    min_center = obj.rotated_size // 2 + 1
    max_center = args.image_size - obj.rotated_size // 2 - 1

    if rel is not None:
        if rel_holds == False:
            # do not want the relation to be true
            max_center_x = objects[rel_obj].pos[0] if rel == 'left_of' else max_center
            min_center_x = objects[rel_obj].pos[0] if rel == 'right_of' else min_center
            max_center_y = objects[rel_obj].pos[1] if rel == 'below' else max_center
            min_center_y = objects[rel_obj].pos[1] if rel == 'above' else min_center
        else:
            # want the relation to be true
            min_center_x = objects[rel_obj].pos[0] if rel == 'left_of' else min_center
            max_center_x = objects[rel_obj].pos[0] if rel == 'right_of' else max_center
            min_center_y = objects[rel_obj].pos[1] if rel == 'below' else min_center
            max_center_y = objects[rel_obj].pos[1] if rel == 'above' else max_center

        if min_center_x >= max_center_x: return None
        if min_center_y >= max_center_y: return None

    else:
        min_center_x = min_center_y = min_center
        max_center_x = max_center_y = max_center


    for attempt in range(10):
        x = rng.randint(min_center_x, max_center_x)
        y = rng.randint(min_center_y, max_center_y)
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


def generate_scene(rng, sampler, objects=[], restrict = False, **kwargs):
    orig_objects = objects

    objects = list(orig_objects)
    place_failures = 0

    if restrict:
        restricted_obj = [obj.shape for obj in orig_objects]
    else:
        restricted_obj = []

    while len(objects) < args.num_objects:
        # first, select which object to draw by rejection sampling
        shape = sampler.sample_object(restricted_obj, [], **kwargs)

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
        print(args)
        if len(args) > 0:
            return self._rejection_sample(args[0])
        else:
            return self._rejection_sample()


class _LongTailSampler(Sampler):
    def __init__(self, dist, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.object_probs = dist

    def sample_object(self, restricted = [], *args, **kwargs):
        if self._test:
            return self._rejection_sample(restricted = restricted)
        else:
            return self._rejection_sample(self.object_probs, restricted = restricted)

    def _rejection_sample(self, shape_probs=None, restricted = []):
        while True:
            rand_object = self._rng.choice(self.objects, p = shape_probs)
            if rand_object not in restricted:
                return rand_object



def LongTailSampler(long_tail_dist):
    return partial(_LongTailSampler, long_tail_dist)


def gen_data(obj_pairs, sampler, seed, vocab, prefix, question_vocab, program_vocab):
    num_examples = len(obj_pairs)

    max_question_len = 3
    if args.program == 'best':
        max_program_len = 7
    elif args.program in ['chain', 'chain2', 'chain3']:
        max_program_len = 6
    elif args.program == 'chain_shortcut':
        max_program_len = 8

    presampled_relations = [sampler.sample_relation() for ex in obj_pairs] # pre-sample relations
    with h5py.File(prefix + '_questions.h5', 'w') as dst_questions, h5py.File(prefix + '_features.h5', 'w') as dst_features:
        features_dtype = h5py.special_dtype(vlen=numpy.dtype('uint8'))
        features_dataset = dst_features.create_dataset('features', (num_examples,), dtype=features_dtype)
        questions_dataset = dst_questions.create_dataset('questions', (num_examples, max_question_len), dtype=numpy.int64)
        programs_dataset = dst_questions.create_dataset('programs', (num_examples, max_program_len), dtype=numpy.int64)
        answers_dataset = dst_questions.create_dataset('answers', (num_examples,), dtype=numpy.int64)
        image_idxs_dataset = dst_questions.create_dataset('image_idxs', (num_examples,), dtype=numpy.int64)

        i = 0
        rejection_sampling = {'a' : 0, 'b' : 0, 'c' : 0, 'd' : 0, 'e' : 0, 'f' : 0}

        # different seeds for train/dev/test
        rng = numpy.random.RandomState(seed)
        before = time.time()
        scenes = []
        while i < len(obj_pairs):
            scene, question, program, success, key = generate_image_and_question(
                obj_pairs[i], sampler, rng, (i % 2) == 0, vocab, presampled_relations[i])
            rejection_sampling[key] += 1
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

                i += 1
                if i % 1000 == 0:
                    time_data = "{} seconds per example".format((time.time() - before) / i )
                    print(time_data)
                print("\r>> Done with %d/%d examples : %s " %(i+1, len(obj_pairs),  rejection_sampling), end = '')
                sys.stdout.flush()

    print("{} seconds per example".format((time.time() - before) / len(obj_pairs) ))

    with open(prefix + '_scenes.json', 'w') as dst:
        json.dump(scenes, dst, indent=2, cls=CustomJSONEncoder)


def generate_image_and_question(pair, sampler, rng, label, vocab, rel):
    # x rel y has value label where pair == (x, y)

    x,y = pair
    if label:
        obj1 = get_random_spot(rng, [])
        obj2 = get_random_spot(rng, [obj1])
        if not obj2 or not obj1.relate(rel, obj2): return None, None, None, False, 'a'
        obj1.shape = x
        obj2.shape = y
        scene = generate_scene(rng, sampler, objects=[obj1, obj2], restrict = False, relation=rel)
    else:
        # first generate a scene
        obj1 = get_random_spot(rng, [])
        obj2 = get_random_spot(rng, [obj1], rel = rel, rel_holds = False)
        if not obj2 or obj1.relate(rel, obj2): return None, None, None, False, 'b'
        obj1.shape = x
        obj2.shape = y


        scene = generate_scene(rng, sampler, objects = [obj1, obj2], restrict = True, relation=rel)
        # choose x,y,x', y' st. x r' y, x r y', x' r y holds true

        obj3 = scene[2] #x'
        obj4 = scene[3] #y'

        if not obj1.relate(rel, obj4): return None, None, None, False, 'c'
        elif not obj3.relate(rel, obj2): return None, None, None, False, 'd'

    color1 = "green"
    color2 = "green"
    shape1 = x
    shape2 = y
    question = [x, rel, y]
    if args.program == 'best':
        program = ["<START>", relation_module(rel),
                   shape_module(shape1), "scene",
                   shape_module(shape2), "scene",
                   "<END>"]
    elif args.program == 'chain':
        program = ["<START>",
                   shape_module(shape1),
                   unary_relation_module(rel),
                   shape_module(shape2),
                   "scene", 
                   "<END>"]
    elif args.program == 'chain2':
        program = ["<START>",
                   shape_module(shape1), 
                   shape_module(shape2),
                   unary_relation_module(rel),
                   "scene", 
                   "<END>"]
    elif args.program == 'chain3':
        program = ["<START>",
                   unary_relation_module(rel),
                   shape_module(shape1),
                   shape_module(shape2),
                   "scene", 
                   "<END>"]
    elif args.program == 'chain_shortcut':
        program = ["<START>",
                   binary_shape_module(shape1), 'scene',
                   unary_relation_module(rel),
                   binary_shape_module(shape2), 'scene',
                   'scene', 
                   "<END>"]

    return scene, question, program, True, 'f'


def gen_sqoop(vocab):
    uniform_dist = [1.0 / len(vocab) ]*len(vocab)
    sampler_class = LongTailSampler(uniform_dist)

    train_sampler = sampler_class(False, 1, vocab)
    val_sampler   = sampler_class(True,  2, vocab)
    test_sampler  = sampler_class(True,  3, vocab)

    train_pairs = []
    val_pairs   = []
    test_pairs  = []

    all_pairs = set([(x,y) for x in vocab for y in vocab if x != y])
    chosen = set(all_pairs)
    for i, x in enumerate(vocab):
        ys = random.sample(vocab[:i] + vocab[i+1:], args.rhs_variety)
        for y in ys:
            chosen.remove((x,y))
            train_pairs += [(x,y)]*args.num_repeats

    random.shuffle(train_pairs)

    if args.split == 'systematic':
        left = list(chosen)
        print('number of zero shot pairs: %d' % len(left))
        # dev / test pairs are all unseen
        val_slice = len(left) // 2
        for pair in left[:val_slice]:
            val_pairs  += [pair] * args.num_repeats_eval
        for pair in left[val_slice:]:
            test_pairs += [pair] * args.num_repeats_eval
    else:
        all_ = list(all_pairs)
        for pair in all_:
            val_pairs += [pair] * args.num_repeats_eval
        for pair in all_:
            test_pairs += [pair] * args.num_repeats_eval

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
    gen_data(val_pairs, val_sampler, 2, vocab, 'val', question_vocab, program_vocab)
    gen_data(test_pairs, test_sampler, 3, vocab, 'test', question_vocab, program_vocab)


def gen_image_understanding_test():
    uniform_dist = [1.0 / len(vocab) ]*len(vocab)
    sampler_class = LongTailSampler(uniform_dist)

    eval_sampler = sampler_class(True, 4, vocab)

    question_file = h5py.File('train_questions.h5', 'r')
    questions = question_file['questions']
    vocab_file = open('vocab.json'); vocab_obj = json.load(vocab_file);

    question_vocab = vocab_obj['question_token_to_idx']
    program_vocab  = vocab_obj['program_token_to_idx']

    inverse_vocab = {idx : sym for (sym, idx) in question_vocab.items() }

    seen_pairs = []
    for question in questions:
        x,y = (inverse_vocab[question[4] ], inverse_vocab[question[7] ])
        seen_pairs.append( (x,y) )

    seen_pairs_uniq = list(set(seen_pairs))
    seen_pairs = []
    for seen_pair in seen_pairs_uniq:
        seen_pairs += [seen_pair]*args.num_repeats_eval


    gen_data(seen_pairs, eval_sampler, 4, vocab, 'test_easy', question_vocab, program_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--program', type=str,
      choices=('best', 'noand', 'chain', 'chain2', 'chain3', 'chain_shortcut'),
      default='best')
    parser.add_argument('--num-shapes', type=int, default=len(SHAPES))
    parser.add_argument('--num-colors', type=int, default=1)
    parser.add_argument('--num-objects', type=int, default=5)
    parser.add_argument('--rhs_variety', type=int, default=len(SHAPES) // 2)
    parser.add_argument('--split', type=str, default='systematic', choices=('systematic', 'vanilla'))
    parser.add_argument('--num_repeats', type=int, default=10)
    parser.add_argument('--num_repeats_eval', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument(
        '--mode', type=str, choices=['sqoop', 'sqoop_easy_test'],
      default='sqoop',
      help='in sqoop_easy_test mode the script generates a test set with the same '
           'questions as the dataset in the current directory, '
           'but with different images')
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--min-obj-size', type=int, default=10)
    parser.add_argument('--max-obj-size', type=int, default=15)
    parser.add_argument('--no-rotate', action='store_false', dest='rotate')
    parser.add_argument('--font', default='arial.ttf')
    args = parser.parse_args()

    args.level = 'relations'
    data_full_dir = "%s/sqoop-variety_%d-repeats_%d" %(args.data_dir, args.rhs_variety, args.num_repeats)
    if args.split == 'vanilla':
        data_full_dir += "_vanilla"
    if not os.path.exists(data_full_dir):
        os.makedirs(data_full_dir)

    os.chdir(data_full_dir)
    with open('args.txt', 'w') as dst:
        print(args, file=dst)

    FONT_OBJECTS = { font_size : ImageFont.truetype(args.font) for font_size in range(10, 16) }

    vocab = SHAPES[:args.num_shapes]
    if args.mode == 'sqoop':
        gen_sqoop(vocab)
    else:
        gen_image_understanding_test(vocab)
