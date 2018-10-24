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
import numpy
import pygame
import argparse
import h5py
import pygame
import collections
import json


COLOR2RGB = [('red', (255, 0, 0)),
             ('green', (0, 255, 0)),
             ('blue', (0, 0, 255)),
             ('yellow', (255, 255, 0)),
             ('cyan', (0, 255, 255)),
             ('purple', (128, 0, 128)),
             ('brown', (165, 42, 42)),
             ('gray', (128, 128, 128))]
COLOR2RGB = collections.OrderedDict(COLOR2RGB)
COLORS = list(COLOR2RGB.keys())
SHAPES = ['square', 'circle', 'triangle',
          'cross', 'hbar', 'vbar']


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
    elif shape == 'cross':
        pygame.draw.line(surf, rgb, (1, 1), (width - 2, height - 3), 2)
        pygame.draw.line(surf, rgb, (width - 2, 1), (1, height - 3), 2)
    elif shape == 'hbar':
        pygame.draw.line(surf, rgb, (1, height // 2 - 1), (width - 2, height // 2 - 1), 2)
    elif shape == 'vbar':
        pygame.draw.line(surf, rgb, (width // 2 - 1, 1), (width // 2 - 1, height - 2), 2)
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

    def __init__(self, grid_size, cell_size, num_objects, seed,
                 object_allowed):
        self._grid_size = grid_size
        self._cell_size = cell_size
        self._num_objects = num_objects
        self._seed = seed
        self._rng = numpy.random.RandomState(seed)
        self._object_allowed = object_allowed

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
            color = self._rng.choice(COLORS)
            if not self._object_allowed((shape, color), purpose='generate'):
                continue
            objects.append(((i, j), shape, color))
            positions.add((i, j))

        for (i, j), shape, color in objects:
            bitmap = get_object_bitmap(shape, color, self._cell_size)
            surface.blit(source=bitmap, dest=(self._cell_size * i, self._cell_size * j))

        return objects, surface


def generate_dataset(prefix, size, seed, object_allowed, save_vocab=False):
    sg = SceneGenerator(grid_size=5, cell_size=10,
                        num_objects=5, seed=1,
                        object_allowed=object_allowed)

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
    with h5py.File(prefix + '_questions.h5', 'w') as dst_questions,\
         h5py.File(prefix + '_features.h5', 'w') as dst_features:
        features_dataset = dst_features.create_dataset(
            'features', (size, 3, 50, 50), dtype=numpy.float32)
        questions_dataset = dst_questions.create_dataset(
            'questions', (size, max_question_len), dtype=numpy.int64)
        programs_dataset = dst_questions.create_dataset(
            'programs', (size, max_program_len), dtype=numpy.int64)
        answers_dataset = dst_questions.create_dataset(
            'answers', (size,), dtype=numpy.int64)
        image_idxs_dataset = dst_questions.create_dataset(
            'image_idxs', (size,), dtype=numpy.int64)

        rng = numpy.random.RandomState(seed)
        i = 0
        for scene, surface in sg:
            if i == size:
                break

            answer = i % 2
            if answer:
                candidate_objects = [(shape, color) for _, shape, color in scene
                                         if object_allowed((shape, color), 'ask')]
                if not candidate_objects:
                    # can't generate a positive question about this scene
                    continue
                shape, color = candidate_objects[rng.randint(len(candidate_objects))]
            else:
                # sample an allowed (shape, color) pair that is not present in the picture
                # if failed 10 times, try another scene
                for attempt in range(11):
                    shape = rng.choice(SHAPES)
                    color = rng.choice(COLORS)
                    if not object_allowed((shape, color), 'ask'):
                        continue
                    found = any((shape, color) == (obj_shape, obj_color)
                                for _, obj_shape, obj_color in scene)
                    if not found:
                        break
                if attempt == 10:
                    continue

            question = ["is", "there", "a"] + [color, shape]
            program = ['<START>', 'And', shape, 'scene', color, 'scene', '<END>']

            features_dataset[i] = surf2array(surface).transpose(2, 0, 1) / 255.0
            questions_dataset[i] = [question_vocab[w] for w in question]
            programs_dataset[i] = [program_vocab[w] for w in program]
            answers_dataset[i] = int(answer)
            image_idxs_dataset[i] = i

            i += 1

    if save_vocab:
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
    train_object_allowed = val_object_allowed = test_object_allowed = lambda _1, _2: True

    class ObjectRestriction:
        def __init__(self, inverse=False, restrict_scene=True):
            """Base class for object restrictions.

            inverse: if True, invert the output of `self.allow`
            restrict_scene: if True, make sure that excluded objects are not
              generated
            """
            self._inverse = inverse
            self._restrict_scene = restrict_scene
        def __call__(self, obj, purpose):
            if not self._restrict_scene and purpose == 'generate':
                return True
            if self._inverse:
                return self.allow(obj, purpose)
            else:
                return not self.allow(obj, purpose)
        def allow(self, obj, puprose):
            raise NotImplementedError()


    if args.split == 'CoGenT':
        class RestrictSquaresAndTriangles:
            def __init__(self, square_colors, triangle_colors, test=False):
                self._square_colors = square_colors
                self._triangle_colors = triangle_colors
                self._test = test
            def __call__(self, obj, purpose):
                shape, color = obj
                if shape == 'square':
                    return color in self._square_colors
                elif shape == 'triangle':
                    return color in self._triangle_colors
                # at the test time we want to ask questions only about the special
                # objects that were not include in the training set
                if self._test and not purpose == 'generate':
                    return False
                return True

        set1 = ['gray', 'blue', 'brown', 'yellow']
        set2 = ['red', 'green', 'purple', 'cyan']

        train_object_allowed = RestrictSquaresAndTriangles(set1, set2)
        val_object_allowed = test_object_allowed = RestrictSquaresAndTriangles(set2, set1,
                                                                               test=True)
    if args.split == 'diagonal':
        class ExcludeDiagonal:
            def __init__(self, test=False):
                self._test = test
            def __call__(self, obj, purpose):
                shape, color = obj
                shape_number = SHAPES.index(shape)
                color_number = COLORS.index(color)
                diagonal = shape_number == color_number
                if self._test:
                    return purpose == 'generate' or diagonal
                else:
                    return not diagonal
        train_object_allowed = ExcludeDiagonal()
        val_object_allowed = test_object_allowed = ExcludeDiagonal(test=True)

    if args.split == 'leave1out':
        class ExcludeRedSquare(ObjectRestriction):
            def allow(self, obj, purpose):
                return obj == ('square', 'red')
        train_object_allowed = ExcludeRedSquare(restrict_scene=args.restrict_scene)
        val_object_allowed = test_object_allowed = ExcludeRedSquare(
            inverse=True, restrict_scene=False)

    generate_dataset('train', args.train, 1, train_object_allowed, save_vocab=True)
    generate_dataset('val', args.val, 2, val_object_allowed)
    generate_dataset('test', args.test, 3, test_object_allowed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1000,
                        help="Size of the training set")
    parser.add_argument('--val', type=int, default=100,
                        help="Size of the development set")
    parser.add_argument('--test', type=int, default=100,
                        help="Size of the test set")
    parser.add_argument('--split', type=str,
                        choices=('none', 'CoGenT', 'diagonal', 'leave1out'),
                        help="The split to use")
    parser.add_argument('--restrict-scene', type=int, default=1,
                        help="Make sure that held-out objects do not appeat in the scene"
                              "during training")
    args = parser.parse_args()
    main()
