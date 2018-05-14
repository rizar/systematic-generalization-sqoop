import argparse
import itertools
import numpy as np
import collections
from matplotlib import pyplot
import pandas

import torch
from torch import nn
from torch.autograd import Variable

# the number of values
parser = argparse.ArgumentParser()
parser.add_argument('--max-value', type=int, default=10)
parser.add_argument('--split', type=str, default='none', choices=('none', 'random'))
parser.add_argument('--model', type=str, default='Vanilla2')
parser.add_argument('--nsteps', type=int, default=5000)
parser.add_argument('--dim', type=int, default=128)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--k', type=int, default=2, help='number of features')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--nseeds', type=int, default=1)
parser.add_argument('--ipython', action='store_true')
parser.add_argument('--save-path', type=str, default=None)
parser.add_argument('--randomize', type=str, default=None, choices=('random', 'consistent', 'half'))
args = parser.parse_args()


class Linear(nn.Module):

    def __init__(self):
        super().__init__()
        self.output = nn.Linear(4 * args.max_value, 1)

    def __call__(self, h):
        return self.output(h)


class Vanilla1(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4 * args.max_value, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        h = self.act(self.linear1(h))
        return self.output(h)


class Vanilla2(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.linear1 = nn.Linear(2 * k * args.max_value, args.dim)
        self.linear2 = nn.Linear(args.dim, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        h = self.act(self.linear1(h))
        h = self.act(self.linear2(h))
        return self.output(h)


class Shortcut(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4 * args.max_value, args.dim)
        self.linear2 = nn.Linear(2 * args.max_value + args.dim, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        y_part = h[:, 2 * args.max_value:]
        h = self.act(self.linear1(h))
        h = self.act(self.linear2(torch.cat([h, y_part], 1)))
        return self.output(h)


class Split(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3 * args.max_value, args.dim)
        self.linear2 = nn.Linear(3 * args.max_value + args.dim, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        x_and_y1_part = h[:, :3 * args.max_value]
        x_and_y2_part = torch.cat([h[:, :2 * args.max_value], h[:, 3 * args.max_value:]], 1)
        h = self.act(self.linear1(x_and_y1_part))
        h = self.act(self.linear2(torch.cat([h, x_and_y2_part], 1)))
        return self.output(h)


class Cheater(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.linear1 = [nn.Linear(2 * args.max_value, args.dim // k)
                        for i in range(k)]
        self.linear2 = nn.Linear((args.dim // k) * k, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        h1 = [self.act(self.linear1[i](h[i])) for i in range(self.k)]
        h2 = self.act(self.linear2(torch.cat(h1, 1)))
        return self.output(h2)


class WeakCheater(nn.Module):

    def __init__(self, k):
        super().__init__()
        self.k = k
        self.linear1 = [nn.Linear((k+1) * args.max_value, args.dim // k)
                        for i in range(k)]
        self.linear2 = nn.Linear((args.dim // k) * k, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        h1 = [self.act(self.linear1[i](h[i])) for i in range(self.k)]
        h2 = self.act(self.linear2(torch.cat(h1, 1)))
        return self.output(h2)


class Dataset:
    def __init__(self, rng, batch_size, k, split, max_value):
        self.rng = rng
        self.batch_size = batch_size
        self.max_value = max_value
        self.k = k

        self._generate_data(k, max_value, split)
        self.random_inputs = np.random.randint(self.max_value, size=(self.max_value**k, k))

    def _generate_data(self, k, max_value, split):
        self.num_examples = max_value**(k*2)
        # inputs = x_1...x_k, y_1...y_k
        inputs_iter = itertools.product(range(max_value), repeat=k*2)
        inputs_np = np.fromiter(itertools.chain.from_iterable(inputs_iter), np.int)
        self.inputs = np.reshape(inputs_np, (max_value**(k*2), k*2))
        # targets = 1 if x_i = y_i, else -1
        xy_equal = []
        for i in range(k):
            xy_equal.append(self.inputs[:, i] == self.inputs[:,i+k])
        self.targets = 2 * np.all(np.stack(xy_equal, axis=1), axis=1) - 1

        is_train = self.rng.binomial(1, 0.5, self.num_examples)
        if split == 'random':
            positive_train = np.where((self.targets == 1) & is_train)[0]
            positive_test = np.where((self.targets == 1) & (~is_train))[0]
            negative_train = np.where((self.targets == -1) & is_train)[0]
            negative_test = np.where((self.targets == -1) & (~is_train))[0]
            self.positive_indices = (positive_train, positive_test)
            self.negative_indices = (negative_train, negative_test)
        elif split == 'none':
            positives = np.where(self.targets == 1)[0]
            negatives = np.where(self.targets == -1)[0]
            self.positive_indices = (positives, positives)
            self.negative_indices = (negatives, negatives)

    def _continue_data(max_value):
        inputs = []
        targets = []
        positive_indices = [[], []]
        negative_indices = [[], []]
        r = range(max_value)
        for x1 in r:
            for x2 in r:
                for y1 in r:
                    for y2 in r:
                        if split == 'none':
                            part = 0
                        elif split == 'random':
                            part = self.rng.randint(2)
                        if x1 == y1 and x2 == y2:
                            positive_indices[part].append(len(inputs))
                            targets.append(1)
                        else:
                            negative_indices[part].append(len(inputs))
                            targets.append(-1)
                        inputs.append([x1, x2, y1, y2])
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        self.positive_indices = list(map(np.array, positive_indices))
        self.negative_indices = list(map(np.array, negative_indices))
        if split == 'none':
            self.positive_indices[1] = positive_indices[0]
            self.negative_indices[1] = negative_indices[0]

    def _onehot(self, tensor):
        tensor.unsqueeze_(-1)
        one_hot = torch.LongTensor(self.batch_size, self.max_value).zero_()
        one_hot.scatter_(1, tensor, 1)
        return one_hot.float()

    def get_batch(self, part, model=None, randomize=None):
        indices = np.concatenate([self.rng.choice(self.positive_indices[part], self.batch_size // 2),
                                  self.rng.choice(self.negative_indices[part], self.batch_size // 2)])
        targets = Variable(torch.FloatTensor(self.targets[indices]))
        xy = [torch.LongTensor(self.inputs[indices, j]) for j in range(2*self.k)]
        x = [self._onehot(v) for v in xy[:self.k]]
        y = [self._onehot(v) for v in xy[self.k:]]

        if model == "WeakCheater":
            # if randomize == 'random':
                # r = torch.LongTensor(self.rng.randint(self.max_value, size=(self.batch_size, 2)))
                # r1 = self._onehot(r[:,0])
                # r2 = self._onehot(r[:,1])
                # h = (Variable(torch.cat([ox1, r2, oy1], 1)),
                     # Variable(torch.cat([r1, ox2, oy2], 1)))
            # elif randomize == 'consistent':
                # r = torch.LongTensor(self.random_inputs[indices])
                # r1 = self._onehot(r[:,0])
                # r2 = self._onehot(r[:,1])
                # h = (Variable(torch.cat([ox1, r2, oy1], 1)),
                     # Variable(torch.cat([r1, ox2, oy2], 1)))
            # elif randomize == 'half':
                # r = torch.LongTensor(0.9*self.random_inputs[indices] +
                                     # 0.1*self.rng.randint(self.max_value, size=(self.batch_size, 2)))
                # r1 = self._onehot(r[:,0])
                # r2 = self._onehot(r[:,1])
                # h = (Variable(torch.cat([ox1, r2, oy1], 1)),
                     # Variable(torch.cat([r1, ox2, oy2], 1)))
            # else:
            h = [Variable(torch.cat(x + [y[i]], 1)) for i in range(self.k)]
        elif model == 'Cheater':
            h = [Variable(torch.cat([x[i]] + [y[i]], 1)) for i in range(self.k)]
        else:
            h = Variable(torch.cat(x + y, 1))

        return h, targets.unsqueeze(-1)


if __name__ == '__main__':
    stats = collections.defaultdict(list)
    for seed in range(args.nseeds):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)

        net = eval(args.model)(args.k)
        data = Dataset(rng, args.batch_size, args.k, args.split, args.max_value)

        for i in range(args.nsteps):
            # train
            h, targets = data.get_batch(0, args.model, randomize=args.randomize)
            net.zero_grad()
            s = net(h)
            train_cost = torch.nn.Softplus()(s * -targets).mean()
            train_acc = (targets * s > 0).float().mean()
            train_cost.backward()
            for p in net.parameters():
                p.data -= args.lr * p.grad.data

            # test
            h, targets = data.get_batch(1, args.model, randomize=args.randomize)
            s = net(h)
            test_cost = torch.nn.Softplus()(s * -targets).mean()
            test_acc = (targets * s > 0).float().mean()

            if i % 2 == 0:
                print("train cost: {}, test cost: {}, train acc: {}, test acc: {}".format(
                    train_cost.data[0], test_cost.data[0], train_acc.data[0], test_acc.data[0]))
            stats['step'].append(i)
            stats['seed'].append(seed)
            stats['train_cost'].append(train_cost.data[0])
            stats['train_acc'].append(train_acc.data[0])
            stats['test_cost'].append(test_cost.data[0])
            stats['test_acc'].append(test_acc.data[0])

    df = pandas.DataFrame.from_dict(stats)
    if args.save_path:
        df.to_csv(args.save_path)
    df_agg = df.groupby('step').agg('mean')

    f, axis = pyplot.subplots(1, 2)
    axis[0].plot(df_agg.index, df_agg['train_cost'])
    axis[0].plot(df_agg.index, df_agg['test_cost'])
    axis[1].plot(df_agg.index, df_agg['train_acc'])
    axis[1].plot(df_agg.index, df_agg['test_acc'])
    pyplot.show()
    if args.ipython:
        import IPython; IPython.embed()
