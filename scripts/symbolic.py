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
parser.add_argument('--num-feats', type=int, default=2, help='number of features')
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

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2 * args.num_feats * args.max_value, args.dim)
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

    def __init__(self):
        super().__init__()
        self.linear1 = [nn.Linear(2 * args.max_value, args.dim // args.num_feats)
                        for i in range(args.num_feats)]
        self.linear2 = nn.Linear((args.dim // args.num_feats) * args.num_feats, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

        for i, linear in enumerate(self.linear1):
            self.add_module('linear1-{}'.format(i), linear)

    def __call__(self, h):
        h1 = [self.act(self.linear1[i](h[i])) for i in range(args.num_feats)]
        h2 = self.act(self.linear2(torch.cat(h1, 1)))
        return self.output(h2)


class WeakCheater(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = [nn.Linear((args.num_feats + 1) * args.max_value, args.dim // args.num_feats)
                        for i in range(args.num_feats)]
        self.linear2 = nn.Linear((args.dim // args.num_feats) * args.num_feats, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

        for i, linear in enumerate(self.linear1):
            self.add_module('linear1-{}'.format(i), linear)

    def __call__(self, h):
        h1 = [self.act(self.linear1[i](h[i])) for i in range(args.num_feats)]
        h2 = self.act(self.linear2(torch.cat(h1, 1)))
        return self.output(h2)


class Dataset:
    def __init__(self, rng):
        self.rng = rng
        self._generate_data()

    def _generate_data(self):
        self.num_examples = args.max_value**(args.num_feats*2)
        # inputs = x_1...x_k, y_1...y_k
        inputs_iter = itertools.product(range(args.max_value), repeat=args.num_feats*2)
        inputs_np = np.fromiter(itertools.chain.from_iterable(inputs_iter), np.int)
        self.inputs = np.reshape(inputs_np, (args.max_value**(args.num_feats*2), args.num_feats*2))
        # targets = 1 if x_i = y_i, else -1
        xy_equal = []
        for i in range(args.num_feats):
            xy_equal.append(self.inputs[:, i] == self.inputs[:,i+args.num_feats])
        self.targets = 2 * np.all(np.stack(xy_equal, axis=1), axis=1) - 1

        is_train = self.rng.binomial(1, 0.5, self.num_examples)
        if args.split == 'random':
            positive_train = np.where((self.targets == 1) & is_train)[0]
            positive_test = np.where((self.targets == 1) & (~is_train))[0]
            negative_train = np.where((self.targets == -1) & is_train)[0]
            negative_test = np.where((self.targets == -1) & (~is_train))[0]
            self.positive_indices = (positive_train, positive_test)
            self.negative_indices = (negative_train, negative_test)
        elif args.split == 'none':
            positives = np.where(self.targets == 1)[0]
            negatives = np.where(self.targets == -1)[0]
            self.positive_indices = (positives, positives)
            self.negative_indices = (negatives, negatives)

        if args.randomize is not None:
            self.random_inputs = np.random.randint(
                args.max_value, size=(args.max_value**args.num_feats, args.num_feats))

    def _onehot(self, array):
        one_hot = np.zeros((args.batch_size, args.max_value))
        one_hot[np.arange(args.batch_size), array] = 1
        return one_hot

    def get_batch(self, part):
        indices = np.concatenate([self.rng.choice(self.positive_indices[part], args.batch_size // 2),
                                  self.rng.choice(self.negative_indices[part], args.batch_size // 2)])
        targets = Variable(torch.FloatTensor(self.targets[indices]))
        xs = self.inputs[indices, :args.num_feats]
        ys = self.inputs[indices, args.num_feats:]
        x = [torch.FloatTensor(self._onehot(xs[:,i])) for i in range(args.num_feats)]
        y = [torch.FloatTensor(self._onehot(ys[:,i])) for i in range(args.num_feats)]

        if args.model == "WeakCheater":
            if args.randomize is None:
                h = [Variable(torch.cat(x + [y[i]], 1)) for i in range(args.num_feats)]
            elif args.randomize == 'random':
                rs = self.rng.randint(args.max_value, size=(args.batch_size, args.num_feats))
                r = [torch.FloatTensor(self._onehot(rs[:,i])) for i in range(args.num_feats)]
                h = [Variable(torch.cat(r[:i] + [x[i]] + r[i+1:] + [y[i]], 1))
                     for i in range(num_feats)]
            elif args.randomize == 'consistent':
                r = [torch.FloatTensor(self._onehot(self.random_inputs[indices][:,i]))
                     for i in range(args.num_feats)]
                h = [Variable(torch.cat(r[:i] + [x[i]] + r[i+1:] + [y[i]], 1))
                     for i in range(num_feats)]
        elif args.model == 'Cheater':
            h = [Variable(torch.cat([x[i], y[i]], 1)) for i in range(args.num_feats)]
        else:
            h = Variable(torch.cat(x + y, 1))

        return h, targets.unsqueeze(-1)


if __name__ == '__main__':
    stats = collections.defaultdict(list)
    for seed in range(args.nseeds):
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        net = eval(args.model)()
        data = Dataset(rng)

        for i in range(args.nsteps):
            # train
            h, targets = data.get_batch(0)
            net.zero_grad()
            s = net(h)
            train_cost = torch.nn.Softplus()(s * -targets).mean()
            train_acc = (targets * s > 0).float().mean()
            train_cost.backward()
            for p in net.parameters():
                p.data -= args.lr * p.grad.data

            # test
            h, targets = data.get_batch(1)
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
