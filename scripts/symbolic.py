import argparse
import numpy
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
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--nseeds', type=int, default=1)
parser.add_argument('--ipython', action='store_true')
parser.add_argument('--save-path', type=str, default=None)
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
        self.linear1 = nn.Linear(4 * args.max_value, args.dim)
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
        self.linear1 = nn.Linear(2 * args.max_value, args.dim // 2)
        self.linear2 = nn.Linear(2 * args.max_value, args.dim // 2)
        self.linear3 = nn.Linear(args.dim, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        x1_and_y1_part = torch.cat([h[:, :1 * args.max_value],
                                    h[:, 2 * args.max_value:3 * args.max_value]], 1)
        x2_and_y2_part = torch.cat([h[:, 1 * args.max_value:2 * args.max_value],
                                    h[:, 3 * args.max_value:]], 1)
        h1 = self.act(self.linear1(x1_and_y1_part))
        h2 = self.act(self.linear2(x2_and_y2_part))
        h = self.act(self.linear3(torch.cat([h1, h2], 1)))
        return self.output(h)


class WeakCheater(nn.Module):


    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3 * args.max_value, args.dim // 2)
        self.linear2 = nn.Linear(3 * args.max_value, args.dim // 2)
        self.linear3 = nn.Linear(args.dim, args.dim)
        self.output = nn.Linear(args.dim, 1)
        self.act = nn.ReLU()

    def __call__(self, h):
        x1_x2_y1_part = torch.cat([h[:, :2 * args.max_value],
                                    h[:, 2 * args.max_value:3 * args.max_value]], 1)
        x1_x2_y2_part = torch.cat([h[:, :2 * args.max_value],
                                    h[:, 3 * args.max_value:]], 1)
        h1 = self.act(self.linear1(x1_x2_y1_part))
        h2 = self.act(self.linear2(x1_x2_y2_part))
        h = self.act(self.linear3(torch.cat([h1, h2], 1)))
        return self.output(h)


stats = collections.defaultdict(list)
for seed in range(args.nseeds):
    torch.manual_seed(seed)
    rng = numpy.random.RandomState(seed)

    net = eval(args.model)()

    inputs = []
    # first train, then test
    positive_indices = [[], []]
    negative_indices = [[], []]
    r = range(args.max_value)
    for x1 in r:
        for x2 in r:
            for y1 in r:
                for y2 in r:
                    if args.split == 'none':
                        part = 0
                    elif args.split == 'random':
                        part = rng.randint(2)
                    if x1 == y1 and x2 == y2:
                        positive_indices[part].append(len(inputs))
                    else:
                        negative_indices[part].append(len(inputs))
                    inputs.append([x1, x2, y1, y2])
    inputs = numpy.array(inputs)
    positive_indices = list(map(numpy.array, positive_indices))
    negative_indices = list(map(numpy.array, negative_indices))
    if args.split == 'none':
        positive_indices[1] = positive_indices[0]
        negative_indices[1] = negative_indices[0]

    for i in range(args.nsteps):
        def get_batch(part):
            indices = numpy.concatenate([rng.choice(positive_indices[part], args.batch_size // 2),
                                        rng.choice(negative_indices[part], args.batch_size // 2)])
            x1, x2, y1, y2 = [torch.LongTensor(inputs[indices, j][:, None]) for j in range(4)]

            def onehot(x):
                o = torch.LongTensor(args.batch_size, args.max_value)
                o.zero_()
                o.scatter_(1, x, 1)
                return o
            ox1, ox2, oy1, oy2 = [onehot(v) for v in [x1, x2, y1, y2]]

            h = Variable(torch.cat([ox1, ox2, oy1, oy2], 1)).float()
            targets = 2 * Variable((x1 == y1) & (x2 == y2)).float() - 1
            return h, targets

        # train
        h, targets = get_batch(0)
        net.zero_grad()
        s = net(h)
        train_cost = torch.nn.Softplus()(s * -targets).mean()
        train_acc = (targets * s > 0).float().mean()
        train_cost.backward()
        for p in net.parameters():
            p.data -= args.lr * p.grad.data

        # test
        h, targets = get_batch(1)
        s = net(h)
        test_cost = torch.nn.Softplus()(s * -targets).mean()
        test_acc = (targets * s > 0).float().mean()

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
