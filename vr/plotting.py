import os
import json
from matplotlib import pyplot
import pandas
import scipy.stats as stats


def load_log(root, file_, data_train, data_val, args):
    slurmid = file_[:-8]
    path = os.path.join(root, file_)
    log = json.load(open(path))

    args[root][slurmid] = log['args']

    for t, train_loss in zip(log['train_losses_ts'], log['train_losses']):
        data_train['root'].append(root)
        data_train['slurmid'].append(slurmid)
        data_train['step'].append(t)
        data_train['train_loss'].append(train_loss)

    assert len(log['val_accs_ts']) == len(log['val_accs'])
    assert len(log['val_accs_ts']) == len(log['train_accs'])
    for t, val_acc, train_acc in zip(log['val_accs_ts'], log['val_accs'], log['train_accs']):
        data_val['root'].append(root)
        data_val['slurmid'].append(slurmid)
        data_val['step'].append(t)
        data_val['val_acc'].append(val_acc)
        data_val['train_acc'].append(train_acc)


def load_logs(root, data_train, data_val, args):
    for root, dirs, files in os.walk(root):
        for file_ in files:
            if file_.endswith('pt.json'):
                load_log(root, file_, data_train, data_val, args)


def plot_average(df, train_quantity='train_acc', val_quantity='val_acc', window=None):
    pyplot.figure(figsize=(15, 5))
    df_mean = df.groupby(['root', 'step']).agg(['mean', 'std'])
    for root, df_root in df_mean.groupby('root'):
      train_values = df_root[train_quantity]['mean']
      if window:
        train_values = train_values.rolling(window).mean()
      train_lines = pyplot.plot(df_root.index.get_level_values(1),
                                train_values,
                                label=root + ' train',
                                linestyle='dotted')
      if val_quantity:
        val_values = df_root[val_quantity]['mean']
        val_std = df_root[val_quantity]['std']
        if window:
          val_values = val_values.rolling(window).mean()
          val_std = val_std.rolling(window).mean()
        pyplot.plot(df_root.index.get_level_values(1),
                    val_values,
                    label=root + " val",
                    color=train_lines[0].get_color())
      n_seeds = len(df[df['root'] == root]['slurmid'].unique())
      to_print = [root, "{} seeds".format(n_seeds), 100 * train_values.iloc[-1]]
      if val_quantity:
        std = val_std.iloc[-1]
        width = std * stats.t.ppf(0.975, n_seeds - 1) / (n_seeds ** 0.5)
        to_print.append("{}+-{}".format(100 * val_values.iloc[-1], 100 * width))
      print(*to_print)
    pyplot.legend()


def plot_all_runs(df, train_quantity='train_acc', val_quantity='val_acc', color=None):
    kwargs = {}
    if color:
        kwargs['color'] = color
    for (root, slurmid), df_run in df.groupby(['root', 'slurmid']):
        path = root + ' ' + slurmid
        train_lines = pyplot.plot(df_run['step'],
                                  df_run[train_quantity],
                                  label=path + ' train',

                                  linestyle='dotted',
                                  **kwargs)
        if val_quantity:
          pyplot.plot(df_run['step'],
                      df_run[val_quantity],
                      label=path + ' val',
                      color=train_lines[0].get_color())
        to_print = [path, df_run['step'].iloc[-1], df_run[train_quantity].iloc[-1]]
        if val_quantity:
          to_print.append(df_run[val_quantity].iloc[-1].mean())
        print(*to_print)
