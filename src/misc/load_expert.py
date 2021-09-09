import itertools
import os
import random

import numpy as np
import joblib


def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def restore_latest_n_traj(dirnames,
                          n_path=10,
                          max_steps=None,
                          shuffle=False,
                          shuffle_n_path=0):
    if isinstance(dirnames, str):
        assert os.path.isdir(dirnames)
        filenames = get_filenames(dirnames, n_path)
    else:
        filenames = []
        for dirname in dirnames:
            assert os.path.isdir(dirname)
            filenames.extend(get_filenames(dirname, n_path))
    return load_trajectories(filenames,
                             max_steps=None,
                             shuffle=shuffle,
                             shuffle_n_path=shuffle_n_path)


def get_filenames(dirname, n_path=None):
    import re
    itr_reg = re.compile(
        # r"expert_epi_(?P<episodes>[0-9]+).pkl"
        r"expert_(?P<step>[0-9]+)_epi_(?P<episodes>[0-9]+)_return_(-?)(?P<return_u>[0-9]+).(?P<return_l>[0-9]+).pkl"
    )

    itr_files = []
    for _, filename in enumerate(os.listdir(dirname)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('episodes')
            itr_files.append((itr_count, filename))

    n_path = n_path if n_path is not None else len(itr_files)
    itr_files = sorted(itr_files, key=lambda x: int(x[0]),
                       reverse=True)[:n_path]
    filenames = []
    for itr_file_and_count in itr_files:
        filenames.append(os.path.join(dirname, itr_file_and_count[1]))
    return filenames


def load_trajectories(filenames,
                      max_steps=None,
                      shuffle_n_path=0,
                      shuffle=False):
    assert len(filenames) > 0
    paths = []
    for filename in filenames:
        paths.append(joblib.load(filename))

    n_paths = len(paths)
    assert shuffle_n_path <= n_paths, \
        f'shuffle_n_path={shuffle_n_path} should be less than n_path={n_paths}'
    if shuffle:
        assert shuffle_n_path > 0, \
            f'shuffle_n_path={shuffle_n_path} must be bigger than 1'

    def get_obs_and_act(path):
        obses = path['obs']
        next_obses = path['next_obs']
        next_obses = path['obs']
        actions = path['act']
        dones = path['done']
        logps = path['logp']
        if max_steps is not None:
            return obses[:max_steps], next_obses[:max_steps], \
                   actions[:max_steps - 1],  dones[:max_steps], \
                logps[:max_steps]
        else:
            return obses, next_obses, actions, dones, logps
            # return obses, next_obses, actions, dones

    # Collect data from given paths
    obses, next_obses, acts, dones, logps = [[]], [[]], [[]], [[]], [[]]
    # obses, next_obses, acts, dones = [[]], [[]], [[]], [[]]
    for i, path in enumerate(paths):
        obs, next_obs, act, done, logp = get_obs_and_act(path)
        # obs, next_obs, act, done = get_obs_and_act(path)
        obses[-1].append(obs)
        next_obses[-1].append(next_obs)
        acts[-1].append(act)
        dones[-1].append(done)
        logps[-1].append(logp)

        if i < n_paths - 1:
            obses.append([])
            next_obses.append([])
            acts.append([])
            dones.append([])
            logps.append([])

    # Shuffle paths if specified
    if shuffle:
        shuffle_idx = list(range(n_paths))
        random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:shuffle_n_path]
        obses = [obses[i] for i in shuffle_idx]
        next_obses = [next_obses[i] for i in shuffle_idx]
        acts = [acts[i] for i in shuffle_idx]
        dones = [dones[i] for i in shuffle_idx]
        logps = [logps[i] for i in shuffle_idx]

    obses = np.concatenate(list(itertools.chain(*obses)), axis=0)
    next_obses = np.concatenate(list(itertools.chain(*next_obses)), axis=0)
    acts = np.concatenate(list(itertools.chain(*acts)), axis=0)
    dones = np.concatenate(list(itertools.chain(*dones)), axis=0)
    logps = np.concatenate(list(itertools.chain(*logps)), axis=0)

    return {
        'obses': obses,
        'next_obses': next_obses,
        'acts': acts,
        'dones': dones,
        'logps': logps
    }
