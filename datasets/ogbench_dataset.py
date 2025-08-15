import numpy as np
import torch
from torch.utils.data import Dataset


class OGBenchGCDataset(Dataset):
    """
    Goal-conditioned trajectory sampling for OGBench datasets.
    Mirrors the sampling rule in D4RLGCDataset:
    - Sample a stride `interval` and sub-sample a plan of length `horizon`.
    - Normalize a scalar `returns = interval / max_interval`.
    Expects a dataset dict with at least: observations, actions, rewards, terminals.
    """

    def __init__(self, dataset_dict, max_interval, horizon):
        data = dataset_dict
        self.episodes = {key: [] for key in data.keys()}
        self.episode_lengths = []

        terminals = data['terminals']
        start = 0
        for i in range(len(terminals)):
            if terminals[i]:
                end = i
                if end - start + 1 > horizon:
                    for key in data.keys():
                        self.episodes[key].append(data[key][start:end + 1])
                    self.episode_lengths.append(end - start + 1)
                start = i + 1

        self.max_interval = max_interval
        self.horizon = horizon

    def __len__(self):
        return int(1e6)

    def __getitem__(self, _):
        epi_i = np.random.randint(len(self.episode_lengths))
        length = self.episode_lengths[epi_i]
        max_interval = np.minimum(self.max_interval, length // (self.horizon - 1))
        interval = np.random.randint(max_interval) + 1
        t = np.random.randint(length - interval * (self.horizon - 1))
        timesteps = t + interval * np.arange(self.horizon)

        returns = interval / self.max_interval
        batch = {
            'observations': torch.as_tensor(self.episodes['observations'][epi_i][timesteps], dtype=torch.float32),
            'actions': torch.as_tensor(self.episodes['actions'][epi_i][timesteps], dtype=torch.float32),
            'returns': torch.as_tensor(returns, dtype=torch.float32),
        }
        return batch


