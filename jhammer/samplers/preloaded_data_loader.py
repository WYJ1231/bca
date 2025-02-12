from collections import OrderedDict

import numpy as np
import torch


class PreloadedDataLoader:
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset = iter(dataset)

    def __next__(self):
        return self.load_batch()

    def __iter__(self):
        return self

    def load_batch(self):
        batch_dict = OrderedDict()
        batch_is_dict = False
        for _ in range(self.batch_size):
            batch = next(self.dataset)
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch)
                if 0 not in batch_dict:
                    batch_dict[0] = []
                    batch_dict[0].append(batch)
            elif type(batch) in (int, tuple, set, list):
                for i, ele in enumerate(batch):
                    if isinstance(ele, np.ndarray):
                        ele = torch.from_numpy(ele)
                    if i not in batch_dict:
                        batch_dict[i] = []
                    batch_dict[i].append(ele)

            elif isinstance(batch, dict):
                batch_is_dict = True
                for k, v in batch.items():
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v)

                    if k not in batch_dict:
                        batch_dict[k] = []
                    batch_dict[k].append(v)

        if batch_is_dict:
            for k, v in batch_dict.items():
                if isinstance(v[0], torch.Tensor):
                    batch_dict[k] = torch.stack(v)
                else:
                    batch_dict[k] = v
            return batch_dict

        batch = []
        for v in batch_dict.values():
            if isinstance(v[0], torch.Tensor):
                batch.append(torch.stack(v))
            else:
                batch.append(v)
        return tuple(batch)
