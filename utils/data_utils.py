import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class BarCrawlData(Dataset):
    def __init__(self, data_folder: str,
                 annotations_file: str,
                 transforms=None,
                 target_dtype=torch.float32,
                 regression: bool = False,
                 standard_scale: bool = False,
                 sample_freq: float = 16
                 ):
        self.data_folder = data_folder
        self.data_files = [i for i in os.listdir(self.data_folder) if os.path.splitext(i)[1] == '.npy']
        self.annotations = pd.read_csv(annotations_file, index_col=0).loc[
                           [os.path.splitext(i)[0] for i in self.data_files], :]
        self.transforms = transforms
        self.regression = regression
        self.target_dtype = target_dtype
        self.standard_scale = standard_scale
        self.sample_freq = sample_freq

    def __getitem__(self, index):
        signal_path = os.path.join(self.data_folder, self.data_files[index])
        if self.regression:
            label = self.annotations.iloc[index, :].TAC
        else:
            label = self.annotations.iloc[index, :].over_limit

        signal = torch.tensor(np.load(signal_path), dtype=torch.float32)
        label = torch.tensor(label, dtype=self.target_dtype)
        if self.target_dtype == torch.float32:
            label = label.reshape(1)
        if self.transforms:
            signal = self.transforms(signal)
        if self.standard_scale:
            raise NotImplementedError
            # m = signal.mean(0, keepdim=True)
            # std = signal.std(0, keepdim=True)
            # signal -= m
            # signal /= std

        return signal, label

    def __len__(self):
        return len(self.data_files)

    def plot_item(self, index):
        dat, lab = self.__getitem__(index)
        start = self.annotations.iloc[index, :].start
        end = self.annotations.iloc[index, :].end
        times = pd.date_range(start, end, freq=f'{round(1 / self.sample_freq, 5)}S')[:-1]
        item_id = self.annotations.iloc[index, :].name

        plt.figure(figsize=(10, 3))
        for i, axis in zip(range(len(dat)), ['x', 'y', 'z']):
            plt.plot(times, dat[i, :], label=f'{axis}-axis')
        plt.title(f'{item_id} Signal, Response={lab.item()}')
        plt.legend()
        plt.tight_layout()
        plt.show()
