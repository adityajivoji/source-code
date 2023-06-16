import os, numpy as np, copy
import sys 
sys.path.append("..")
import torch
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import random
import os
import torch
from torch.utils.data import Dataset

class Shopping_dataset(Dataset):
    def __init__(self, data_file_path, past_length, future_length):
        self.data_file = data_file_path
        self.past_length = past_length
        self.future_length = future_length

        # Check if the file exists
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"No file found at {self.data_file}")

        # Load the data from file
        self.dataset = np.load(self.data_file)

    def __getitem__(self, index):
        trajectory = self.dataset[index]

        # Convert to tensor and return
        return torch.from_numpy(trajectory)

    def __len__(self):
        return len(self.dataset)
    

class Data2Numpy:
    def __init__(self, subset, past_length, future_length, split):
        self.subset = subset
        self.past_length = past_length
        self.future_length = future_length
        self.split = split
        processed_data_folder_path = f"./preprocessed_dataset/{self.subset}"
        self.data_path = f"./shopping_dataset/{self.subset}"
        output_dir = os.path.dirname(processed_data_folder_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        assert subset in ['german_1', 'german_2','german_3', 'german_4']


    def generate_data(self):
        # data is of the format [tag_id', 'time', 'x', 'y', 'description', 'trajectory_name', 'point_type']
        # extract only tag_id, x, y, trajectory_name from self.data
        for split in self.split.keys:
            data = self.split[split][['tag_id', 'x', 'y', 'trajectory_name']].copy()
            grouped_data = data.groupby(['tag_id', 'trajectory_name'])[['x', 'y']].apply(lambda x: x.values.tolist()).to_dict()
            data_len = len(grouped_data)
            dataset = np.zeros((data_len, self.past_length + self.future_length, 2))
            for i, key in enumerate(grouped_data.keys()):
                trajectory = grouped_data[key]
                if len(trajectory) < self.past_length + self.future_length:
                    continue
                else:
                    dataset[i] = trajectory[:self.past_length + self.future_length]
            np.save(f"./preprocessed_dataset/{self.subset}/{split}.npy", dataset)





        



