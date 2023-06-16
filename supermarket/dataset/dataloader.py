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

class Supermarket(Dataset):
    def __init__(self, data_folder_path, past_length, future_length, device):
        self.data_file = data_folder_path + 'train_data.npy'
        self.num_file = data_folder_path + 'train_num.npy'
        self.past_length = past_length
        self.future_length = future_length

        # Check if the file exists
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"No file found at {self.data_file} \n Consider Preprocessing")
        if not os.path.exists(self.num_file):
            raise FileNotFoundError(f"No file found at {self.num_file} \n Consider Preprocessing")

        # Load the data from file
        self.dataset = torch.Tensor(np.load(self.data_file)).to(device)
        self.num_agent = torch.Tensor(np.load(self.num_file)).to(device)
        assert self.dataset.shape[0] == self.num_agent.shape[0]

    def __getitem__(self, index):
        trajectory = self.dataset[index]
        num_agent = self.num_agent[index]

        # Convert to tensor and return
        return trajectory, num_agent

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
            num_data = np.ones((data_len))
            dataset = np.zeros((data_len, self.past_length + self.future_length, 2))
            for i, key in enumerate(grouped_data.keys()):
                trajectory = grouped_data[key]
                if len(trajectory) < self.past_length + self.future_length:
                    continue
                else:
                    dataset[i] = trajectory[:self.past_length + self.future_length]
            np.save(f"./preprocessed_dataset/{self.subset}/{split}_data.npy", dataset)
            np.save(f"./preprocessed_dataset/{self.subset}/{split}_num.npy", num_data)





        



