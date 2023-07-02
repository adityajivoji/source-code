from typing import Any
import pandas as pd
import numpy as np
import os
import random

class preprocessor(object):
    def __init__(self, file_name, data_root='dataset/',delimiter='\t', past_length = 8, future_length = 12, normalize = False, traj_scale=1, split = 'train', train_ratio = 0.8, test_ratio = 0.1) -> None:
        self.file_name = file_name
        self.data_root = data_root
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),self.data_root, self.file_name)
        self.delimiter = delimiter
        self.past_length = past_length
        self.future_length = future_length
        self.normalize = normalize
        self.traj_scale = traj_scale
        self.split = split
        self.skiprows = 10
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.metadata = self.get_metadata()
        self.max_num_agents = 0
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.generate_data()
    
    
    def generate_data(self):
        self.chunks_to_traj(self.create_chunks())
    
    def validate_trajectory(self, trajectory):
        # conditions to validate a trajectory
        return True
    
    def save_as_numpy(self, trajectories):
        train, num_train = [], []
        test, num_test = [], []
        val, num_val = [], []
        for traj in trajectories:
            # print(traj.shape)
            num_agent = traj.shape[0]
            pad_width = ((0, self.max_num_agents - num_agent), (0, 0), (0, 0))
            traj = np.pad(traj, pad_width, mode='constant', constant_values=0)
            rand_num = random.random()
            if rand_num < self.train_ratio:
                train.append(traj)
                num_train.append(num_agent)
            elif rand_num < (self.train_ratio + self.test_ratio):
                test.append(traj)
                num_test.append(num_agent)
            else:
                val.append(traj)
                num_val.append(num_agent)
                
        destination_folder = [os.path.join(os.path.dirname(os.path.abspath(__file__)),"processed", split) for split in ['train', 'test', 'val']]
        for folder in destination_folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
            split = folder.split('/')[-1]
            if split == "train":
                np.save(os.path.join(folder, self.file_name[:-4]+'_data_train.npy'), train)
                np.save(os.path.join(folder, self.file_name[:-4]+'_num_train.npy'), num_train)
                
            elif split == "test":
                np.save(os.path.join(folder, self.file_name[:-4]+'_data_test.npy'), test)
                np.save(os.path.join(folder, self.file_name[:-4]+'_num_train.npy'), num_test)
                
            else:
                np.save(os.path.join(folder, self.file_name+'data_val.npy'), val)
                np.save(os.path.join(folder, self.file_name+'num_train.npy'), num_val)
                
        
    def chunks_to_traj(self, chunks):
        trajectories = []
        total_length = self.past_length + self.future_length
        for trajectory_name in list(chunks.keys()):
            chunk_metadata = chunks[trajectory_name][0]
            chunk_traj = chunks[trajectory_name][1]
            num_traj = chunk_metadata["length"] // total_length
            
            if chunk_metadata["num_agents"] > self.max_num_agents:
                self.max_num_agents = chunk_metadata["num_agents"]
                
            # print("num_traj", num_traj, chunk_metadata["length"], type(chunk_metadata["length"]), "total_length", total_length)
            # these many trajectories can be extracted from this chunk
            for i in range(num_traj):
                trajectory = [] # storing the trajectory of all the agents that are moving
                for markers in (chunk_traj.keys()):
                    traj = np.array(chunk_traj[markers][total_length*i:total_length*(i+1)])
                    # print("traj.shape", traj.shape)
                    if self.validate_trajectory(traj):
                        trajectory.append(traj)
                # print("trajectory", np.array(trajectory).shape)
                # print("trajectory", np.array(trajectory)[None].shape)
                
                trajectories.append(np.array(trajectory))
        self.save_as_numpy(trajectories)
                        
    def get_metadata(self):
        data_dict = {}
        with open(self.data_path, 'r') as file:
            for i, line in enumerate(file):
                line = line.strip()  # Remove leading/trailing whitespace and newline character
                if i < 9:
                    # Split the line into key and value
                    key, value = line.split('\t', 1)
                    data_dict[key] = value
                elif i == 9:
                    # Split the line into key and value as list
                    key, values = line.split('\t', 1)
                    data_dict[key] = values.split('\t')
        return data_dict
    
    def create_chunks(self,):
        # Read the dataset file
        df = pd.read_csv(self.data_path, sep=self.delimiter, skiprows=10)
        df = df.dropna(axis=1)
        data_dict = self.metadata
        # Initialize variables
        active_members = set()
        chunks = {}
        trajectory_num = 0
        start_frame = None
        markers = data_dict["MARKER_NAMES"]
        # Iterating over every row
        current_trajectory = dict()

        for i in range(0, int(self.metadata['NO_OF_FRAMES']), self.skiprows):
            row = df.iloc[i]
            new_active_mem = set()
            # finding active members
            for marker in markers:
                x, y, z = row[marker + ' X'], row[marker + ' Y'], row[marker + ' Z']
                if x > 0 and y > 0 and z > 0:
                    new_active_mem.add(marker)
            if active_members == new_active_mem:
                # continue with previous trajectory
                for marker in active_members:
                    current_trajectory[marker].append((
                        row[marker + ' X'], 
                        row[marker + ' Y'],
                        row[marker + ' Z']))

            else:
                # start new trajectory
                # save previous trajectory if length greater than 20
                if len(active_members)>0 and len(current_trajectory[list(active_members)[0]]) >= 20:
                    chunks[f'trajectory_{trajectory_num}'] = [{
                        'start_frame': start_frame,
                        'end_frame': row['Frame'],
                        'num_agents': len(active_members),
                        'length': len(current_trajectory[list(active_members)[0]])},
                        {
                            marker: current_trajectory[marker] for marker in active_members}]
                # starting new trajectory
                trajectory_num += 1
                current_trajectory = dict()
                start_frame = row["Frame"]
                active_members = new_active_mem
                for marker in active_members:
                    current_trajectory[marker] = [(
                        row[marker + ' X'],
                        row[marker + ' Y'],
                        row[marker + ' Z'])]
        return chunks
    
if __name__ == "__main__":
    preprocess = preprocessor(file_name='dataset.tsv', data_root='./')
    preprocess()


