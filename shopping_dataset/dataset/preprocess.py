# Importing libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import seaborn as sns
from tqdm import tqdm
import argparse
from datetime import timedelta
import pandas as pd
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import *
import os

def main():
    parser = argparse.ArgumentParser(description="arguments for preprocessing")
    parser.add_argument('--subset',
                        type=str,
                        default='eth',
                        help='Name of the subset.(german_1)')
    parser.add_argument('--resting_time',
                        type=int,
                        default=5,
                        help='Name of the subset.')
    parser.add_argument('--single_agent',
                        action='store_true',
                        default= True,
                        help='number of agents')
    parser.add_argument('--min_length',
                        action='store_true',
                        default= True,
                        help='total length of trajectory to be valid past + future length')
    parser.add_argument('--include_halt',
                        action='store_true',
                        default= False,
                        help='end trajectory if agent stops')
    parser.add_argument('--create_split',
                        action='store_true',
                        default= True,
                        help='divide into train test and val')
    args = parser.parse_args()


    data_path = f"./shopping_dataset/{args.subset}/{args.subset}.txt"
    processed_data_folder_path = f"./shopping_dataset/{args.subset}"
    
    data = pd.read_csv(data_path, sep=";")
    print(f"Total Duplicate Rows: {data[data.duplicated()]}")
    print("Dropping Duplicate Entries")
    # removing duplicate rows entries
    data = data.drop_duplicates(subset=['tag_id', 'time'])
    if args.single_agent:
        if args.consider_halt:
            trajectory_data_generation(data, processed_data_folder_path, args)
            # convert to machine readable format
        else:
            trajectory_data_generation_no_rest(data, processed_data_folder_path)
    else:
        pass


if __name__ == "__main__":
    main()
         
    
