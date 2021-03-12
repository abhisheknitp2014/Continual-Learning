from torchvision import datasets, transforms
import random
import torch
import torch.utils.data as data

from os import listdir

import numpy as np

def slit_train_val(input_path, target_path, phase):
    input_data = np.load(input_path)
    target_data = np.load(target_path)
    input_data = transpose_data(input_data)
    # target_data = transpose_data(target_data)
    data_size = input_data.shape[2]
    print("total data size is " + str(data_size))

    # shuffle_id = list(range(data_size))
    # random.shuffle(shuffle_id)
    
    # train_size = int(0.9*data_size)
    # print("total training data size from DataLoader is " + str(train_size))
    # train_id = shuffle_id[:train_size]
    # val_id = shuffle_id[train_size:]
    return input_data, target_data, data_size
    # if phase == "train":
    #     return input_data[train_id,:,:], target_data[train_id]
    # elif phase == "test":
    #     return input_data[val_id,:,:], target_data[val_id]
    # else:
    #     print("Wrong phase information")
    #     return None

def transpose_data(data):
    return np.transpose(data, (1,2,0))

class DataLoader(data.Dataset):
    """docstring for DataLoader"""
    def __init__(self, input_path, target_path, phase):
        super(DataLoader, self).__init__()
        self.input, self.target, self.data_size= slit_train_val(input_path, target_path, phase)
        
    def __getitem__(self, index):
        return self.input[index,:,:], self.target[index]

    def __len__(self):
        return self.input.shape[2]


def get_train_data(input_path, target_path):
    return DataLoader(input_path, target_path, 'train')

def get_test_data(input_path, target_path):
    return DataLoader(input_path, target_path, 'test')