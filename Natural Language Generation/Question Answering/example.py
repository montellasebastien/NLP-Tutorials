# coding: utf-8
from transformers import *
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, file_path):
        # TODO Load file -> checking for evaluation

    def __len__(self):
        # TODO

    def __getitem__(self, idx):
        # TODO return the idx'th sample



