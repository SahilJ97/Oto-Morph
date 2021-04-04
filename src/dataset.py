"""Written by Tiansheng Sun"""

import math
import json
import torch
import os
import numpy as np
from glob import glob

from torch.utils.data import DataLoader, Dataset

SUPERSCRIPTS = ["⁰", "¹", "²", "³", "⁴", "⁵", "ᐟ", "ᵈ", "ᵖ", "ˀ", "'"]
CHARACTER_SEQUENCE_LENGTH = 27
NUM_TAG = 5


class OtoMangueanDataset(Dataset):

    def __init__(self, file_names):
        self.language, self.lemma, self.inflected, self.tags = np.array([]), np.array([]), np.array([]), np.array([])
        self.character_set, self.tags_set = set(), set()
        self.n_samples = 0

        # Load data
        for file in file_names:
            curr = np.loadtxt(file, delimiter="\t", dtype=np.str, encoding="utf-8")
            file_name = os.path.splitext(os.path.basename(file))[0]
            curr_shape = curr.shape[0]
            self.n_samples += curr_shape

            self.language = np.concatenate((self.language, np.array([file_name] * curr_shape)))
            current_lemma = curr[:, 0]
            current_inflected = curr[:, 1]
            current_tag = curr[:, 2]
            self.lemma = np.concatenate((self.lemma, current_lemma))
            self.inflected = np.concatenate((self.inflected, current_inflected))
            self.tags = np.concatenate((self.tags, current_tag))

        # Load dictionaries
        dict_names = glob('../dictionaries/*.json')
        for d in dict_names:
            dict_name = os.path.splitext(os.path.basename(d))[0]
            file_dict = json.load(open(d, encoding="utf-8"))
            setattr(self, dict_name, file_dict)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        current_tagset = self.tags[index].split(";")
        tag_len = len(self.tags_to_index)
        character_len = len(self.character_to_index)

        encoded_word, encoded_tag = [], []
        one_hot_tag = [0] * tag_len
        one_hot_word = [0] * character_len

        current_lemma = self.lemma[index]
        current_inflected = self.inflected[index]

        # Get character sequences
        curr_word = 0
        for word in [current_lemma, current_inflected]:
            lemma_len = len(word)
            curr_index = 0
            while curr_index < lemma_len:
                temp = word[curr_index]
                while curr_index + 1 < lemma_len and word[curr_index + 1] in SUPERSCRIPTS:
                    temp += word[curr_index + 1]
                    curr_index += 1
                one_hot_word[self.character_to_index[temp]] = 1
                encoded_word.append(one_hot_word)
                curr_index += 1
                one_hot_word = [0] * character_len
            pad_length = CHARACTER_SEQUENCE_LENGTH - len(encoded_word)
            if curr_word == 0:  # pre-pad for input sequence
                encoded_lemma = [[0] * character_len] * pad_length + encoded_word
            else:  # post-pad for output sequence
                encoded_inflection = encoded_word + [[0] * character_len] * pad_length
            curr_word += 1
            encoded_word = []

        # Get tag sequence
        if len(current_tagset) != NUM_TAG:
            encoded_tag += [one_hot_tag] * (NUM_TAG - len(current_tagset))  # what's going on here...

        for tag in current_tagset:
            one_hot_tag[self.tags_to_index[tag]] = 1
            encoded_tag.append(one_hot_tag)
            one_hot_tag = [0] * tag_len

        return {
                    "language": torch.tensor(self.language_to_index[self.language[index]]),
                    "character_sequence": torch.tensor(encoded_lemma),
                    "tagset": torch.tensor(encoded_tag)
               }, torch.tensor(encoded_inflection, dtype=torch.float)


if __name__ == '__main__':
    dataset = OtoMangueanDataset(glob('../data/*.trn'))
    print(len(dataset))
    test_datapoint, test_output = dataset[1000]
    print(test_datapoint)
