import math

import torch
import os
import numpy as np
from glob import glob

from torch.utils.data import DataLoader, Dataset

SUPERSCRIPTS = ["⁰", "¹", "²", "³", "⁴", "⁵", "ᐟ", "ᵈ", "ᵖ", "ˀ", "'"]
CHARACTER_SEQUENCE_LENGTH = 27
NUM_TAG = 5


class otoMangueanDataset(Dataset):

    def __init__(self):
        self.language_type = []
        self.language, self.lemma, self.inflected, self.tags = np.array([]), np.array([]), np.array([]), np.array([])
        self.character_set, self.tags_set = set(), set()
        self.n_samples = 0

        # data loading
        file_names = glob('data/*.trn')
        for file in file_names:
            curr = np.loadtxt(file, delimiter="\t", dtype=np.str, encoding="utf-8")
            curr_shape = curr.shape[0]
            self.n_samples += curr_shape
            self.language_type.append(os.path.splitext(os.path.basename(file))[0])
            self.language = np.concatenate((self.language, np.array([self.language_type[-1]] * curr_shape)))

            current_lemma = curr[:, 0]
            current_inflected = curr[:, 1]
            current_tag = curr[:, 2]

            # iterate through lemmas and inflected forms
            self.lemma = np.concatenate((self.lemma, current_lemma))
            self.inflected = np.concatenate((self.inflected, current_inflected))
            for i in np.concatenate((current_lemma, current_inflected)):
                character_len = 0
                index = 0
                while index < len(i):
                    character_len += 1
                    temp = i[index]
                    while index + 1 < len(i) and i[index + 1] in SUPERSCRIPTS:
                        temp += i[index + 1]
                        index += 1

                    self.character_set.add(temp)
                    index += 1

            # iterate through tags
            self.tags = np.concatenate((self.tags, current_tag))
            for t in current_tag:
                current = t.split(";")
                for i in current:
                    self.tags_set.add(i)

        # generate dictionaries for use
        self.lan_types = sorted(self.language_type)
        self.index_to_language = dict(enumerate(self.lan_types))
        self.language_to_index = {t: i for i, t in enumerate(self.lan_types)}
        character_sorted = sorted(list(self.character_set))
        self.index_to_character = dict(enumerate(character_sorted))
        self.character_to_index = {t: i for i, t in enumerate(character_sorted)}
        tags_sorted = sorted(list(self.tags_set))
        self.index_to_tags = dict(enumerate(tags_sorted))
        self.tags_to_index = {t: i for i, t in enumerate(tags_sorted)}

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        current_tag = self.tags[index].split(";")
        tag_len = len(self.tags_to_index)
        character_len = len(self.character_to_index)

        encoded_word, encoded_tag = [], []
        one_hot_tag = [0] * tag_len
        one_hot_word = [0] * character_len

        current_lemma = self.lemma[index]
        current_inflected = self.inflected[index]

        # get character sequences
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
            padding_number = CHARACTER_SEQUENCE_LENGTH - len(encoded_word)
            if curr_word == 0:
                encoded_lemma = [[0] * character_len] * padding_number + encoded_word
            else:
                encoded_inflection = encoded_word + [[0] * character_len] * padding_number
            curr_word += 1
            encoded_word = []

        # get tag sequence
        if len(current_tag) != NUM_TAG:
            encoded_tag += [one_hot_tag] * (NUM_TAG - len(current_tag))

        for l in current_tag:
            one_hot_tag[self.tags_to_index[l]] = 1
            encoded_tag.append(one_hot_tag)
            one_hot_tag = [0] * tag_len

        return {"language": self.language_to_index[self.language[index]], "character_sequence": np.array(encoded_lemma),
                "tags": np.array(encoded_tag)}, np.array(encoded_inflection)


def run():
    torch.multiprocessing.freeze_support()


if __name__ == '__main__':
    run()
    dataset = otoMangueanDataset()
    dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True, num_workers=2)

    # training loop
    num_epoch = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/5)

    for epoch in range(num_epoch):
        for i, (input, output) in enumerate(dataloader):
            continue

