"""Written by Tiansheng Sun"""

import json
from glob import glob
import csv
import os

SUPERSCRIPTS = ["⁰", "¹", "²", "³", "⁴", "⁵", "ᐟ", "ᵈ", "ᵖ", "ˀ", "'"]

if __name__ == '__main__':
    file_names = glob('data/*.trn')
    language_type = []
    character_set, tags_set = {'[PAD]'}, set()

    # Collect character set
    for file in file_names:
        with open(file, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            language_type.append(os.path.splitext(os.path.basename(file))[0])
            for row in csv_reader:
                for dict_type in [row[0], row[1]]:
                    character_len = 0
                    index = 0
                    while index < len(dict_type):
                        character_len += 1
                        temp = dict_type[index]
                        while index + 1 < len(dict_type) and dict_type[index + 1] in SUPERSCRIPTS:
                            temp += dict_type[index + 1]
                            index += 1

                        character_set.add(temp)
                        index += 1

                for t in row[2].split(";"):
                    tags_set.add(t)

    # Generate dictionaries
    lan_types = sorted(language_type)
    index_to_language = dict(enumerate(lan_types))
    language_to_index = {t: i for i, t in enumerate(lan_types)}
    character_sorted = sorted(list(character_set))
    index_to_character = dict(enumerate(character_sorted))
    character_to_index = {t: i for i, t in enumerate(character_sorted)}
    tags_sorted = sorted(list(tags_set))
    index_to_tags = dict(enumerate(tags_sorted))
    tags_to_index = {t: i for i, t in enumerate(tags_sorted)}

    dict_types = ["index_to_language", "language_to_index", "index_to_character", "character_to_index", "index_to_tags",
                  "tags_to_index"]

    for dict_type in dict_types:
        output_fname = "dictionaries/" + dict_type + ".json"
        with open(output_fname, "w+", encoding="utf-8") as outfile:
            curr = json.dumps(eval(dict_type), ensure_ascii=False)
            outfile.write(curr)
