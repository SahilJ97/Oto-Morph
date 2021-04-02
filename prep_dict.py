import json
from glob import glob
import csv
import os

SUPERSCRIPTS = ["⁰", "¹", "²", "³", "⁴", "⁵", "ᐟ", "ᵈ", "ᵖ", "ˀ", "'"]


if __name__ == '__main__':
    file_names = glob('data/*.trn')
    language_type = []
    character_set, tags_set = set(), set()

    for file in file_names:
        with open(file, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\t')
            language_type.append(os.path.splitext(os.path.basename(file))[0])
            for row in csv_reader:
                for i in [row[0], row[1]]:
                    character_len = 0
                    index = 0
                    while index < len(i):
                        character_len += 1
                        temp = i[index]
                        while index + 1 < len(i) and i[index + 1] in SUPERSCRIPTS:
                            temp += i[index + 1]
                            index += 1

                        character_set.add(temp)
                        index += 1

                for t in row[2].split(";"):
                    tags_set.add(t)

    # generate dictionaries for use
    lan_types = sorted(language_type)
    index_to_language = dict(enumerate(lan_types))
    language_to_index = {t: i for i, t in enumerate(lan_types)}
    character_sorted = sorted(list(character_set))
    index_to_character = dict(enumerate(character_sorted))
    character_to_index = {t: i for i, t in enumerate(character_sorted)}
    tags_sorted = sorted(list(tags_set))
    index_to_tags = dict(enumerate(tags_sorted))
    tags_to_index = {t: i for i, t in enumerate(tags_sorted)}

    file_dir = ["index_to_language", "language_to_index", "index_to_character", "character_to_index", "index_to_tags",
                "tags_to_index"]

    for i in file_dir:
        curr_dir = "dictionary/" + i + ".json"
        with open(curr_dir, "w", encoding="utf-8") as outfile:
            curr = json.dumps(eval(i), ensure_ascii=False)
            outfile.write(curr)


