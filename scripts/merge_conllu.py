import conllu
from conllu import parse_incr

from io import open
import glob


def merge_list_of_files(files, dest_path):
    """
    :param: filepaths (list(str)): list of filepaths to merge, all ending in conllu
    """
    merged_tokenlists = []

    for file in files:
        data_file = open(file, "r", encoding="utf-8")

        for tokenlist in parse_incr(data_file):
                merged_tokenlists.append(tokenlist)

    with open(dest_path, 'w') as f:
        f.writelines([tokenlist.serialize() + "\n" for tokenlist in merged_tokenlists])


if __name__ == '__main__':
    merge_list_of_files(
        glob.glob('UD_French*/*.conllu'),
        'UD_French_all.conllu'
    )
