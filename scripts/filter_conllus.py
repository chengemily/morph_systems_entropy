from conllu import parse
from conllu.models import TokenList
from scipy.stats import entropy
import matplotlib.pyplot as plt
import functools
import tikzplotlib as tpl
from scipy import stats
import numpy as np
import seaborn as sns
import time 
import random
from tqdm import tqdm

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

# https://universaldependencies.org/u/pos/index.html
# Filter everything that's not on the left column
TO_FILTER = set([
    'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PRON', 'SCONJ', 'PART', 'PUNCT', 'SYM', 'X'
])
# TO_KEEP = set([
    # 'ADJ', 'ADV', 'NOUN', 'PROPN', 'INTJ', 'VERB'
# ])

TO_FILTER = set([
    'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PRON', 'SCONJ', 'PART', 'PUNCT', 'SYM', 'X',
    'ADJ', 'ADV', 'PROPN', 'INTJ', 'VERB'
])

TO_KEEP = set([
    'NOUN'
])


if __name__=='__main__':
    lang = 'ca'
    fpaths = ['/home/echeng/morph_systems_entropy/wiki/{}_{}_wikipedia.conllu'.format(lang, i) for i in range(5)]
    writepaths = ['/home/echeng/morph_systems_entropy/wiki/nouns_{}_{}_wikipedia.conllu'.format(lang, i) for i in range(5)]

    for i, filepath in tqdm(enumerate(fpaths)):
        lines = []
        with open(filepath, 'r') as file:
            line = True
            while line is not None:
                line = file.readline()
                split_line = set(line.split('\t'))

                if len(TO_KEEP.intersection(split_line)): 
                    lines.append(line)
                elif len(TO_FILTER.intersection(split_line)): 
                    continue
                else:
                    lines.append(line)

        data = ''.join(lines) 
        print(data[:100])
        print('WRITING')
        with open(writepaths[i], 'w') as file:
            file.write(data)

    # Test that we can open and parse the file (this works.)
    # with open(writepaths[0], 'r') as file:
    #     data = file.read()
    #     conll = parse(data)