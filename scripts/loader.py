"""
Loading utilities. Includes:

1. Conllu file reading
2. Filtering by part of speech
3.
"""

from conllu import parse, parse_incr
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
TO_KEEP = set([
    'ADJ', 'ADV', 'NOUN', 'PROPN', 'INTJ', 'VERB'
])


class UD_Dataset:
    def __init__(self, filepath: str):
        # All data
        self.data = UD_Dataset.read_conllu_file(filepath)
        print('Sentence lengths:', sum([len(sentence) for sentence in self.data]) / len(self.data))
        self.nouns = [UD_Dataset.get_nouns(sentence) for sentence in self.data]
        self.tokens = self.filter_nouns_by_number_gender('form')
        self.types = self.dedup(self.tokens)

        self.tokens_lemmas = self.filter_nouns_by_number_gender('lemma')
        self.types_lemmas = self.dedup(self.tokens_lemmas)

    def context_entropy(self):
        #TODO
        pass

    def remove_token_frequency(freq=1):
        """Removes tokens from self.tokens that only occur once.

        Args:
            freq (int, optional): _description_. Defaults to 1.
        """
        pass

    def type_token_comparison(self):
        # plurals_tokens = set(self.tokens_lemmas['MascPlur']).union(set(self.tokens_lemmas['FemPlur']))
        plurals_types = set(self.types_lemmas['MascPlur']).union(set(self.types_lemmas['FemPlur']))
        # sing_tokens = set(self.tokens_lemmas['MascSing']).union(set(self.tokens_lemmas['FemSing']))
        sing_types = set(self.types_lemmas['MascSing']).union(set(self.types_lemmas['FemSing']))

        print('Size plur. types: ', len(plurals_types))
        print('Size sing. types: ', len(sing_types))

        print('Size intersection: ', len(plurals_types.intersection(sing_types)))
        print('Sing - plural: ', len(sing_types - plurals_types))
        print('Plural - sing: ', len(plurals_types - sing_types))

    def get_windows_around_nouns(self, window_size:int):
        # For each noun, get a window of
        pass

    
    def plot_noun_frequency_histogram(self, figname: str, token_dataset, type_dataset, lang=None):
        """
        :param figname: (str) tex file to save.
        :param token_dataset: Iterable(dict) The datasets of tokens (e.g. All nouns, Animate nouns) to plot pdf
        """
        # Count number
        data = []
        for dataset in [type_dataset, token_dataset]:
            counts = {k: len(dataset[k]) for k in dataset}
            total = sum([counts[k] for k in counts])
            frequencies = {k: counts[k] / total for k in counts}
            data.append(frequencies)

        labels = ['All nouns - Types', 'All nouns - Tokens']

        x = np.arange(len(labels))  # the label locations
        width = 0.15  # the width of the bars

        fig, ax = plt.subplots()
        ax.set_ylim(top=0.45)

        # create the bars
        rects = []
        for i, nountype in enumerate(type_dataset.keys()):
            values = [dataset[nountype] for dataset in data]
            rect = ax.bar(x + (2 * i - 3) * width / 2, values, width, label=nountype)
            rects.append(rect)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Probability Density')
        ax.set_title('Distribution of nouns across Gender and Number ({})'.format(lang))
        ax.set_xticks(x, labels)
        ax.legend()

        for rect in rects:
            ax.bar_label(rect, padding=3, fmt='%.2f')

        # Plot line at uniform distribution (y = 0.25)
        plt.axhline(y=0.25, color='r', linestyle='--')

        plt.savefig(figname)
        # tpl.save(f'{{{figname}}}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')

    def plot_token_frequency_pdf(self, figname: str, token_dataset: dict, sample=100, lang=None, save_tex=False):
        """
        :param figname: (str) tex file to save.
        :param token_dataset: (dict) The dataset of tokens (e.g. All nouns, Animate nouns) to plot pdf
        """
        fig, ax = plt.subplots()
        ax.set_xlim(right=7)
        ax.set_ylim(top=0.45)
        for nountype, tokens in token_dataset.items():
            if sample:
                unique_tokens = random.sample(set(tokens), sample)
            else:
                unique_tokens = set(tokens)

            token_counts = np.array([np.log(tokens.count(token)) for token in unique_tokens])
            loc = np.mean(token_counts)
            scale = np.std(token_counts)
            pdf = stats.norm.pdf(token_counts, loc=loc, scale=scale)
            sns.lineplot(x=token_counts, y=pdf, ax=ax, label=nountype)

        plt.legend()
        ax.set_title('All nouns - Token Frequency' + ('({})'.format(lang) if lang else ''))
        ax.set_xlabel('Token frequency of occurrence (log)')
        ax.set_ylabel('Probability Density')
        plt.show()
        plt.savefig(figname)
        if save_tex:
            tpl.save(f'{{{figname}}}.tex', axis_width=r'\figwidth', axis_height=r'\figheight')

    def filter_nouns_by_number_gender(self, kyword):
        """
        :param: kyword (str): e.g. 'form' (the string of the noun), 'lemma'
        :return: Dict with keys "MascSing", "MascPlur", "FemSing", "FemPlur", and values = one list containing
        all instances of each type of noun in the corpus.
        """
        filtered = {}

        for number in ['Sing', 'Plur']:
            for gender in ['Masc', 'Fem']:
                filtered_nouns = list(filter(
                    lambda x: len(x) > 0,
                    [sentence.filter(feats__Number=number, feats__Gender=gender) for sentence in self.nouns]
                ))  # list of TokenLists (want to combine)
                final = filtered_nouns[0]
                for i in range(1, len(filtered_nouns)):
                    final.extend(filtered_nouns[i])

                # Filter out single letters and numbers
                filtered[gender + number] = [
                    token[kyword].lower() for token in list(
                        filter(lambda x: len(x) > 1 and not any(char.isdigit() for char in x), final)
                    )
                ]

        return filtered

    def dedup(self, data):
        """
        :param data: self.tokens or self.tokens_lemmas
        :return: types
        """
        types = {}
        for category in self.tokens:
            types[category] = list(set(data[category]))
        return types

    def show_distribution(self, dataset):
        # Count number
        counts = {k : len(dataset[k]) for k in dataset}
        total = sum([counts[k] for k in counts])
        print('Total nouns: ', total)
        print("Counts: ", counts)

        # Make distribution over MS, MP, FS, FP
        frequencies = {k : counts[k] / total for k in counts}
        print("Frequencies: ", frequencies)

        # Print entropy
        freq_vector = [frequencies[k] for k in frequencies]
        print("Entropy: ", entropy(freq_vector))

        # Print KL
        print("KL: ", entropy(freq_vector, [0.25 for _ in range(4)]))

    @staticmethod
    def read_and_filter_conllu(filepaths: Iterable[str]):
        """_summary_

        Args:
            filepaths (Iterable[str]): _description_

        Returns:
            _type_: _description_
        """
        lines = []

        for filepath in tqdm(filepaths):
            with open(filepath, 'r') as file:
                line = file.readline()
                while line:
                    line = file.readline()
                    split_line = set(line.split('\t'))

                    if len(TO_KEEP.intersection(split_line)): 
                        lines.append(line)
                        continue
                    elif len(TO_FILTER.intersection(split_line)): 
                        continue
                    lines.append(line)
            file.close()

        data = '\n'.join(lines) 

    @staticmethod
    def read_conllu_file(filepaths: Iterable[str]):
        """
        Loads file as list of sentences, where each sentence is a TokenList.

        E.g.

        TokenList<The, quick, brown, fox, jumps, over, the, lazy, dog, ., metadata={text: "The quick brown fox jumps over the lazy dog."}>

        :param filepath: (str)
        :return: parsed file as a list of sentences. Each sentence represented by a TokenList.
        """
        sentences = TokenList()

        for filepath in tqdm(filepaths):
            data_file = open(filepath, 'r', encoding='utf-8')
            for sentence in parse_incr(data_file): sentences.extend(sentence)
            data_file.close()

        print('parsed conllu')
        return sentences

    @staticmethod
    def get_nouns(sentence: TokenList):
        """
        Filters sentence
        :param sentence: (TokenList)
        :return:
        """
        return sentence.filter(upos='NOUN')


if __name__=='__main__':

    # for fpath in [
    #     '/home/echeng/morph_systems_entropy/wiki/ca_0_wikipedia.conllu'
    #     # './wiki/fr_wikipedia.conllu'
    #    # './UD_Italian/it_isdt-ud-train.conllu',
    #    # './UD_French/fr_gsd-ud-train.conllu',
    #    # './UD_Catalan/ca_ancora-ud-train.conllu',
    #    # './UD_Spanish/es_ancora-ud-train.conllu'

    # ]:
        # print('File: ', fpath)
    fpaths = ['/home/echeng/morph_systems_entropy/wiki/filtered_ca_{}_wikipedia.conllu'.format(i) for i in range(5)]
    # lang = fpath.split('/')[1].split('_')[1]
    lang = 'ca'
    data = UD_Dataset(fpaths)
    data.type_token_comparison()
    print('----------------------------------------------')
    print('Tokens')
    data.show_distribution(data.tokens)
    print('----------------------------------------------')
    print('Types')
    data.show_distribution(data.types)

    print('==================================================================================')
    data.plot_token_frequency_pdf('token_frequency_{}.png'.format(lang), data.tokens, lang=lang)
    data.plot_noun_frequency_histogram('noun_frequency_histogram_{}.png'.format(lang), data.tokens, data.types, lang=lang)
