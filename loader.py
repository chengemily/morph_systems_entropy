"""
Loading utilities. Includes:

1. Conllu file reading
2. Filtering by part of speech
3.
"""

from conllu import parse
from conllu.models import TokenList
from scipy.stats import entropy
import matplotlib.pyplot as plt
import functools
import tikzplotlib as tpl
from scipy import stats
import numpy as np
import seaborn as sns

class UD_Dataset:
    def __init__(self, filepath: str):
        # All data
        self.data = UD_Dataset.read_conllu_file(filepath)
        print('Sentence lengths:', sum([len(sentence) for sentence in self.data]) / len(self.data))
        self.nouns = [UD_Dataset.get_nouns(sentence) for sentence in self.data]
        self.tokens = self.filter_nouns_by_number_gender()
        self.types = self.dedup()

        # self.context_entropy_data =

    def get_windows_around_nouns(self, window_size:int):
        # For each noun, get a window of
        pass

    # TODO
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

    def plot_token_frequency_pdf(self, figname: str, token_dataset: dict, lang=None, save_tex=False):
        """
        :param figname: (str) tex file to save.
        :param token_dataset: (dict) The dataset of tokens (e.g. All nouns, Animate nouns) to plot pdf
        """
        fig, ax = plt.subplots()
        ax.set_xlim(right=7)
        ax.set_ylim(top=0.45)
        for nountype, tokens in token_dataset.items():
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

    # TODO
    def plot_types_vs_tokens(self):
        pass

    def filter_nouns_by_number_gender(self):
        """
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
                    token['form'].lower() for token in list(
                        filter(lambda x: len(x) > 1 and not any(char.isdigit() for char in x), final)
                    )
                ]

        return filtered

    def dedup(self):
        self.types = {}
        for category in self.tokens:
            self.types[category] = list(set(self.tokens[category]))
        return self.types

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
    def read_conllu_file(filepath: str):
        """
        Loads file as list of sentences, where each sentence is a TokenList.

        E.g.

        TokenList<The, quick, brown, fox, jumps, over, the, lazy, dog, ., metadata={text: "The quick brown fox jumps over the lazy dog."}>

        :param filepath: (str)
        :return: parsed file as a list of sentences. Each sentence represented by a TokenList.
        """
        with open(filepath, 'r') as file:
            data = file.read()
        sentences = parse(data)
        file.close()
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

    for fpath in [
        './UD_Italian/it_isdt-ud-train.conllu',
        './UD_French/fr_gsd-ud-train.conllu',
        './UD_Catalan/ca_ancora-ud-train.conllu',
        './UD_Spanish/es_ancora-ud-train.conllu'
    ]:
        print('File: ', fpath)
        lang = fpath.split('/')[1].split('_')[1]
        data = UD_Dataset(fpath)
        # print('----------------------------------------------')
        # print('Tokens')
        # data.show_distribution(data.tokens)
        # print('----------------------------------------------')
        # print('Types')
        # data.show_distribution(data.types)
        # print('==================================================================================')
        # data.plot_token_frequency_pdf('token_frequency_{}.png'.format(lang), data.tokens, lang=lang)
        data.plot_noun_frequency_histogram('noun_frequency_histogram_{}.png'.format(lang), data.tokens, data.types, lang=lang)