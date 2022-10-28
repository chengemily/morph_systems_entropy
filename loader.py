"""
Loading utilities. Includes:

1. Conllu file reading
2. Filtering by part of speech
3.
"""

from conllu import parse
from conllu.models import TokenList
from scipy.stats import entropy
from matplotlib.pyplot import plt
import functools

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

    def plot_distribution(self):
        pass

    def filter_nouns_by_number_gender(self):
        """
        :return: Dict with keys "MascSing", "MascPlur", "FemSing", "FemPlur", and values = one TokenList containing
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

                # Filter out single letters (?)
                # TODO: filter out numbers
                filtered[gender + number] = [token['form'].lower() for token in list(filter(lambda x: len(x) > 1, final))]

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

    for fpath in ['./UD_French/fr_gsd-ud-train.conllu', './UD_Catalan/ca_ancora-ud-train.conllu', './UD_Spanish/es_ancora-ud-train.conllu']:
        data = UD_Dataset(fpath)
        print('File: ', fpath)
        print('Tokens')
        data.show_distribution(data.tokens)
        print('Types')
        data.show_distribution(data.types)