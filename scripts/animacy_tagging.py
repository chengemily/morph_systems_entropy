import wn as wordnet
from nltk.tokenize import TreebankWordTokenizer
from nltk.wsd import lesk
from nltk.corpus import wordnet31 as wn
from nltk import sent_tokenize
from googletrans import Translator

from loader import UD_Dataset

TRANSLATOR = Translator()
ANIMATE_SYNSETS = [synset for synset in list(wn.all_synsets('n')) if synset.lexname() in 
    ['noun.animal', 'noun.person']
]
SYNSETS = [synset for synset in list(wn.all_synsets('n'))]
ANIMATE_SYNSETS = [synset for synset in SYNSETS if synset.lexname() in ['noun.animal', 'noun.person']]


GERMAN_SYNSETS = wordnet.synsets(pos='n', lexicon='odenet:1.4')
GERMAN_WORDS = wordnet.words(pos='n', lexicon='odenet:1.4')

de_lemmas_file = open('/home/echeng/morph_systems_entropy/german_animate_lemmas_from_english.txt', 'r')
GERMAN_ANIMATE_LEMMAS = de_lemmas_file.read().split('\n')
de_lemmas_file.close()

def get_all_lemmas(lang):
    # Special case for German as it's not integrated in nltk
    if lang not in ('de', 'deu'):
        return set([n.lower() for synset in SYNSETS for n in synset.lemma_names(lang)])
    else:
        return set([n.lemma().lower().replace(' ', '_') for n in GERMAN_WORDS])

def get_animate_lemmas(lang):
    """Returns all animate lemmas in the target language.

    Args:
        lang (str): One of {fra, ita, cat, spa}

    Returns:
        set: set of animate lemmas in the target language
    """
    if lang not in ('de', 'deu'):
        return set([n.lower() for synset in ANIMATE_SYNSETS for n in synset.lemma_names(lang)])
    else:        
        return GERMAN_ANIMATE_LEMMAS

def multilingual_lesk(context_sentence, ambiguous_word, lang, pos=None, synsets=None):
    """Return a synset for an ambiguous word in a context.
    :param iter context_sentence: The context sentence where the ambiguous word
         occurs, passed as a string of words.
    :param str ambiguous_word: The ambiguous word that requires WSD.
    :param str lang: e.g. 'en', 'fra'
    :param str pos: A specified Part-of-Speech (POS).
    :param iter synsets: Possible synsets of the ambiguous word.
    :return: ``lesk_sense`` The Synset() object with the highest signature overlaps.
    This function is an implementation of the original Lesk algorithm (1986) [1].
    Usage example::
        >>> lesk(['I', 'went', 'to', 'the', 'bank', 'to', 'deposit', 'money', '.'], 'bank', 'n')
        Synset('savings_bank.n.02')
    [1] Lesk, Michael. "Automatic sense disambiguation using machine
    readable dictionaries: how to tell a pine cone from an ice cream
    cone." Proceedings of the 5th Annual International Conference on
    Systems Documentation. ACM, 1986.
    https://dl.acm.org/citation.cfm?id=318728
    """
    # Translate context to English to use with synsets
    english_sentence = TRANSLATOR.translate(context_sentence, src=lang, dest='en')

    context = set(context_sentence)
    if synsets is None:
        synsets = wn.synsets(ambiguous_word, lang=lang)

    if pos:
        synsets = [ss for ss in synsets if str(ss.pos()) == pos]

    if not synsets:
        return None

    _, sense = max(
        (len(context.intersection(ss.definition().split())), ss) for ss in synsets
    )

    return sense


if __name__=='__main__':     
    fpath = '/home/echeng/morph_systems_entropy/wiki/it_9_wikipedia.conllu' #'/home/echeng/morph_systems_entropy/wiki/it_wikipedia.conllu'

    # Load in conllu file
    data = UD_Dataset([fpath])

    # For the list of animate nouns, get list of synsets
    animate_lemmas = set(get_animate_lemmas('ita'))
    print(animate_lemmas)

    # Refine the set of animate lemmas by those that are in all of [Masc, Fem] x [Sing, Plur]
    types = data.types_lemmas
    lemmas_all_inflections = set(types['MascSing']).intersection(
        set(types['FemSing'])
    ).intersection(
        set(types['FemPlur'])
    ).intersection(
        set(types['MascPlur'])
    )
    filtered_animate_lemmas = animate_lemmas.intersection(lemmas_all_inflections)
    print(filtered_animate_lemmas)

    # filter out nouns ending in e masculine

    # Get nouns whose inflections exist for all Gender x Number and also whose inflections are "transparent"
    # nonex: il camariere le camariere  