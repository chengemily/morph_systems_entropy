import pandas as pd
import argparse
from collections import deque
import os

def subsample_nouns_and_save(lang, subset, genders, numbers):

    filepath = f'/home/echeng/morph_systems_entropy/wiki/{lang}/{subset}_tokens_{lang}_wikipedia.csv'

    tokens_df = pd.read_csv(filepath)
    print(tokens_df.head(10))

    # cut tokens_df into forms existing in both sing & plur
    number_dfs = {
        number: tokens_df[tokens_df['number']==number] for number in numbers
    }
    dfs = []
    for number in number_dfs:
        number_df = number_dfs[number]

        # sample 125 masc and 125 fem nouns
        gender_dfs = {
            gender: number_df[number_df['gender']==gender] for gender in genders
        }
        for gender in gender_dfs:
            print(f'{number} {gender}: {len(gender_dfs[gender])}')

        # get their "other" number counterpart
        other_number = 'Sing' if number == 'Plur' else 'Plur'
        other_number_df = sing_plur_dfs[other_number]
        masc_other_df = other_number_df[other_number_df['gender']=='Masc']
        fem_other_df = other_number_df[other_number_df['gender']=='Fem']

        # cut dfs to those existing in both plur/sing
        masc_lemmas_both_plur_sing = set(masc_df['lemma']).intersection(set(masc_other_df['lemma']))
        fem_lemmas_both_plur_sing = set(fem_df['lemma']).intersection(set(fem_other_df['lemma']))

        masc_df = masc_df[masc_df['lemma'].isin(masc_lemmas_both_plur_sing)]
        fem_df = fem_df[fem_df['lemma'].isin(fem_lemmas_both_plur_sing)]

        # Sample df
        sampled_masc_df = masc_df.sample(min(125, len(masc_df)))
        sampled_fem_df = fem_df.sample(min(125, len(fem_df)))
        print(f'Sampled masc {len(sampled_masc_df)}, fem {len(sampled_fem_df)}')
        # Match to other side
        sampled_masc_other_df = masc_other_df[masc_other_df['lemma'].isin(sampled_masc_df['lemma'])]
        sampled_fem_other_df = fem_other_df[fem_other_df['lemma'].isin(sampled_fem_df['lemma'])]
        print(f'Matched {other_number} masc: {len(sampled_masc_other_df)}. fem: {len(sampled_fem_other_df)}')

        dfs.append(sampled_masc_other_df)
        dfs.append(sampled_masc_df)
        dfs.append(sampled_fem_df)
        dfs.append(sampled_fem_other_df)

    subsampled_df = pd.concat(dfs)
    print('Number of nouns before dedup: ', len(subsampled_df))

    subsampled_df = subsampled_df.drop_duplicates()
    print('After dedup: ', len(subsampled_df))
    savepath = f'/home/echeng/morph_systems_entropy/wiki/{lang}/{subset}_tokens_contextentropy_subsample_{lang}_wikipedia.csv'
    subsampled_df.to_csv(savepath)

def collect_contexts_for_nouns(lang, subset, fpath_number, context_length=20):
    """
    Collects all contexts for nouns in filepath.
    :param context_length (int): the size of the context around the noun (must be even).
    """
    filepath = '/home/echeng/morph_systems_entropy/wiki/{}/{}_{}_wikipedia.conllu'.format(lang, lang, fpath_number)
    if not os.path.exists(filepath): return

    noun_list = f'/home/echeng/morph_systems_entropy/wiki/{lang}/{subset}_tokens_contextentropy_subsample_{lang}_wikipedia.csv'
    noun_list = list(pd.read_csv(noun_list)['text'])
    savepaths = {noun: f'/home/echeng/morph_systems_entropy/wiki/contexts/{lang}/{subset}/{noun}10_10.txt' for noun in noun_list}

    assert context_length % 2 == 0
    max_len = context_length + 1

    buf = deque(maxlen=max_len)

    # stream in
    with open(filepath, 'r') as file:
        line = True

        while line:
            line = file.readline()
            if '#' in line or len(line) < 2: continue

            # Read new token
            split_line = line.split('\t')
            text = split_line[1].lower()
            pos = split_line[3]

            if pos == '_': continue

            # Add to deque
            buf.append(text + '_' + pos) # abacus_NOUN

            # Check if the center noun is in the subset of nouns of interest
            if len(buf) < max_len: continue
            center_word = buf[context_length // 2].split('_')[0]

            if center_word in savepaths:
                # Append buffer to relevant file
                with open(savepaths[center_word], 'a') as w:
                    w.write(' '.join(buf) + '\n')




if __name__=='__main__':
    parser = argparse.ArgumentParser(
                            prog='ProgramName',
                            description='What the program does',
                            epilog='Text at the bottom of help'
    )
    parser.add_argument('--lang', type=str)
    parser.add_argument('--subset', type=str, choices=['anim', 'all'])
    parser.add_argument('--file_number', type=str, default='')
    args = parser.parse_args()

    # Subsample nouns
    # subsample_nouns_and_save(args.lang, args.subset)

    # Collect contexts for nouns
    collect_contexts_for_nouns(args.lang, args.subset, args.file_number)

