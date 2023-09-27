# FILE FORMATS
all_tokens_df_format = '/home/echeng/morph_systems_entropy/wiki/{}/all_tokens_{}_wikipedia.csv' 
all_types_df_format = '/home/echeng/morph_systems_entropy/wiki/{}/all_types_{}_wikipedia.csv' 
anim_tokens_df_format = '/home/echeng/morph_systems_entropy/wiki/{}/anim_tokens_{}_wikipedia.csv'
anim_types_df_format = '/home/echeng/morph_systems_entropy/wiki/{}/anim_types_{}_wikipedia.csv'

wordnet_mapping = {
    'de': 'deu', # [x]
    'ca': 'cat', # [x]
    'en': 'eng', # [x]
    'ar': 'arb', # [ ]
    'fr': 'fra', # [x]
    'es': 'spa', # [x]
    'it': 'ita', # [x]
    'sv': 'swe', # [x]
    'he': 'heb', # [ ]
    'nl': 'nld', # [x]
    'pl': 'pol', # [x]
    'ru': 'eng', # [ ] There is no Russian wordnet.
    'sl': 'slv', # [x]
}

gender_mapping = {}

for l in ['de', 'ru', 'sl', 'pl']:
    gender_mapping[l] = ['Masc', 'Fem', 'Neut']
for l in ['ar', 'he', 'ca', 'fr', 'it', 'es']:
    gender_mapping[l] = ['Masc', 'Fem']
for l in ['sv']:
    gender_mapping[l] = ['Com', 'Neut']
for l in ['nl']:
    gender_mapping[l] = ['Com', 'Neut', ''] # plural has no gender
for l in ['en']:
    gender_mapping[l] = ['']
    
number_mapping = {}

for l in wordnet_mapping:
    if l in ['he', 'ar', 'sl']: number_mapping[l] = ['Sing', 'Plur', 'Dual']
    else:
        number_mapping[l] = ['Sing', 'Plur']