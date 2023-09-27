import re
import json
import stanza
from stanza.utils.conll import CoNLL

import typing


def tag_data(lang: str, filepath: str):
    nlp = stanza.Pipeline(lang=lang, processors='tokenize,mwt,pos,lemma')

    # Load in file
    f = open(filepath)
    docs_dict = json.load(f)['rows']
    f.close()

    # Convert to "doc" objects.
    docs = [doc['row']['text'] for doc in docs_dict]
    print("Number of docs: ", len(docs))

    # Clean strings.
    docs = [clean_doc(doc, lang) for doc in docs]
    docs = [doc for doc in docs if doc is not None] # if the doc is invalid, remove it
    print(docs)

    # Convert to ""

def clean_doc(doc: str, lang: str):
    ## General (remove document):
    # structured data like |--menu--|.
    if '\u007c' in doc or '~' in doc or '\u007c' in doc:
        return
    # emails, urls TODO: with regex
    elif '@' in doc or 'http' in doc or 'www.' in doc:
        return

    # General: strip whitespace characters, things inside brackets
    doc = re.sub('\n', '', doc)
    doc = doc.replace('\u2009', "").replace('\ufeff', '')
    doc = doc.replace('\u005c', '').replace('\u0027', '')
    doc = doc.replace("\\", "")
    doc = removeunicode(doc)

    # French: \' -> '     &apos; -> '
    if lang == 'fr':
        doc = doc.replace("\\\\'", "\'").replace("\\\'", "\'").replace("\\'", "\'").replace(r"\'", "\'")
        doc = re.sub('\&apos;', '\'', doc)
    return doc

def removeunicode(text):
    return re.sub(r'[^ A-Za-z0-9À-ÖØ-öø-ÿ/.\']+', '', text)
    #bytes = text.encode('ascii', 'ignore')
    #text = bytes.decode()
    #return text

if __name__ == '__main__':
    #tag_data('fr', '/home/echeng/morph_systems_entropy/oscar/french.json')
    tag_data('es', '/home/echeng/morph_systems_entropy/oscar/spanish.json')
