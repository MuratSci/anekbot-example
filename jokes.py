from collections import Counter, defaultdict
from pymorphy2 import MorphAnalyzer
from razdel import tokenize

morph = MorphAnalyzer()


def lemmatize(word):
    parses = morph.parse(word)
    if parses:
        return parses[0].normal_form or word
    return word


def load_jokes(filename='jokes.txt'):
    with open(filename, 'r', encoding='utf-8') as f:
        return [l.replace('\\n', '\n').strip() for l in f.readlines()]


def load_index(filename='index.txt'):
    index = defaultdict(list)
    with open(filename, 'r', encoding='utf-8') as f:
        for l in f.readlines():
            parts = l.strip().split('\t')
            if len(parts) > 1:
                index[parts[0]] = [int(p) for p in parts[1:]]
    return index


def find_jokes(query, index):
    result = Counter()
    for token in tokenize(query):
        lemma = lemmatize(token.text)
        docs = index.get(lemma, [])  # find documents with this word
        for doc in docs:
            result[doc] += 1 / len(docs)
    return result
