from collections import defaultdict
from tqdm.auto import tqdm

from jokes import tokenize, load_jokes, lemmatize


if __name__ == '__main__':
    print('creating inverse index for jokes...')
    texts = load_jokes()
    inverse_index = defaultdict(list)
    for i, text in enumerate(tqdm(texts)):
        for token in tokenize(text):
            inverse_index[token.text.lower()].append(i)
    print(len(inverse_index))

    lemma_index = defaultdict(list)
    for k, v in tqdm(inverse_index.items()):
        lemma_index[lemmatize(k)].extend(v)
    print(len(lemma_index))

    with open('index.txt', 'w', encoding='utf-8') as f:
        for k, v in tqdm(lemma_index.items()):
            if len(v) < 2 or len(v) > 100_000:
                continue
            f.write('\t'.join([k] + [str(vv) for vv in v]))
            f.write('\n')
