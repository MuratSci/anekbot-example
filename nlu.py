import compress_fasttext
import json
import numpy as np
import onnxruntime as rt
import os

from collections import defaultdict
from functools import lru_cache
from jokes import tokenize, lemmatize


def load_ft(resources_path='resources'):
    return compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        os.path.join(resources_path, 'ft_freqprune_100K_20K_pq_100.bin')
    )


def normalize_vec(vec):
    norm = sum(vec**2)
    if norm > 0:
        return vec / norm ** 0.5
    return vec


def text2matrix(text, w2v, length=None, dim=300):
    vecs = []
    for token in tokenize(text):
        lemma = lemmatize(token.text)
        vecs.append(w2v(lemma))
    result = np.stack(vecs).astype(np.float32)
    if not length:
        return result
    elif len(vecs) < length:
        return np.concatenate([result, np.zeros([length-len(vecs), dim], dtype=np.float32)])
    elif len(vecs) > length:
        return result[:length]
    else:
        return result


class NLU:
    def __init__(self, resources_path='resources'):
        self.resources_path = resources_path
        self.ft = load_ft(resources_path)
        self.classifier = rt.InferenceSession(os.path.join(self.resources_path, 'classifier.onnx'))
        with open(os.path.join(self.resources_path, 'classifier.json')) as f:
            self.classes = {label_id: label for label, label_id in json.load(f).items()}
        self.taggers = {}
        self.tagsets = {}
        for fn in os.listdir(os.path.join(resources_path, 'taggers')):
            label, ext = fn.split('.')
            if ext == 'json':
                with open(os.path.join(self.resources_path, 'taggers', fn)) as f:
                    self.tagsets[label] = {label_id: label for label, label_id in json.load(f).items()}
            elif ext == 'onnx':
                self.taggers[label] = rt.InferenceSession(os.path.join(self.resources_path, 'taggers', fn))

    @lru_cache(10000)
    def w2v(self, word):
        return self.ft[word]

    def classify(self, vectors):
        input_name = self.classifier.get_inputs()[0].name
        pred_onx = self.classifier.run(None, {input_name: vectors})[0]
        return [self.classes[label] for label in pred_onx.argmax(axis=1)]

    def apply_tagger(self, vectors, label, tokens, empty_tag='O'):
        if label not in self.tagsets or label not in self.taggers:
            return {}
        tagger = self.taggers[label]
        input_name = tagger.get_inputs()[0].name
        pred_onx = tagger.run(None, {input_name: vectors})[0][0]  # only the first line in batch
        tags = [self.tagsets[label][tag_id] for tag_id in pred_onx.argmax(axis=1)]
        form = defaultdict(list)
        for token, tag in zip(tokens, tags):
            if tag != empty_tag:
                form[tag].append(token)
        return {k: ' '.join(v) for k, v in form.items()}

    def process_text(self, text):
        inputs = text2matrix(text, self.w2v)[np.newaxis]
        tokens = [t.text for t in tokenize(text)]
        label = self.classify(inputs)[0]
        form = self.apply_tagger(inputs, label, tokens)
        return label, form

    def text2vec(self, text):
        return normalize_vec(text2matrix(text, self.w2v).mean(axis=0))

    def score_text(self, joke, coef):
        return np.dot(self.text2vec(joke), coef)

    def update_coef(self, update, coef, lr, sign):
        coef += (update * sign - coef) * lr
        return coef

