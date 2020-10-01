import json
import keras2onnx
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras import layers

from jokes import tokenize
from nlu import text2matrix, load_ft


def parse_markup(text):
    state = 'O'
    x = []
    y = []
    tagged = []
    tag = None
    for tok in tokenize(text):
        t = tok.text
        if state == 'O':
            if t == "'":
                state = 'I'
                tagged = []
            else:
                x.append(t)
                y.append('O')
        elif state == 'I':
            if t == "'":
                state = 'X'
            else:
                tagged.append(t)
        elif state == 'X':
            if t == '(':
                pass
            elif t == ')':
                x.extend(tagged)
                y.extend([tag] * len(tagged))
                state = 'O'
            else:
                tag = t
    return x, y


def encode_tags(old_tags, tagset, length=None, zero_tag='O'):
    result = [tagset[t] for t in old_tags]
    if length:
        if len(result) > length:
            result = result[:length]
        elif len(result) < length:
            result.extend([tagset[zero_tag]] * (length - len(result)))
    return result


def load_training_data(intents_path='intents'):

    clf_texts = []
    clf_labels = []
    tagger_data = {}

    for fn in os.listdir(intents_path):
        label = fn.split('.')[0]
        label_texts = []
        label_tags = []
        with open(os.path.join(intents_path, fn), encoding='utf-8') as f:
            for l in f.readlines():
                text = l.strip()
                if not text:
                    continue
                toks, tags = parse_markup(text)
                label_texts.append(' '.join(toks))
                label_tags.append(tags)
                clf_labels.append(label)
        clf_texts.extend(label_texts)
        tagset = {tag for line in label_tags for tag in line}
        if len(tagset) > 1:
            tagger_data[label] = {'texts': label_texts, 'tags': label_tags}

    return clf_texts, clf_labels, tagger_data


def classifier_model(classes):
    model = keras.Sequential([
        layers.Input(shape=(None, 300), dtype="float32"),
        layers.Dropout(0.5),
        layers.Conv1D(128, 3, activation='elu', padding='same'),
        layers.GlobalMaxPooling1D(),
        layers.Dense(len(classes), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def tagger_model(tagset):
    model = keras.Sequential([
        layers.Input(shape=(None, 300), dtype="float32"),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dense(len(tagset), activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_classifier(clf_texts, clf_labels, w2v):
    classes = {label: i for i, label in enumerate(sorted(set(clf_labels)))}
    with open('resources/classifier.json', 'w') as f:
        json.dump(classes, f, indent=2)
    y = np.array([classes[label] for label in clf_labels])
    max_len = max(len(t.split()) for t in clf_texts)
    X = np.stack([text2matrix(t, w2v, length=max_len) for t in clf_texts])
    model = classifier_model(classes)
    print('training classifier...')
    model.summary()
    model.fit(X, y, shuffle=True, epochs=50, batch_size=8)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    with open("resources/classifier.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


def train_tagger(label, label_texts, label_tags, w2v):
    max_len = max(len(t.split()) for t in label_texts)
    X = np.stack([text2matrix(t, w2v, length=max_len) for t in label_texts])
    tagset = {tag: i for i, tag in enumerate(sorted({t for ts in label_tags for t in ts}))}
    with open(f'resources/taggers/{label}.json', 'w') as f:
        json.dump(tagset, f, indent=2)
    y = np.stack([encode_tags(t, tagset=tagset, length=max_len) for t in label_tags])
    model = tagger_model(tagset)
    print('training tagger for {}...'.format(label))
    model.summary()
    model.fit(X, y[..., np.newaxis], shuffle=True, epochs=50, batch_size=8)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    with open(f"resources/taggers/{label}.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())


if __name__ == '__main__':
    clf_texts, clf_labels, tagger_data = load_training_data()
    ft = load_ft()
    w2v = lambda word: ft[word]
    train_classifier(clf_texts, clf_labels, w2v=w2v)
    for key, item in tagger_data.items():
        train_tagger(label=key, label_texts=item['texts'], label_tags=item['tags'], w2v=w2v)
