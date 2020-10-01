import random
import numpy as np

from collections import Counter

from jokes import load_jokes, load_index, find_jokes
from nlu import NLU, normalize_vec

JOKES = load_jokes()
INVERSE_INDEX = load_index()

GLOBAL_COEF = np.zeros(300, dtype=np.float32)  # todo: save it to a database
GLOBAL_LEARNING_RATE = 0.001
LOCAL_LEARNING_RATE = 0.01


def get_coef(state):
    return state.get('coef', np.zeros(300, dtype=np.float32))


def rerank_jokes(joke_indices, state, nlu_engine: NLU, max_count=30, noise=0.5):
    vector = normalize_vec(GLOBAL_COEF + get_coef(state))
    new_scores = Counter()
    for key, score in joke_indices.items():
        new_scores[key] = score + nlu_engine.score_text(JOKES[key], vector)
        if noise:
            new_scores[key] += random.gauss(mu=0, sigma=noise)
    result = []

    for joke_id, score in new_scores.most_common(max_count):
        result.append(joke_id)
    return result


def respond(text, state, nlu_engine: NLU):
    if text:
        intent, form = nlu_engine.process_text(text)
    else:
        intent = 'hello'
        form = {}

    def random_joke():
        joke_id = random.randint(0, len(JOKES) - 1)
        state['joke_id'] = joke_id
        state['jokes'] = None
        return JOKES[joke_id]

    def choose_joke(item_id):
        joke_id = state['jokes'][item_id]
        state['joke_id'] = joke_id
        state['item_id'] = item_id
        return JOKES[joke_id]

    if intent == 'hello':
        response = 'Привет! Я бот @a_nek_bot, рассказываю анекдоты. ' \
                   '\nСкажите, например, "расскажи анекдот про щуку".' \
                   '\nСкажите "лайк" или "дизлайк", чтобы выразить отношение к шутке.' \
                   '\nСкажите "ещё", чтобы получить следующий анекдот.'
    elif intent == 'find_joke':
        if form.get('topic'):
            jokes = rerank_jokes(find_jokes(form['topic'], index=INVERSE_INDEX), state, nlu_engine)
            if not jokes:
                response = 'Простите, ничего не нашла. ' \
                           f'Зато вот какая шутка есть: \n{random_joke()}'
            else:
                state['jokes'] = jokes
                response = choose_joke(0)
        else:
            response = random_joke()
    elif intent == 'next':
        if 'item_id' in state and state.get('jokes') and state['item_id'] + 1 < len(state['jokes']):
            response = choose_joke(state['item_id'] + 1)
        else:
            response = 'Я забыла, что там дальше. ' \
                f'Зато вот какая шутка есть: \n{random_joke()}'
    elif intent in {'like', 'dislike'}:
        if state.get('joke_id'):
            vec = nlu_engine.text2vec(JOKES[state['joke_id']])
            if intent == 'like':
                sign = +1
                response = 'Запомню, что вам такое нравится!'
            else:
                response = 'Запомню, что вам такое не нравится!'
                sign = -1
            nlu_engine.update_coef(vec, GLOBAL_COEF, lr=GLOBAL_LEARNING_RATE, sign=sign)
            state['coef'] = nlu_engine.update_coef(vec, get_coef(state), lr=LOCAL_LEARNING_RATE, sign=sign)
        else:
            response = 'Я забыла, о чём мы говорили. ' \
                f'Зато вот какая шутка есть: \n{random_joke()}'
    else:
        response = 'Скажите мне "расскажи анекдот", и я смешно пошучу'
    return response, state
