import random

from jokes import load_jokes, load_index, find_jokes
from nlu import NLU

JOKES = load_jokes()
INVERSE_INDEX = load_index()


def rerank_jokes(joke_indices, state, max_count=30):
    result = []
    for joke_id, score in joke_indices.most_common(max_count):
        result.append(joke_id)
    return result


def respond(text, state, nlu_engine: NLU):
    if text:
        intent, form = nlu_engine.process_text(text)
    else:
        intent = 'hello'
        form = {}

    def random_joke():
        joke_id = random.randint(len(JOKES))
        state['joke_id'] = joke_id
        state['jokes'] = None
        return JOKES[joke_id]

    def choose_joke(item_id):
        joke_id = state['jokes'][item_id]
        state['joke_id'] = joke_id
        state['item_id'] = item_id
        return JOKES[joke_id]

    if intent == 'hello':
        response = 'Привет! Я бот-анебот, рассказываю анекдоты. ' \
                   '\nСкажите, например, "расскажи анекдот про щуку".' \
                   '\nСкажите "лайк" или "дизлайк", чтобы выразить отношение к шутке.' \
                   '\nСкажите "ещё", чтобы получить следующий анекдот.'
    elif intent == 'find_joke':
        if form.get('topic'):
            jokes = rerank_jokes(find_jokes(form['topic'], index=INVERSE_INDEX), state)
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
    elif intent == 'like':
        # todo: save the like
        response = 'Запомню, что вам такое нравится!'
    elif intent == 'dislike':
        # todo: save the dislike
        response = 'Запомню, что вам такое не нравится!'
    else:
        response = 'Скажите мне "расскажи анекдот", и я смешно пошучу'
    return response, state
