import argparse
import os
import telebot
from collections import defaultdict
from flask import Flask, request

from logic import respond
from nlu import NLU


TOKEN = os.getenv('TOKEN')
BASE_URL = os.getenv('BASE_URL')
WEBHOOK_URL = '/telebot_webhook/{}'.format(TOKEN)
bot = telebot.TeleBot(TOKEN)
nlu_engine = NLU()
states = defaultdict(dict)  # todo: keep them in a database if possible

server = Flask(__name__)


# basic responder for Telegram bot
@bot.message_handler()
def respond_in_telegram(message):
    uid = message.chat.id
    state = states.get(uid, {})
    state['uid'] = uid
    response, state = respond(message.text, state, nlu_engine=nlu_engine)
    states[uid] = state
    bot.send_message(chat_id=message.chat.id, text=response)


@server.route(WEBHOOK_URL, methods=['POST'])
def get_message():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cli', action='store_true', default=False)
    parser.add_argument('--poll', action='store_true', default=False)
    args = parser.parse_args()

    if args.cli:
        print('Чтобы выйти из режима разговора, нажмите Ctrl+C или Cmd+C')
        input_sentence = '/start'
        state = {'uid': 'cli'}
        while True:
            response, state = respond(input_sentence, state, nlu_engine=nlu_engine)
            print(response)
            input_sentence = input('> ')
    elif args.poll:
        bot.polling()
    else:
        bot.remove_webhook()
        bot.set_webhook(url=BASE_URL + WEBHOOK_URL)
        server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
