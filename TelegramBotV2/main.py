from random import randint
import requests
import telebot
from telebot import types
import authors_model

client = telebot.TeleBot("2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc")
base_url_img = "https://api.telegram.org/bot2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc/sendPhoto"
base_url_txt = "https://api.telegram.org/bot2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc/sendMessage"


@client.message_handler(commands=["start"])
def application(message):
    rmk = types.ReplyKeyboardMarkup()
    rmk.add(types.KeyboardButton("Authors"), types.KeyboardButton("Styles"))
    msg = client.send_message(message.chat.id, "Hi! Select Game Type!", reply_markup=rmk)
    client.register_next_step_handler(msg, user_answer)


def user_answer(message):
    if message.text == "Authors":
        k = randint(1, 371)
        msg = client.send_message(message.chat.id, "Game Started! Mode: Authors ")
        client.register_next_step_handler(message, send_image_authors(msg.chat.id, k))

        rmk = types.ReplyKeyboardMarkup()
        rmk.add(types.KeyboardButton("Fernand Leger"), types.KeyboardButton("Ivan Aivazovsky"),
                types.KeyboardButton("Rembrandt"), types.KeyboardButton("Salvador Dali"),
                types.KeyboardButton("Vincent Van Gogh"))
        msg = client.send_message(message.chat.id, "Guess the author!", reply_markup=rmk)
        client.register_next_step_handler(msg, authors_model.computer_guess_authors, message.chat.id, k)

    elif message.text == "Styles":
        msg = client.send_message(message.chat.id, "Game Started! Mode: Authors")
        client.register_next_step_handler(msg, send_image_styles)


def user_answer_authors(message, j):
    if message.text == "Fernand Leger" or message.text == "Ivan Aivazovsky" or message.text == "Rembrandt"\
            or message.text == "Salvador Dali" or message.text == "Vincent Van Gogh":
        print(j)
        msg = client.send_message(message.chat.id, "results:")
        print(j)


def send_image_authors(chat_id, j):
    if j <= 69:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\fernand_leger\\0 ("+str(j)+").jpg"
    elif j <= 145:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\ivan_aivazovsky\\1 ("+str(j-69)+").jpg"
    elif j <= 220:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\rembrandt\\2 ("+str(j-145)+").jpg"
    elif j <= 291:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\salvador_dali\\3 ("+str(j-220)+").jpg"
    else:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\vincent_van_gogh\\4 ("+str(j-291)+").jpg"

    my_file = open(img_path, "rb")
    parameters = {
        "chat_id": chat_id,
    }
    files = {
        "photo": my_file
    }
    resp = requests.get(base_url_img, data=parameters, files=files)


def send_image_styles(message):
    return


client.polling()
