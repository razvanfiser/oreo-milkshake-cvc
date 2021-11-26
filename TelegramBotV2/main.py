from random import randint
import telebot
import torch
from telebot import types
import authors_model
import styles_model

client = telebot.TeleBot("2113002847:AAExiWKJATrnpbx9koHwZp0cZOehGJibccA")
base_url_img = "https://api.telegram.org/bot2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc/sendPhoto"
base_url_txt = "https://api.telegram.org/bot2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc/sendMessage"
USER_SCORE = 0
COMPUTER_SCORE = 0
i=0


@client.message_handler(commands=["start"])
def application(message):
    rmk = types.ReplyKeyboardMarkup()
    rmk.add(types.KeyboardButton("Authors"), types.KeyboardButton("Styles"))
    msg = client.send_message(message.chat.id, "Hi! Select Game Type!", reply_markup=rmk)
    client.register_next_step_handler(msg, user_answer)


def user_answer(message):
    global USER_SCORE
    global COMPUTER_SCORE
    global i
    if message.text == "Authors":
        msg = client.send_message(message.chat.id, "Game Started! Mode: Authors."+'\n'+"Press any key to start playing")
        i = 0
        USER_SCORE = 0
        COMPUTER_SCORE = 0
        client.register_next_step_handler(msg, authors_game, i)
    elif message.text == "Styles":
        msg = client.send_message(message.chat.id, "Game Started! Mode: Styles."+'\n'+"Press any key to start playing")
        i = 0
        USER_SCORE = 0
        COMPUTER_SCORE = 0
        client.register_next_step_handler(msg, styles_game, i)


def authors_game(message, level):
    k = randint(1, 370)
    if k <= 69:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\fernand_leger\\0 ("+str(k)+").jpg"
    elif k <= 145:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\ivan_aivazovsky\\1 ("+str(k-69)+").jpg"
    elif k <= 220:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\rembrandt\\2 ("+str(k-145)+").jpg"
    elif k <= 291:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\salvador_dali\\3 ("+str(k-220)+").jpg"
    else:
        img_path = "..\\author_classification_ds\\authors_test\\authors_test\\vincent_van_gogh\\4 ("+str(k-291)+").jpg"
    my_file = open(img_path, "rb")
    client.send_photo(message.chat.id, my_file)

    rmk = types.ReplyKeyboardMarkup()
    rmk.add(types.KeyboardButton("Fernand Leger"), types.KeyboardButton("Ivan Aivazovsky"),
            types.KeyboardButton("Rembrandt"), types.KeyboardButton("Salvador Dali"),
            types.KeyboardButton("Vincent Van Gogh"))
    msg = client.send_message(message.chat.id, "Guess the author!", reply_markup=rmk)
    client.register_next_step_handler(msg, computer_guess_authors, msg.chat.id, k, level)


def styles_game(message, level):
    k = randint(1, 1186)
    if k <= 240:
        img_path = "..\\styles_dataset\\styles_test\\styles_test\\abstract_expressionism\\0 ("+str(k)+").jpg"
    elif k <= 480:
        img_path = "..\\styles_dataset\\styles_test\\styles_test\\baroque\\1 ("+str(k-240)+").jpg"
    elif k <= 720:
        img_path = "..\\styles_dataset\\styles_test\\styles_test\\cubism\\2 ("+str(k-480)+").jpg"
    elif k <= 960:
        img_path = "..\\styles_dataset\\styles_test\\styles_test\\romanticism\\3 ("+str(k-720)+").jpg"
    else:
        img_path = "..\\styles_dataset\\styles_test\\styles_test\\ukiyo-e\\4 ("+str(k-960)+").jpg"
    my_file = open(img_path, "rb")
    client.send_photo(message.chat.id, my_file)

    rmk = types.ReplyKeyboardMarkup()
    rmk.add(types.KeyboardButton("Abstract Expressionism"), types.KeyboardButton("Baroque"),
            types.KeyboardButton("Cubism"), types.KeyboardButton("Romanticism"),
            types.KeyboardButton("Ukiyo-e"))
    msg = client.send_message(message.chat.id, "Guess the style!", reply_markup=rmk)
    client.register_next_step_handler(msg, computer_guess_styles, msg.chat.id, k, level)


def computer_guess_authors(message, chat_id, ind, level):
    global COMPUTER_SCORE
    global USER_SCORE

    user_option = 111
    if message.text == "Fernand Leger":
        user_option = 0
    elif message.text == "Ivan Aivazovsky":
        user_option = 1
    elif message.text == "Rembrandt":
        user_option = 2
    elif message.text == "Salvador Dali":
        user_option = 3
    elif message.text == "Vincent Van Gogh":
        user_option = 4

    if user_option != 111:
        with torch.no_grad():
            out = authors_model.loaded_model(authors_model.test_data[ind]["pic"].unsqueeze(0).to(device=authors_model.device))
            authors_model.pred[ind] = torch.argmax(out)
            authors_model.true[ind] = authors_model.test_data[ind]["label"]
            if authors_model.pred[ind] == authors_model.true[ind]:
                COMPUTER_SCORE += 1
            if user_option == authors_model.true[ind]:
                USER_SCORE += 1

        rmk = types.ReplyKeyboardMarkup()
        rmk.add(types.KeyboardButton("Continue"))
        msg = client.send_message(
            message.chat.id,
            "Level: " + str(level+1) +
            '\n' + "Correct: " + translate(authors_model.authors_test.classes[int(authors_model.true[ind])]) +
            '\n' + "Computer's guess: " + translate(authors_model.authors_test.classes[int(authors_model.pred[ind])]) +
            '\n' + "Score: " + str(USER_SCORE) + ":" + str(COMPUTER_SCORE),
            reply_markup=rmk)
        level += 1
        if level < 10:
            client.register_next_step_handler(message, authors_game, level)
        else:
            if(USER_SCORE<COMPUTER_SCORE):
                client.send_message(message.chat.id, "Game Over!"+'\n'+"Computer won!"+'\n'+"Better luck next time!")
            else:
                client.send_message(message.chat.id, "Game Over!"+'\n'+"You won!!!"+'\n'+"Want to do it again?")
            client.register_next_step_handler(message, application)


def computer_guess_styles(message, chat_id, ind, level):
    global COMPUTER_SCORE
    global USER_SCORE

    user_option = 111
    if message.text == "Abstract Expressionism":
        user_option = 0
    elif message.text == "Baroque":
        user_option = 1
    elif message.text == "Cubism":
        user_option = 2
    elif message.text == "Romanticism":
        user_option = 3
    elif message.text == "Ukiyo-e":
        user_option = 4

    if user_option != 111:
        with torch.no_grad():
            out = styles_model.loaded_model(styles_model.test_data[ind]["pic"].unsqueeze(0).to(device=styles_model.device))
            styles_model.pred[ind] = torch.argmax(out)
            styles_model.true[ind] = styles_model.test_data[ind]["label"]
            if styles_model.pred[ind] == styles_model.true[ind]:
                COMPUTER_SCORE += 1
            if user_option == styles_model.true[ind]:
                USER_SCORE += 1

        rmk = types.ReplyKeyboardMarkup()
        rmk.add(types.KeyboardButton("Continue"))
        msg = client.send_message(
            message.chat.id,
            "Level: " + str(level+1) +
            '\n' + "Correct: " + translate(styles_model.styles_test.classes[int(styles_model.true[ind])]) +
            '\n' + "Computer's guess: " + translate(styles_model.styles_test.classes[int(styles_model.pred[ind])]) +
            '\n' + "Score: " + str(USER_SCORE) + ":" + str(COMPUTER_SCORE),
            reply_markup=rmk)
        level += 1
        if level < 10:
            client.register_next_step_handler(message, styles_game, level)
        else:
            if USER_SCORE < COMPUTER_SCORE:
                client.send_message(message.chat.id, "Game Over!"+'\n'+"Computer won"+'\n'+"Better luck next time!")
            else:
                client.send_message(message.chat.id, "Game Over!"+'\n'+"You won!!!"+'\n'+"Want to do it again?")
            client.register_next_step_handler(message, application)


def translate(word):
    if word == "fernand_leger":
        return "Fernand Leger"
    if word == "ivan_aivazovsky":
        return "Ivan Aivazovsky"
    if word == "rembrandt":
        return "Rembrandt"
    if word == "salvador_dali":
        return "Salvador Dali"
    if word == "vincent_van_gogh":
        return "Vincent Van Gogh"
    if word == "abstract_expressionism":
        return "Abstract Expressionism"
    if word == "baroque":
        return "Baroque"
    if word == "cubism":
        return "Cubism"
    if word == "romanticism":
        return "Romanticism"
    if word == "ukiyo-e":
        return "Ukiyo-e"

client.polling()
