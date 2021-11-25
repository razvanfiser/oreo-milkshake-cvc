import os
from random import randint
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import CallbackContext
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import requests
import constants

test_path = os.path.join(os.getcwd(), "..\\author_classification_ds\\authors_test\\authors_test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
authors_test = ImageFolder(test_path, transform=transform)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Authors_Test(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = {"pic": self.X[index][0], "label": self.X[index][1]}
        return sample


test_data = Authors_Test(authors_test)
test_size = len(authors_test)
test_data = torch.utils.data.Subset(test_data, np.arange(test_size))


def get_model(n_classes):
    model = torchvision.models.resnet101(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=n_classes)
    return model


model = get_model(len(authors_test.classes)).to(device=device)

model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.001)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

loaded_model = get_model(n_classes=5).to(device=device)
loaded_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "..\\models\\authors_model_resnet101_augment.pth"), map_location=torch.device('cpu')))

pred = np.zeros(len(test_data))
true = np.zeros(len(test_data))
loaded_model.eval()
comp_score = 0
user_score = 0

def get_author(ind, chat_id) :
    global comp_score
    with torch.no_grad():
        out = loaded_model(test_data[ind]["pic"].unsqueeze(0).to(device=device))
        pred[ind] = torch.argmax(out)
        true[ind] = test_data[ind]["label"]
        if pred[ind] == true[ind]:
            comp_score += 1

        parameters = {
            "chat_id": chat_id,
            "text": "Correct: " + authors_test.classes[int(true[ind])] +
                    '\n' + "Computer's guess: " + authors_test.classes[int(pred[ind])]
        }
    resp = requests.get(constants.base_url_txt, data=parameters)


def button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    query.answer()

    query.edit_message_text(text=f"Selected option: {query.data}")
    print(query.data)


def authors_game(update: Update, context: CallbackContext):

    id = update.effective_chat.id

    update.message.reply_text('Game started! mode: authors' + '\n' + "Please guess the authors of the following images:")

    for i in range(10):

        parameters = {
            "chat_id": id,
            "text": 'Level ' + str(i + 1)
        }
        resp = requests.get(constants.base_url_txt, data=parameters)
        j = randint(1, test_size - 1)

        if j <= 69:
            img_path = "..\\author_classification_ds\\authors_test\\authors_test\\fernand_leger\\0 (" + str(j) + ").jpg"
        elif j<=145:
            img_path = "..\\author_classification_ds\\authors_test\\authors_test\\ivan_aivazovsky\\1 (" + str(j-69) + ").jpg"
        elif j<=220:
            img_path = "..\\author_classification_ds\\authors_test\\authors_test\\rembrandt\\2 (" + str(j-145) + ").jpg"
        elif j<=291:
            img_path = "..\\author_classification_ds\\authors_test\\authors_test\\salvador_dali\\3 (" + str(j-220) + ").jpg"
        else:
            img_path = "..\\author_classification_ds\\authors_test\\authors_test\\vincent_van_gogh\\4 (" + str(j-291) + ").jpg"

        my_file = open(img_path, "rb")
        parameters = {
            "chat_id": id,
        }
        files = {
            "photo": my_file
        }
        resp = requests.get(constants.base_url_img, data=parameters, files=files)

        keyboard = [
            [
                InlineKeyboardButton("Fernand Leger", callback_data="Fernand Leger"),
                InlineKeyboardButton("Ivan Aivazovsky", callback_data="Ivan Aivazovsky"),
                InlineKeyboardButton("Rembrandt", callback_data="Rembrandt")
            ],
            [
                InlineKeyboardButton("Salvador Dali", callback_data='Salvador Dali'),
                InlineKeyboardButton("Vincent Van Gogh", callback_data='Vincent Van Gogh'),
            ],
        ]

        reply_markup = InlineKeyboardMarkup(keyboard)

        update.message.reply_text('Please choose:', reply_markup=reply_markup)

        get_author(j, id)
    print(comp_score)



