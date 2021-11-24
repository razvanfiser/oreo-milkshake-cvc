import os
from random import random, randint
from torchvision.datasets import ImageFolder
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import requests
import time

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

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                          shuffle=True)

loaded_model = get_model(n_classes=5).to(device=device)
loaded_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "..\\models\\authors_model_resnet101_augment.pth"), map_location=torch.device('cpu')))

pred = np.zeros(len(test_data))
true = np.zeros(len(test_data))
loaded_model.eval()

base_url_img = "https://api.telegram.org/bot2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc/sendPhoto"
base_url_txt = "https://api.telegram.org/bot2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc/sendMessage"


def authorsGame(update, context) :
    update.message.reply_text('Game started! mode: authors')
    with torch.no_grad():
        compScore = 0
        userScore = 0

        for i in range(10):

            j = randint(1, test_size)
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
                "chat_id": "627898835",
                #"caption": test_data[j]["label"]
            }
            files = {
                "photo": my_file
            }
            resp = requests.get(base_url_img, data=parameters, files=files)
            out = loaded_model(test_data[j]["pic"].unsqueeze(0).to(device=device))
            pred[j] = torch.argmax(out)
            true[j] = test_data[j]["label"]
            if pred[j] == true[j]:
                compScore += 1

            parameters = {
                "chat_id": "627898835",
                "text": 'Level ' + str(i+1) + '\n' + "Computer's guess: " + authors_test.classes[int(pred[j])] +
                        '\n' + "Correct: " + authors_test.classes[int(true[j])] + '\n' + "Computer score = "
                        + str(compScore)
            }
            resp = requests.get(base_url_txt, data=parameters)

            print("Level ", i)
            print("Computer's guess: ", authors_test.classes[int(pred[j])])
            print("Computer score = ", compScore)