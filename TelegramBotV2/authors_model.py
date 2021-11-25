import os
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import requests

base_url_txt = "https://api.telegram.org/bot2121589320:AAFe0WiStJID-1QTs2Gfmn6vJqzU2AjwMPc/sendMessage"

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


def computer_guess_authors(message, chat_id, ind):
    if message.text == "Fernand Leger" or message.text == "Ivan Aivazovsky" or message.text == "Rembrandt"\
            or message.text == "Salvador Dali" or message.text == "Vincent Van Gogh":
        print(message.text)
        comp_score = 0
        user_score = 0
        with torch.no_grad():
            out = loaded_model(test_data[ind]["pic"].unsqueeze(0).to(device=device))
            ## FIX LA LINIA ASTA (77) INCEPE EROAREA
            pred[ind] = torch.argmax(out)
            true[ind] = test_data[ind]["label"]
            if pred[ind] == true[ind]:
                comp_score += 1

            parameters = {
                "chat_id": chat_id,
                "text": "Correct: " + authors_test.classes[int(true[ind])] +
                        '\n' + "Computer's guess: " + authors_test.classes[int(pred[ind])]
            }
        resp = requests.get(base_url_txt, data=parameters)



