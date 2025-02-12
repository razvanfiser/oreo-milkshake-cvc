import os
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import Dataset
import torch.optim as optim

base_url_txt = "https://api.telegram.org/TOKEN/sendMessage"

test_path = os.path.join(os.getcwd(), "..\\styles_dataset\\styles_test\\styles_test")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
styles_test = ImageFolder(test_path, transform=transform)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Styles_Test(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = {"pic": self.X[index][0], "label": self.X[index][1]}
        return sample


test_data = Styles_Test(styles_test)
test_size = len(styles_test)
test_data = torch.utils.data.Subset(test_data, np.arange(test_size))


def get_model(n_classes):
    model = torchvision.models.resnet101(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=n_classes)
    return model


model = get_model(len(styles_test.classes)).to(device=device)

model.to(device=device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.001)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

loaded_model = get_model(n_classes=5).to(device=device)
loaded_model.load_state_dict(torch.load(os.path.join(os.getcwd(), "..\\models\\styles_model_resnet101_augment.pth"), map_location=torch.device('cpu')))

pred = np.zeros(len(test_data))
true = np.zeros(len(test_data))
loaded_model.eval()

