import torch
import click

from torch import optim, nn

from tqdm import tqdm

from matplotlib import pyplot as plt

from torch import nn, optim
import torch.nn.functional as F

from torchvision import datasets, transforms

import numpy as np

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

#@click.group()
def cli():
    pass


#@click.command()
#@click.option("--lr", default=1e-3, help='learning rate to use for training')
#@click.option("--epochs", default=5, help='number of epochs to train for')
#@click.option("--model_checkpoint", default='saved_model.pth', help='path to save model checkpoint')
def train(lr, epochs, model_checkpoint):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()
    # load trainset from data folder
    trainset = torch.load('data/processed/trainset.pt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    training_loss = []

    for e in range(epochs):
        running_loss = 0
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            training_loss.append(running_loss/len(trainloader))

    plt.plot(training_loss, label='Training loss')
    plt.legend(frameon=False)
    # save plot to the visualizations folder
    plt.savefig('src/visualization/training_loss.png')
    # save model to the trained_models folder
    print("Saving model checkpoint to: {}".format(model_checkpoint))
    torch.save(model, model_checkpoint)
    
if __name__ == "__main__":
    cli()
    train(lr=1e-3, epochs=5, model_checkpoint='./src/models/trained_models/saved_model.pt')