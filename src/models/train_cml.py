import torch
from torch import optim, nn
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch import nn, optim
from src.models.model import Model

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
    
    # assume we have a trained model
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    preds, target = [], []
    for batch in trainloader:
        x, y = batch
        probs = model(x)
        preds.append(probs.argmax(dim=-1))
        target.append(y.detach())

    target = torch.cat(target, dim=0)
    preds = torch.cat(preds, dim=0)

    report = classification_report(target, preds)
    with open("classification_report.txt", 'w') as outfile:
        outfile.write(report)
    confmat = confusion_matrix(target, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix = confmat, )
    plt.savefig('confusion_matrix.png')

    
if __name__ == "__main__":
    cli()
    train(lr=1e-3, epochs=5, model_checkpoint='./src/models/trained_models/saved_model.pt')