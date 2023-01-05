import torch
import click
from torch import optim, nn

@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint='./src/models/trained_models/saved_model.pt'):
    print("Evaluating until hitting the ceiling")
    print("From Checkpoint: ", model_checkpoint)
    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    criterion = nn.NLLLoss()
    _, testset = torch.load('data/processed/testset.pt')
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    with torch.no_grad():
        model.eval()
        
        test_loss = 0
        accuracy = 0
        for images, labels in testloader:
            logprobs = model(images)
            loss = criterion(logprobs, labels)
            test_loss += loss.item()
            top_p, top_class = logprobs.topk(1, dim=1)
            
            equals = top_class == labels.view(*top_class.shape)
            #print(torch.mean(equals.type(torch.FloatTensor)))
            accuracy += torch.mean(equals.type(torch.FloatTensor))
            
        accuracy = accuracy/len(testloader)
        print(f'Accuracy: {accuracy.item()*100}%')
        model.train()
        
if __name__ == "__main__":
    evaluate(model_checkpoint='./src/models/trained_models/saved_model.pt')