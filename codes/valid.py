import torch
import tqdm
from torch import nn


def valid(model, device, dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = nn.CrossEntropyLoss(output, y)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    print(
        "Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss, correct, len(dataset), 100. * correct / len(dataset)))
