import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import random_split
# from tensorboardX import SummaryWriter

def train(model, device, dataset, optimizer, EPOCH, batch_size, dir_ckpt):
    dir_ckpt = Path(dir_ckpt)
    criteration = nn.CrossEntropyLoss()
    model.train()

    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size * 2, shuffle=False,
                            num_workers=4, pin_memory=True, drop_last=True)
    # writer = SummaryWriter()
    for epoch in range(1, EPOCH + 1):
        correct = 0
        for i, (x, y) in tqdm(enumerate(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
            loss = criteration(output, y)
            loss.backward()
            optimizer.step()
        # writer.add_scalar('loss',loss ,epoch)
        print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch, loss, correct, len(train_set),
                                                                 100 * correct / len(train_set)))
        val_num = len(val_set)
        correct, acc = valid(model, device, val_loader, val_num)
        # writer.add_scalar('accuracy',acc ,epoch)
        torch.save(model.state_dict(), dir_ckpt.joinpath(f'ckpt_e{epoch}_acc{acc}.pth'))


def valid(model, device, dataset, val_num):
    model.eval()
    correct = 0
    criteration = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (x, y) in tqdm(enumerate(dataset)):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criteration(output, y)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    acc = correct / val_num
    print(
        "Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss, correct, val_num, 100. * correct / val_num))
    return correct, acc


if __name__ == '__main__':
    import torchvision
    import torch.optim as optim

    DEVICE = torch.device('cuda')
    LR = 0.001
    EPOCH = 50
    BTACH_SIZE = 32
    train_root = './train'
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, 2)
    )
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.09)

    for epoch in range(1, EPOCH + 1):
        train(model, DEVICE, train_set, optimizer, epoch)
        vaild(model, DEVICE, test_set)
