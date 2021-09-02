if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torchvision
    import torch.optim as optim
    from codes.train import train
    from codes.dataset import DealDataset
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR = 0.001
    EPOCH = 50
    BTACH_SIZE = 128
    data_path = './data'
    model_path = './models_1'
    labels = ['电杆', '光交箱', '人手井']

    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Sequential(
        nn.Linear(2048, len(labels))
    )
    model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    dataset = DealDataset(data_path, labels)

    train(model, DEVICE, dataset, optimizer, EPOCH, BTACH_SIZE, model_path)

