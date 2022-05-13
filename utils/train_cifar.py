"""
Reference code: https://github.com/weiaicunzai/pytorch-cifar100
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet import resnet50


def get_train_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(train_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return train_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return test_loader


def train(args, model, epoch, loss_function, optimizer, train_loader):
    model.train()

    for batch_index, (images, labels) in enumerate(train_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'Training Epoch: {epoch} '
              f'[{batch_index * args.batch_size + len(images)}/{len(train_loader.dataset)}]\t'
              f'Loss: {loss.item():0.4f}\t'
              f'LR: {optimizer.param_groups[0]["lr"]:0.6f}')


@torch.no_grad()
def evaluation(args, model, epoch, loss_function, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0.0

    for (images, labels) in test_loader:
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print(f'Test set: Epoch: {epoch}, '
          f'Average loss: {test_loss / len(test_loader.dataset):.4f}, '
          f'Accuracy: {correct.float() / len(test_loader.dataset):.4f}')
    print()
    return correct.float() / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-epoch', type=int, default=200, help='training epoch')
    args = parser.parse_args()

    # Model
    model = resnet50()
    if args.gpu:
        model = model.cuda()

    # Dataset
    train_loader = get_train_dataloader(
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        num_workers=4, batch_size=args.batch_size, shuffle=True
    )

    test_loader = get_test_dataloader(
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
        num_workers=4, batch_size=args.batch_size, shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    for epoch in range(1, args.epoch + 1):
        train_scheduler.step(epoch)
        train(args, model, epoch, loss_function, optimizer, train_loader)
        evaluation(args, model, epoch, test_loader)


if __name__ == '__main__':
    main()
