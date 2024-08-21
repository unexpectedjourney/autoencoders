import torchvision.datasets as datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader


train_cifar10_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.247, 0.243, 0.261),
    )
])

test_cifar10_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.247, 0.243, 0.261),
    )
])


train_mnist_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

test_mnist_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])


def get_cifar10_datasets():
    trainset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_cifar10_transforms
    )
    valset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_cifar10_transform
    )
    return trainset, valset


def get_cifar10_dataloaders(batch_size):
    trainset, valset = get_cifar10_datasets()
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size)
    return train_loader, val_loader


def get_mnist_datasets():
    trainset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_mnist_transforms
    )
    valset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_mnist_transform
    )
    return trainset, valset


def get_mnist_dataloaders(batch_size):
    trainset, valset = get_mnist_datasets()
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size)
    return train_loader, val_loader
