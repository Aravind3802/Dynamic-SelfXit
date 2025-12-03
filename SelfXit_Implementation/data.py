from typing import Tuple
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torch


""" The method below" is used as a data preparation utility, which essentially accesses either the CIFAR-10/CIFAR-100 datasets
and does augmentations and then makes it useful for the input operation"""

def GetCifarLoaders(DatasetName: str,
                    BatchSize: int,
                    NumWorkers: int = 2) -> Tuple[DataLoader, DataLoader]:
    assert DatasetName in ("cifar10", "cifar100")

    TransformTrain = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    TransformTest = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2470, 0.2435, 0.2616)),
    ])

    if DatasetName == "cifar10":
        TrainSet = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=TransformTrain
        )
        TestSet = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=TransformTest
        )
    else:
        TrainSet = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=TransformTrain
        )
        TestSet = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=TransformTest
        )

    PinMemory = torch.cuda.is_available()

    TrainLoader = DataLoader(
        TrainSet, batch_size=BatchSize, shuffle=True,
        num_workers=NumWorkers, pin_memory=PinMemory
    )
    TestLoader = DataLoader(
        TestSet, batch_size=BatchSize, shuffle=False,
        num_workers=NumWorkers, pin_memory=PinMemory
    )
    return TrainLoader, TestLoader

