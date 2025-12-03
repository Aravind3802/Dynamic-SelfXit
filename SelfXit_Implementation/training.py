from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .model import EarlyExitResNet

"""This module handles backbone training, freezes the backbone, and trains the early-exit classifiers
via knowledge distillation so they can serve as reliable early predictors for your dynamic MLP gating mechanism. """

def TrainBackbone(Model: EarlyExitResNet,
                  TrainLoader: DataLoader,
                  TestLoader: DataLoader,
                  Device: torch.device,
                  Epochs: int,
                  Lr: float) -> None:
    print(f"[Backbone] Training backbone for {Epochs} epochs (lr={Lr})")
    Model.to(Device)
    Optimizer = torch.optim.SGD(
        Model.Backbone.parameters(),
        lr=Lr, momentum=0.9, weight_decay=5e-4
    )
    Scheduler = torch.optim.lr_scheduler.MultiStepLR(
        Optimizer,
        milestones=[max(1, Epochs // 2), max(2, 3 * Epochs // 4)],
        gamma=0.1,
    )
    Criterion = nn.CrossEntropyLoss()

    for Epoch in range(Epochs):
        Model.train()
        RunningLoss = 0.0
        for Images, Targets in TrainLoader:
            Images, Targets = Images.to(Device), Targets.to(Device)
            Optimizer.zero_grad()
            Logits = Model.ForwardBackboneLogits(Images)
            Loss = Criterion(Logits, Targets)
            Loss.backward()
            Optimizer.step()
            RunningLoss += Loss.item() * Images.size(0)

        Scheduler.step()
        TrainLoss = RunningLoss / len(TrainLoader.dataset)

        Model.eval()
        Correct = 0
        Total = 0
        with torch.no_grad():
            for Images, Targets in TestLoader:
                Images, Targets = Images.to(Device), Targets.to(Device)
                Logits = Model.ForwardBackboneLogits(Images)
                Preds = Logits.argmax(dim=1)
                Correct += (Preds == Targets).sum().item()
                Total += Targets.size(0)
        Acc = Correct / Total * 100.0
        print(f"[Backbone][Epoch {Epoch+1}/{Epochs}] "
              f"loss={TrainLoss:.4f} acc={Acc:.2f}%")


def FreezeBackbone(Model: EarlyExitResNet) -> None:
    for Param in Model.Backbone.parameters():
        Param.requires_grad = False


def TrainExitHeadsDistillation(Model: EarlyExitResNet,
                               TrainLoader: DataLoader,
                               Device: torch.device,
                               Epochs: int,
                               Lr: float,
                               Temperature: float) -> None:
    print(f"[Exits] Training exits via distillation for {Epochs} epochs "
          f"(T={Temperature}, lr={Lr})")
    Model.to(Device)
    FreezeBackbone(Model)
    Model.train()

    Params = list(Model.Exit1.parameters()) + \
             list(Model.Exit2.parameters()) + \
             list(Model.Exit3.parameters())
    Optimizer = torch.optim.Adam(Params, lr=Lr)
    KlDiv = nn.KLDivLoss(reduction="batchmean")

    for Epoch in range(Epochs):
        RunningLoss = 0.0
        for Images, _ in TrainLoader:
            Images = Images.to(Device)
            with torch.no_grad():
                TeacherLogits = Model.ForwardBackboneLogits(Images)
                TeacherProbs = F.softmax(TeacherLogits / Temperature, dim=1)

            ExitLogitsList, _ = Model.ForwardWithExits(Images)
            Loss = 0.0
            for ExitLogits in ExitLogitsList:
                StudentLogProbs = F.log_softmax(ExitLogits / Temperature, dim=1)
                Loss += KlDiv(StudentLogProbs, TeacherProbs)
            Loss = Loss / len(ExitLogitsList)

            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()
            RunningLoss += Loss.item() * Images.size(0)

        TrainLoss = RunningLoss / len(TrainLoader.dataset)
        print(f"[Exits][Epoch {Epoch+1}/{Epochs}] loss={TrainLoss:.4f}")



