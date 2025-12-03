from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights
from .utils import EntropyFromProbs

"""This code defines the full SelfXit model: a modified ResNet backbone with three early-exit classifiers
and three MLP gate modules, plus all logic for generating gate features and conducting static vs dynamic early exits. """

class ExitHead(nn.Module):
    def __init__(self, InChannels: int, NumClasses: int, HiddenDim: int = 512):
        super().__init__()
        self.Pool = nn.AdaptiveAvgPool2d((1, 1))
        self.Fc1 = nn.Linear(InChannels, HiddenDim)
        self.Dropout = nn.Dropout(p=0.1)
        self.Fc2 = nn.Linear(HiddenDim, NumClasses)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.Pool(X)
        X = X.view(X.size(0), -1)
        X = F.relu(self.Fc1(X))
        X = self.Dropout(X)
        X = self.Fc2(X)
        return X


class GateMLP(nn.Module):
    
    def __init__(self, InDim: int = 5, HiddenDim: int = 16):
        super().__init__()
        self.Net = nn.Sequential(
            nn.Linear(InDim, HiddenDim),
            nn.ReLU(inplace=True),
            nn.Linear(HiddenDim, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.Net(X).squeeze(-1)


class EarlyExitResNet(nn.Module):
    
    def __init__(self,
                 ModelName: str,
                 NumClasses: int,
                 UsePretrained: bool = True):
        super().__init__()
        assert ModelName in ("resnet18", "resnet50")
        self.ModelName = ModelName

        if ModelName == "resnet18":
            Weights = ResNet18_Weights.IMAGENET1K_V1 if UsePretrained else None
            Backbone = torchvision.models.resnet18(weights=Weights)
            FeatDimLayer2 = 128
            FeatDimLayer3 = 256
            FeatDimLayer4 = 512
        else:
            Weights = ResNet50_Weights.IMAGENET1K_V1 if UsePretrained else None
            Backbone = torchvision.models.resnet50(weights=Weights)
            FeatDimLayer2 = 512
            FeatDimLayer3 = 1024
            FeatDimLayer4 = 2048

        Backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        Backbone.maxpool = nn.Identity()
        Backbone.fc = nn.Linear(Backbone.fc.in_features, NumClasses)

        self.Backbone = Backbone

        self.Exit1 = ExitHead(FeatDimLayer2, NumClasses)
        self.Exit2 = ExitHead(FeatDimLayer3, NumClasses)
        self.Exit3 = ExitHead(FeatDimLayer4, NumClasses)

        self.Gate1 = GateMLP()
        self.Gate2 = GateMLP()
        self.Gate3 = GateMLP()

    def ForwardBackboneLogits(self, X: torch.Tensor) -> torch.Tensor:
        return self.Backbone(X)

    def ForwardWithExits(self, X: torch.Tensor):
        X = self.Backbone.conv1(X)
        X = self.Backbone.bn1(X)
        X = self.Backbone.relu(X)

        X = self.Backbone.layer1(X)
        X = self.Backbone.layer2(X)
        Feats2 = X

        X = self.Backbone.layer3(X)
        Feats3 = X

        X = self.Backbone.layer4(X)
        Feats4 = X

        Pooled = self.Backbone.avgpool(Feats4)
        Pooled = torch.flatten(Pooled, 1)
        FinalLogits = self.Backbone.fc(Pooled)

        ExitLogits1 = self.Exit1(Feats2)
        ExitLogits2 = self.Exit2(Feats3)
        ExitLogits3 = self.Exit3(Feats4)
        return [ExitLogits1, ExitLogits2, ExitLogits3], FinalLogits

    def MakeGateFeatures(self,
                         ExitProbs: torch.Tensor,
                         ExitLogits: torch.Tensor,
                         DepthNorm: float) -> torch.Tensor:
        with torch.no_grad():
            MaxConf, _ = ExitProbs.max(dim=1)
            Ent = EntropyFromProbs(ExitProbs)
            Top2Vals, _ = torch.topk(ExitLogits, k=2, dim=1)
            Margin = Top2Vals[:, 0] - Top2Vals[:, 1]
            L2Norm = ExitLogits.norm(p=2, dim=1)
            DepthVec = torch.full_like(MaxConf, DepthNorm)
        Features = torch.stack(
            [MaxConf, Ent, Margin, DepthVec, L2Norm],
            dim=1,
        )
        return Features

    def InferenceStatic(self,
                        X: torch.Tensor,
                        Tau: float = 0.9):
        
        BatchSize = X.size(0)
        NumClasses = self.Backbone.fc.out_features
        LogitsOut = torch.zeros(BatchSize, NumClasses, device=X.device)
        ExitIds = torch.zeros(BatchSize, dtype=torch.long, device=X.device)

        ExitsLogits, FinalLogits = self.ForwardWithExits(X)
        ExitsProbs = [F.softmax(L, dim=1) for L in ExitsLogits]
        Decided = torch.zeros(BatchSize, dtype=torch.bool, device=X.device)

        for Index, ExitLogits in enumerate(ExitsLogits):
            Probs = ExitsProbs[Index]
            MaxConf, _ = Probs.max(dim=1)
            ShouldExit = (MaxConf >= Tau) & (~Decided)
            if ShouldExit.any():
                Idx = ShouldExit.nonzero(as_tuple=False).squeeze(1)
                LogitsOut[Idx] = ExitLogits[Idx]
                ExitIds[Idx] = Index
                Decided[Idx] = True

        Remaining = ~Decided
        if Remaining.any():
            Idx = Remaining.nonzero(as_tuple=False).squeeze(1)
            LogitsOut[Idx] = FinalLogits[Idx]
            ExitIds[Idx] = 3
        return LogitsOut, ExitIds

    def InferenceDynamic(self,
                         X: torch.Tensor,
                         GateThreshold: float = 0.8):
        
        BatchSize = X.size(0)
        NumClasses = self.Backbone.fc.out_features
        LogitsOut = torch.zeros(BatchSize, NumClasses, device=X.device)
        ExitIds = torch.zeros(BatchSize, dtype=torch.long, device=X.device)

        ExitsLogits, FinalLogits = self.ForwardWithExits(X)
        ExitsProbs = [F.softmax(L, dim=1) for L in ExitsLogits]
        Gates = [self.Gate1, self.Gate2, self.Gate3]
        DepthNorms = [0.33, 0.66, 1.0]
        Decided = torch.zeros(BatchSize, dtype=torch.bool, device=X.device)

        for Index, (ExitLogits, ExitProbs, Gate, DepthNorm) in enumerate(
            zip(ExitsLogits, ExitsProbs, Gates, DepthNorms)
        ):
            Features = self.MakeGateFeatures(ExitProbs, ExitLogits, DepthNorm)
            GateLogit = Gate(Features)
            GateProb = torch.sigmoid(GateLogit)
            ShouldExit = (GateProb >= GateThreshold) & (~Decided)
            if ShouldExit.any():
                Idx = ShouldExit.nonzero(as_tuple=False).squeeze(1)
                LogitsOut[Idx] = ExitLogits[Idx]
                ExitIds[Idx] = Index
                Decided[Idx] = True

        Remaining = ~Decided
        if Remaining.any():
            Idx = Remaining.nonzero(as_tuple=False).squeeze(1)
            LogitsOut[Idx] = FinalLogits[Idx]
            ExitIds[Idx] = 3
        return LogitsOut, ExitIds


