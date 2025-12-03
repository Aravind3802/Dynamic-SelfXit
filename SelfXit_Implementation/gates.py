from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import EarlyExitResNet, GateMLP
from .training import FreezeBackbone

"""This code collects self-supervised gate training data from early-exit vs. final predictions, and
trains the GateMLPs to dynamically decide whether each exit should fire, enabling learned early-exit policies instead of static thresholds. """

def CollectGateTrainingData(Model: EarlyExitResNet,
                            TrainLoader: DataLoader,
                            Device: torch.device,
                            MaxBatches: int,
                            GateLabelConf: float,
                            Tolerance: float) -> Dict[str, torch.Tensor]:
    
    EffectiveConf = max(0.0, min(1.0, GateLabelConf - Tolerance))
    print(f"[Gates] Collecting gate data (max {MaxBatches} batches)")
    print(f"[Gates] GateLabelConf={GateLabelConf:.3f}, "
          f"Tolerance={Tolerance:.3f}, EffectiveConf={EffectiveConf:.3f}")

    Model.to(Device)
    FreezeBackbone(Model)
    Model.eval()

    Feats1, Labels1 = [], []
    Feats2, Labels2 = [], []
    Feats3, Labels3 = [], []

    with torch.no_grad():
        for BatchIndex, (Images, _) in enumerate(TrainLoader):
            if BatchIndex >= MaxBatches:
                break
            Images = Images.to(Device)
            ExitLogitsList, FinalLogits = Model.ForwardWithExits(Images)
            FinalPreds = FinalLogits.argmax(dim=1)

            DepthNorms = [0.33, 0.66, 1.0]
            FeatsAcc = [Feats1, Feats2, Feats3]
            LabelsAcc = [Labels1, Labels2, Labels3]

            for ExitIdx, (ExitLogits, DepthNorm) in enumerate(
                zip(ExitLogitsList, DepthNorms)
            ):
                ExitProbs = F.softmax(ExitLogits, dim=1)
                ExitPreds = ExitLogits.argmax(dim=1)
                MaxConf, _ = ExitProbs.max(dim=1)

                Labels = ((ExitPreds == FinalPreds) &
                          (MaxConf >= EffectiveConf)).float()
                Features = Model.MakeGateFeatures(ExitProbs, ExitLogits, DepthNorm)

                FeatsAcc[ExitIdx].append(Features.cpu())
                LabelsAcc[ExitIdx].append(Labels.cpu())

    def StackOrEmpty(ListOfTensors: List[torch.Tensor]) -> torch.Tensor:
        if len(ListOfTensors) == 0:
            return torch.empty(0)
        return torch.cat(ListOfTensors, dim=0)

    Feats1 = StackOrEmpty(Feats1)
    Feats2 = StackOrEmpty(Feats2)
    Feats3 = StackOrEmpty(Feats3)
    Labels1 = StackOrEmpty(Labels1)
    Labels2 = StackOrEmpty(Labels2)
    Labels3 = StackOrEmpty(Labels3)

    for Name, Feats, Labs in [
        ("Exit1", Feats1, Labels1),
        ("Exit2", Feats2, Labels2),
        ("Exit3", Feats3, Labels3),
    ]:
        if Feats.numel() == 0:
            print(f"[Gates] {Name}: 0 samples")
        else:
            Pos = int(Labs.sum().item())
            Neg = Labs.numel() - Pos
            print(f"[Gates] {Name}: {Feats.size(0)} samples (pos={Pos}, neg={Neg})")

    return {
        "feats1": Feats1, "labels1": Labels1,
        "feats2": Feats2, "labels2": Labels2,
        "feats3": Feats3, "labels3": Labels3,
    }


def TrainGates(Model: EarlyExitResNet,
               GateData: Dict[str, torch.Tensor],
               Device: torch.device,
               Epochs: int,
               Lr: float,
               BatchSize: int = 512) -> None:
    print(f"[Gates] Training gates for {Epochs} epochs (lr={Lr})")
    Model.to(Device)
    FreezeBackbone(Model)
    Model.train()

    def TrainSingleGate(Gate: GateMLP,
                        Features: torch.Tensor,
                        Labels: torch.Tensor,
                        Name: str):
        if Features.size(0) == 0:
            print(f"[Gates] {Name}: no data, skipping")
            return

        Positives = Labels.sum().item()
        Total = Labels.numel()
        Negatives = Total - Positives
        if Positives == 0 or Negatives == 0:
            PosWeight = torch.tensor([1.0], device=Device)
        else:
            PosWeight = torch.tensor([Negatives / max(Positives, 1e-6)],
                                     device=Device)

        Dataset = torch.utils.data.TensorDataset(Features, Labels)
        Loader = DataLoader(Dataset, batch_size=BatchSize, shuffle=True)

        Optimizer = torch.optim.Adam(Gate.parameters(), lr=Lr)
        BCE = nn.BCEWithLogitsLoss(pos_weight=PosWeight)

        for Epoch in range(Epochs):
            RunningLoss = 0.0
            for XBatch, YBatch in Loader:
                XBatch, YBatch = XBatch.to(Device), YBatch.to(Device)
                Logits = Gate(XBatch)
                Loss = BCE(Logits, YBatch)
                Optimizer.zero_grad()
                Loss.backward()
                Optimizer.step()
                RunningLoss += Loss.item() * XBatch.size(0)
            AvgLoss = RunningLoss / len(Loader.dataset)
            print(f"[Gates][{Name}][Epoch {Epoch+1}/{Epochs}] loss={AvgLoss:.4f}")

    TrainSingleGate(Model.Gate1, GateData["feats1"], GateData["labels1"], "Gate1")
    TrainSingleGate(Model.Gate2, GateData["feats2"], GateData["labels2"], "Gate2")
    TrainSingleGate(Model.Gate3, GateData["feats3"], GateData["labels3"], "Gate3")


