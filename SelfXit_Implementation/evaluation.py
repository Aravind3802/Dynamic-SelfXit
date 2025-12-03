from dataclasses import dataclass
from typing import List, Optional, Dict
import time
import torch
from torch.utils.data import DataLoader

from .model import EarlyExitResNet


"""The class below is used for the complete evaluation framework for the early-exit model, measuring accuracy, latency, FLOPs,
and exit behavior under both static and dynamic policies, and printing consolidated results."""

@dataclass
class EvalResults:
    Policy: str
    Accuracy: float
    AvgLatencyMs: float
    ExitDistribution: List[float]
    AvgFlops: float
    Description: str


def EvaluatePolicy(Model: EarlyExitResNet,
                   TestLoader: DataLoader,
                   Device: torch.device,
                   Policy: str,
                   Tau: float,
                   GateThreshold: float,
                   ExitFlops: Optional[Dict[int, float]] = None) -> EvalResults:
    assert Policy in ("static", "dynamic")
    Model.to(Device)
    Model.eval()

    TotalCorrect = 0
    TotalSamples = 0
    TotalTime = 0.0
    ExitCounts = torch.zeros(4, dtype=torch.long)
    TotalFlops = 0.0

    if ExitFlops is not None:
        ExitFlopsTensor = torch.tensor(
            [ExitFlops[e] for e in range(4)],
            device=Device
        )
    else:
        ExitFlopsTensor = None

    with torch.no_grad():
        for Images, Targets in TestLoader:
            Images, Targets = Images.to(Device), Targets.to(Device)

            T0 = time.time()
            if Policy == "static":
                Logits, ExitIds = Model.InferenceStatic(Images, Tau=Tau)
            else:
                Logits, ExitIds = Model.InferenceDynamic(
                    Images, GateThreshold=GateThreshold
                )
            T1 = time.time()

            TotalTime += (T1 - T0)
            BatchSize = Targets.size(0)
            TotalSamples += BatchSize
            TotalCorrect += (Logits.argmax(dim=1) == Targets).sum().item()

            for ExitIndex in range(4):
                ExitCounts[ExitIndex] += (ExitIds == ExitIndex).sum().item()

            if ExitFlopsTensor is not None:
                BatchFlops = ExitFlopsTensor[ExitIds].sum().item()
                TotalFlops += BatchFlops

    Accuracy = TotalCorrect / TotalSamples * 100.0
    AvgLatencyMs = (TotalTime / len(TestLoader)) * 1000.0
    ExitDistribution = (ExitCounts.float() / TotalSamples * 100.0).tolist()
    AvgFlops = TotalFlops / TotalSamples if ExitFlopsTensor is not None else 0.0
    Desc = (f"policy=static, tau={Tau}"
            if Policy == "static"
            else f"policy=dynamic, gate_threshold={GateThreshold}")
    return EvalResults(
        Policy=Policy,
        Accuracy=Accuracy,
        AvgLatencyMs=AvgLatencyMs,
        ExitDistribution=ExitDistribution,
        AvgFlops=AvgFlops,
        Description=Desc,
    )


def PrintEvalResults(Results: List[EvalResults]) -> None:
    print("\n================= Evaluation Summary =================")
    for R in Results:
        print(f"Policy: {R.Policy}")
        print(f"  Description: {R.Description}")
        print(f"  Accuracy: {R.Accuracy:.2f}%")
        print(f"  Avg Latency per batch: {R.AvgLatencyMs:.3f} ms")
        print(f"  Avg FLOPs per sample: {R.AvgFlops:.2e}")
        print(f"  Exit distribution (% of samples):")
        print(f"    Exit 0 (after layer2): {R.ExitDistribution[0]:.2f}%")
        print(f"    Exit 1 (after layer3): {R.ExitDistribution[1]:.2f}%")
        print(f"    Exit 2 (after layer4): {R.ExitDistribution[2]:.2f}%")
        print(f"    Exit 3 (final backbone): {R.ExitDistribution[3]:.2f}%")
        print("--------------------------------------------------")


