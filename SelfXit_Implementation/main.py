
import argparse
from typing import List, Optional

from .utils import GetDevice
from .data import GetCifarLoaders
from .model import EarlyExitResNet
from .flops import ComputeExitFlops
from .training import TrainBackbone, TrainExitHeadsDistillation
from .gates import CollectGateTrainingData, TrainGates
from .checkpoint import SaveCheckpoint, LoadCheckpoint
from .evaluation import EvalResults, EvaluatePolicy, PrintEvalResults
from .plotting import PlotPolicyComparison, PlotExitDistributionComparison

"""This module is the main entry point. This module configures, trains (backbone, exits, gates), evaluates static vs dynamic early-exit policies on CIFAR with FLOPs + latency metrics,
and plots how your MLP-based dynamic gating compares to a static threshold."""

def Main() -> None:
    Parser = argparse.ArgumentParser(
        description="MVP: SelfXit Early Exits + MLP Dynamic Gates (PascalCase, Modular)"
    )

    Parser.add_argument("--dataset", dest="Dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])
    Parser.add_argument("--model", dest="ModelName", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    Parser.add_argument("--batch_size", dest="BatchSize", type=int, default=128)

    Parser.add_argument("--epochs_backbone", dest="EpochsBackbone", type=int, default=5)
    Parser.add_argument("--epochs_exits", dest="EpochsExits", type=int, default=10)
    Parser.add_argument("--epochs_gates", dest="EpochsGates", type=int, default=5)

    Parser.add_argument("--lr_backbone", dest="LrBackbone", type=float, default=0.1)
    Parser.add_argument("--lr_exits", dest="LrExits", type=float, default=1e-3)
    Parser.add_argument("--lr_gates", dest="LrGates", type=float, default=1e-3)

    Parser.add_argument("--temperature", dest="Temperature", type=float, default=1.0)

    Parser.add_argument("--tau", dest="Tau", type=float, default=0.9)
    Parser.add_argument("--gate_threshold", dest="GateThreshold", type=float, default=0.8)

    Parser.add_argument("--gate_label_conf", dest="GateLabelConf",
                        type=float, default=0.8)
    Parser.add_argument("--tolerance", dest="Tolerance",
                        type=float, default=0.0)
    Parser.add_argument("--gate_max_batches", dest="GateMaxBatches",
                        type=int, default=500)

    Parser.add_argument("--policy", dest="Policy", type=str, default="both",
                        choices=["static", "dynamic", "both"])

    Parser.add_argument("--num_workers", dest="NumWorkers",
                        type=int, default=0,
                        help="DataLoader workers")

    Parser.add_argument("--no_pretrained_backbone", dest="UsePretrainedBackbone",
                        action="store_false")
    Parser.set_defaults(UsePretrainedBackbone=True)

    Parser.add_argument("--eval_only", dest="EvalOnly", action="store_true")

    Parser.add_argument("--save_checkpoint", dest="SaveCheckpointPath",
                        type=str, default="")
    Parser.add_argument("--load_checkpoint", dest="LoadCheckpointPath",
                        type=str, default="")

    Args = Parser.parse_args()

    Device = GetDevice()
    print(f"Using device: {Device}")

    TrainLoader, TestLoader = GetCifarLoaders(
        Args.Dataset,
        Args.BatchSize,
        NumWorkers=Args.NumWorkers
    )
    NumClasses = 10 if Args.Dataset == "cifar10" else 100

    Model = EarlyExitResNet(
        Args.ModelName,
        NumClasses=NumClasses,
        UsePretrained=Args.UsePretrainedBackbone,
    )

    if Args.LoadCheckpointPath:
        LoadCheckpoint(Model, Args.LoadCheckpointPath, Device)

    ExitFlops = ComputeExitFlops(Model, Device)

    if not Args.EvalOnly:
        if Args.EpochsBackbone > 0:
            TrainBackbone(
                Model, TrainLoader, TestLoader,
                Device=Device,
                Epochs=Args.EpochsBackbone,
                Lr=Args.LrBackbone,
            )
        else:
            print("[Backbone] Skipping backbone training (EpochsBackbone=0).")

        if Args.EpochsExits > 0:
            TrainExitHeadsDistillation(
                Model,
                TrainLoader,
                Device=Device,
                Epochs=Args.EpochsExits,
                Lr=Args.LrExits,
                Temperature=Args.Temperature,
            )
        else:
            print("[Exits] Skipping exit training (EpochsExits=0).")

        if Args.EpochsGates > 0:
            GateData = CollectGateTrainingData(
                Model,
                TrainLoader,
                Device=Device,
                MaxBatches=Args.GateMaxBatches,
                GateLabelConf=Args.GateLabelConf,
                Tolerance=Args.Tolerance,
            )
            TrainGates(
                Model,
                GateData,
                Device=Device,
                Epochs=Args.EpochsGates,
                Lr=Args.LrGates,
            )
        else:
            print("[Gates] Skipping gate training (EpochsGates=0).")

        if Args.SaveCheckpointPath:
            SaveCheckpoint(Model, Args.SaveCheckpointPath)
    else:
        if not Args.LoadCheckpointPath:
            print("[EvalOnly] No checkpoint path provided; evaluating untrained model.")

    Results: List[EvalResults] = []

    StaticResults: Optional[EvalResults] = None
    DynamicResults: Optional[EvalResults] = None

    if Args.Policy in ("static", "both"):
        StaticResults = EvaluatePolicy(
            Model,
            TestLoader,
            Device=Device,
            Policy="static",
            Tau=Args.Tau,
            GateThreshold=Args.GateThreshold,
            ExitFlops=ExitFlops,
        )
        Results.append(StaticResults)

    if Args.Policy in ("dynamic", "both"):
        DynamicResults = EvaluatePolicy(
            Model,
            TestLoader,
            Device=Device,
            Policy="dynamic",
            Tau=Args.Tau,
            GateThreshold=Args.GateThreshold,
            ExitFlops=ExitFlops,
        )
        Results.append(DynamicResults)

    PrintEvalResults(Results)

    if StaticResults is not None and DynamicResults is not None:
        PlotPolicyComparison(
            StaticResults,
            DynamicResults,
            OutputPrefix="selfxit_mlp_mvp",
        )
        PlotExitDistributionComparison(
            StaticResults,
            DynamicResults,
            OutputPrefix="selfxit_mlp_mvp",
        )
    else:
        print("[Plot] Need both static and dynamic results to plot comparison.")


if __name__ == "__main__":
    Main()

