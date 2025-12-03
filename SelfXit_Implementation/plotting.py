from typing import Optional
import matplotlib.pyplot as plt

from .evaluation import EvalResults

"""This module generates the key visualizations that compare static and dynamic early-exit policies—plotting accuracy, latency, FLOPs,
and exit-distribution differences—and saves them as figure files for analysis."""

def PlotPolicyComparison(StaticResults: EvalResults,
                         DynamicResults: EvalResults,
                         OutputPrefix: str = "policy_comparison") -> None:
    PolicyNames = ["Static", "Dynamic"]
    AccuracyValues = [StaticResults.Accuracy, DynamicResults.Accuracy]
    LatencyValues = [StaticResults.AvgLatencyMs, DynamicResults.AvgLatencyMs]
    FlopsValues = [StaticResults.AvgFlops, DynamicResults.AvgFlops]

    Figure, Axes = plt.subplots(1, 3, figsize=(12, 4))

    # Accuracy
    AxisAcc = Axes[0]
    Positions = [0, 1]
    AxisAcc.bar(Positions, AccuracyValues)
    AxisAcc.set_xticks(Positions)
    AxisAcc.set_xticklabels(PolicyNames)
    AxisAcc.set_ylabel("Accuracy (%)")
    AxisAcc.set_title("Accuracy")
    for Index, Value in enumerate(AccuracyValues):
        AxisAcc.text(Index, Value + 0.5, f"{Value:.1f}%", ha="center", va="bottom", fontsize=8)

    # Latency
    AxisLat = Axes[1]
    AxisLat.bar(Positions, LatencyValues)
    AxisLat.set_xticks(Positions)
    AxisLat.set_xticklabels(PolicyNames)
    AxisLat.set_ylabel("Latency (ms / batch)")
    AxisLat.set_title("Latency")
    for Index, Value in enumerate(LatencyValues):
        AxisLat.text(Index, Value + 0.5, f"{Value:.1f}", ha="center", va="bottom", fontsize=8)

    # FLOPs
    AxisFlops = Axes[2]
    AxisFlops.bar(Positions, FlopsValues)
    AxisFlops.set_xticks(Positions)
    AxisFlops.set_xticklabels(PolicyNames)
    AxisFlops.set_ylabel("Avg FLOPs per sample")
    AxisFlops.set_title("Compute")
    for Index, Value in enumerate(FlopsValues):
        AxisFlops.text(Index, Value * 1.02, f"{Value:.1e}", ha="center", va="bottom", fontsize=6)

    Figure.suptitle("Static vs Dynamic: Accuracy, Latency, FLOPs", fontsize=12)
    Figure.tight_layout()
    FileName = f"{OutputPrefix}_acc_lat_flops.png"
    plt.savefig(FileName, dpi=150)
    print(f"[Plot] Saved policy comparison to {FileName}")
    plt.close(Figure)


def PlotExitDistributionComparison(StaticResults: EvalResults,
                                   DynamicResults: EvalResults,
                                   OutputPrefix: str = "policy_comparison") -> None:
    ExitLabels = ["Exit0", "Exit1", "Exit2", "Exit3"]
    StaticDist = StaticResults.ExitDistribution
    DynamicDist = DynamicResults.ExitDistribution

    NumExits = len(ExitLabels)
    BarWidth = 0.35
    Positions = list(range(NumExits))
    PositionsStatic = [Pos - BarWidth / 2 for Pos in Positions]
    PositionsDynamic = [Pos + BarWidth / 2 for Pos in Positions]

    Figure, Axis = plt.subplots(1, 1, figsize=(8, 4))
    Axis.bar(PositionsStatic, StaticDist, width=BarWidth, label="Static")
    Axis.bar(PositionsDynamic, DynamicDist, width=BarWidth, label="Dynamic")

    Axis.set_xticks(Positions)
    Axis.set_xticklabels(ExitLabels)
    Axis.set_ylabel("Samples Exited (%)")
    Axis.set_title("Exit Distribution: Static vs Dynamic")
    Axis.legend()

    for Index, Value in enumerate(StaticDist):
        Axis.text(PositionsStatic[Index], Value + 0.5, f"{Value:.1f}",
                  ha="center", va="bottom", fontsize=7)
    for Index, Value in enumerate(DynamicDist):
        Axis.text(PositionsDynamic[Index], Value + 0.5, f"{Value:.1f}",
                  ha="center", va="bottom", fontsize=7)

    Figure.tight_layout()
    FileName = f"{OutputPrefix}_exit_distribution.png"
    plt.savefig(FileName, dpi=150)
    print(f"[Plot] Saved exit distribution comparison to {FileName}")
    plt.close(Figure)


