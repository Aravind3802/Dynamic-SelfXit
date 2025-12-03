import torch
import torch.nn.functional as F

"""This block provides small but essential helper functions for device selection, accuracy measurement,
and entropy computation — all of which support training, evaluation, and the dynamic gating mechanism in the SelfXit model. """

def GetDevice() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def AccuracyFromLogits(Logits: torch.Tensor, Targets: torch.Tensor) -> float:
    Predictions = Logits.argmax(dim=1)
    Correct = (Predictions == Targets).sum().item()
    return Correct / Targets.size(0)


def EntropyFromProbs(Probs: torch.Tensor, Eps: float = 1e-8) -> torch.Tensor:
    return -(Probs * (Probs + Eps).log()).sum(dim=1)

