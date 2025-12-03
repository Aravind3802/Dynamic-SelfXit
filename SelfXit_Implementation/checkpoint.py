import torch
from .model import EarlyExitResNet


"""The functions below are used a methods to save the models parameters as a checkpoint
and then load and use them at any time."""

def SaveCheckpoint(Model: EarlyExitResNet, FilePath: str) -> None:
    if not FilePath:
        return
    State = Model.state_dict()
    torch.save(State, FilePath)
    print(f"[Checkpoint] Saved model to {FilePath}")


def LoadCheckpoint(Model: EarlyExitResNet,
                   FilePath: str,
                   Device: torch.device) -> None:
    if not FilePath:
        return
    State = torch.load(FilePath, map_location=Device)
    Model.load_state_dict(State)
    print(f"[Checkpoint] Loaded model from {FilePath}")


