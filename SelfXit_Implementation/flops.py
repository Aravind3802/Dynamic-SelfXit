from typing import Dict, Tuple
import torch
import torch.nn as nn
from collections import defaultdict

from .model import EarlyExitResNet

"""This method below measures the exact number of FLOPs each early-exit path consumes by hooking
into the model and computing per-layer operations, enabling accurate evaluation of compute savings in SelfXit. """

def ComputeExitFlops(Model: EarlyExitResNet,
                     Device: torch.device,
                     InputSize: Tuple[int, int, int, int] = (1, 3, 32, 32)
                     ) -> Dict[int, float]:
    
    Model.to(Device)
    Model.eval()

    ModuleToGroup: Dict[nn.Module, str] = {}

    for Name, Module in Model.named_modules():
        if Name.startswith("Backbone.conv1") or \
           Name.startswith("Backbone.bn1") or \
           Name.startswith("Backbone.relu") or \
           Name.startswith("Backbone.layer1"):
            ModuleToGroup[Module] = "stem_layer1"
        elif Name.startswith("Backbone.layer2"):
            ModuleToGroup[Module] = "layer2"
        elif Name.startswith("Backbone.layer3"):
            ModuleToGroup[Module] = "layer3"
        elif Name.startswith("Backbone.layer4"):
            ModuleToGroup[Module] = "layer4"
        elif Name.startswith("Exit1"):
            ModuleToGroup[Module] = "exit1"
        elif Name.startswith("Exit2"):
            ModuleToGroup[Module] = "exit2"
        elif Name.startswith("Exit3"):
            ModuleToGroup[Module] = "exit3"
        elif Name.startswith("Backbone.avgpool"):
            ModuleToGroup[Module] = "avgpool"
        elif Name.startswith("Backbone.fc"):
            ModuleToGroup[Module] = "fc"

    FlopsByGroup: Dict[str, float] = defaultdict(float)
    Hooks = []

    def ConvHook(Module: nn.Conv2d, Input, Output):
        if not isinstance(Output, torch.Tensor):
            return
        BatchSize = Output.shape[0]
        OutChannels = Output.shape[1]
        OutH = Output.shape[2]
        OutW = Output.shape[3]

        KernelH, KernelW = Module.kernel_size
        InChannels = Module.in_channels
        Groups = Module.groups

        FlopsPerPosition = 2.0 * (InChannels / Groups) * KernelH * KernelW
        TotalFlops = FlopsPerPosition * OutChannels * OutH * OutW * BatchSize

        GroupName = ModuleToGroup.get(Module, None)
        if GroupName is not None:
            FlopsByGroup[GroupName] += TotalFlops

    def LinearHook(Module: nn.Linear, Input, Output):
        if not isinstance(Output, torch.Tensor):
            return
        BatchSize = Output.shape[0]
        InFeatures = Module.in_features
        OutFeatures = Module.out_features
        TotalFlops = 2.0 * InFeatures * OutFeatures * BatchSize

        GroupName = ModuleToGroup.get(Module, None)
        if GroupName is not None:
            FlopsByGroup[GroupName] += TotalFlops

    for Module in Model.modules():
        if isinstance(Module, nn.Conv2d):
            Hooks.append(Module.register_forward_hook(ConvHook))
        elif isinstance(Module, nn.Linear):
            Hooks.append(Module.register_forward_hook(LinearHook))

    DummyInput = torch.randn(*InputSize, device=Device)
    with torch.no_grad():
        _ = Model.ForwardWithExits(DummyInput)

    for Hook in Hooks:
        Hook.remove()

    StemLayer1Flops = FlopsByGroup["stem_layer1"]
    Layer2Flops = FlopsByGroup["layer2"]
    Layer3Flops = FlopsByGroup["layer3"]
    Layer4Flops = FlopsByGroup["layer4"]
    Exit1Flops = FlopsByGroup["exit1"]
    Exit2Flops = FlopsByGroup["exit2"]
    Exit3Flops = FlopsByGroup["exit3"]
    AvgpoolFlops = FlopsByGroup["avgpool"]
    FcFlops = FlopsByGroup["fc"]

    Exit0Cost = StemLayer1Flops + Layer2Flops + Exit1Flops
    Exit1Cost = Exit0Cost + Layer3Flops + Exit2Flops
    Exit2Cost = Exit1Cost + Layer4Flops + Exit3Flops
    Exit3Cost = Exit2Cost + AvgpoolFlops + FcFlops

    ExitFlops = {
        0: Exit0Cost,
        1: Exit1Cost,
        2: Exit2Cost,
        3: Exit3Cost,
    }

    print("[FLOPs] Estimated FLOPs per exit path (per sample):")
    for ExitId, Cost in ExitFlops.items():
        print(f"  Exit {ExitId}: {Cost:.2e} FLOPs")

    return ExitFlops
