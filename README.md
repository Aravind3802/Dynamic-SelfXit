Modern deep learning networks achieve good performance for various applications like vision, language and multimodal tasks but their computational cost is a major problem for real time and resource constrained deployment. For conventional inference pipelines, each input goes through the entire network even when early intermediate representations are already discriminative. This leads to unnecessary computation, energy consumption and latency in edge devices, mobile platforms and high throughout inference systems.

Early exit architectures address this problem by adding auxiliary classifiers to intermediate layers and hence, enable exit of samples as soon as we obtain confidence prediction. This significantly reduces average computation while preserving accuracy. However, when to exit still remains a problem. Static confidence thresholds, like the ones used in SelfXit and BranchyNet treat all samples uniformly and cannot adapt to difficult samples. SelfXit paper gives us a confidence based gating mechanism in which feature level embeddings improve exit decisions.

Based on this, we reimplement the SelfXit framework and extend it by adding a lightweight MLP that learns to predict exit decisions from a set of statistics that we get from each exit head. Instead of relying on confidence alone, the MLP will introduce features like softmax entropy and depth normalization indicators that provide a better characterization of prediction reliability and allows the gate to learn decision boundaries that improve exiting behavior.

To evaluate both these approaches, we integrate early exits into a CIFAR based ResNet—18 backbone. Overall, this work demonstrates that learned exit policies, using simple MLP can improve early exit networks and help us achieve low latency adaptive inference systems

To setup the environment:(Mac/Linux)

```bash
git clone https://github.com/<your-username>/DynamicSelfXit.git
cd DynamicSelfXit
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To run a smoke test:(Mac/Linux)

```bash
python3 -m SelfXit_Implementation.main \
    --dataset cifar10 \
    --model resnet18 \
    --epochs_backbone 0 \
    --epochs_exits 1 \
    --epochs_gates 1 \
    --gate_max_batches 50 \
    --policy both \
    --num_workers 0
```

