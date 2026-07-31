import json

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.ansatz_builder import build_qnn_from_spec, param_labels
from src.data_encoding import embedding_unitary
from src.train import train_loop

DEVICE_NAME = "default.qubit"


def _tiny_loader(num_qubits: int) -> DataLoader:
    size = 2 ** (num_qubits // 2)
    images = torch.rand(4, size, size)
    for i in range(4):
        images[i] = images[i] / torch.linalg.norm(images[i].reshape(-1))
    embeddings = torch.stack([embedding_unitary(img) for img in images])
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    return DataLoader(TensorDataset(embeddings, labels), batch_size=2)


def test_custom_spec_trains_end_to_end(tmp_path, monkeypatch):
    """A hand-drawn circuit (not one of the 5 named architectures) must be
    trainable through the same train_loop used by the standard pipeline,
    and produce a self-contained checkpoint an inference service could
    reload (see src/api.py's "circuit_spec" branch)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WANDB_MODE", "disabled")

    num_qubits = 4
    spec = [
        {"gate": "RX", "wires": [0], "param": {"init": "random", "frozen": False}},
        {"gate": "RY", "wires": [1], "param": {"init": "random", "frozen": False}},
        {
            "gate": "RZ",
            "wires": [2],
            "param": {"init": "custom", "value": 0.2, "frozen": True},
        },
        {"gate": "CNOT", "wires": [0, 1]},
        {"gate": "CNOT", "wires": [2, 3]},
    ]
    qnn, initial_params, resolved_spec = build_qnn_from_spec(
        DEVICE_NAME, num_qubits, 0.0, spec
    )
    names = param_labels(resolved_spec)

    loader = _tiny_loader(num_qubits)

    result = train_loop(
        loader,
        loader,
        loader,
        epochs=2,
        learning_rate=0.1,
        patience=5,
        min_delta=1e-4,
        dev="cpu",
        seed=0,
        N=4,
        dataset="mnist",
        qnn=qnn,
        initial_params=initial_params,
        param_names=names,
        run_name="custom_test",
        checkpoint_config={
            "device": DEVICE_NAME,
            "num_qubits": num_qubits,
            "p_err": 0.0,
            "circuit_spec": resolved_spec,
            "img_size": 4,
        },
        wandb_extra_config={"architecture": "custom"},
        verbose=False,
    )

    trained_params, _, train_loss_hist, *_ = result
    assert len(train_loss_hist) == 2
    assert torch.isfinite(trained_params).all()

    checkpoints = list(tmp_path.rglob("final_model.pt"))
    assert len(checkpoints) == 1
    checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    assert checkpoint["config"]["circuit_spec"] == resolved_spec
    assert checkpoint["params"].shape == initial_params.shape

    job_dir = checkpoints[0].parent
    assert (job_dir / "confusion_matrix.png").exists()
    assert (job_dir / "circuit.txt").exists()

    summary = json.loads((job_dir / "summary.json").read_text())
    assert summary["param_names"] == names
    assert len(summary["final_params"]) == len(names)
    assert summary["val_accuracy"] == result[5][0]
    assert summary["p4m_equivariance"]["checked"] is True
    assert isinstance(summary["p4m_equivariance"]["is_invariant"], bool)
