"""End-to-end training + evaluation of one model with one seed.

Order of operations (the test set is loaded only in step 6):
    1. load the train and validation splits of the prepared dataset
    2. pos_weight from TRAIN labels only
    3. seed everything, build the model
    4. train; after every epoch evaluate on validation; keep the best checkpoint
    5. reload the best checkpoint
    6. load the TEST split, predict once, compute slice- and patient-level metrics
    7. save predictions, metrics, curves
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flyconnectome_mri.connectome.loader import ConnectomeGraph, load_edge_list, load_node_ids
from flyconnectome_mri.datasets.mri_dataset import METADATA_NAME, MRIDataset
from flyconnectome_mri.models.cnn_baseline import CNNBaseline, count_parameters
from flyconnectome_mri.models.graph_classifier import GraphClassifier, parameter_breakdown
from flyconnectome_mri.training.evaluate import Predictions, collect_diagnostics, predict
from flyconnectome_mri.training.metrics import aggregate_by_patient, binary_metrics
from flyconnectome_mri.training.plots import plot_training_curves
from flyconnectome_mri.training.reproducibility import make_generator, set_seed
from flyconnectome_mri.training.train import compute_pos_weight, fit
from flyconnectome_mri.utils.config import Config
from flyconnectome_mri.utils.provenance import PROJECT_ROOT, sha256_file

CHECKPOINT_NAME = "best_model.pt"


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_graph(config: Config) -> ConnectomeGraph:
    """Load the configured graph with its full node list (node i = i-th sorted node ID)."""
    g = config.graph
    node_ids = load_node_ids(_resolve(g.nodes_file), column=g.node_column)
    graph = load_edge_list(_resolve(g.edges_file), node_ids=node_ids)
    if graph.num_nodes != g.num_nodes:
        raise ValueError(f"{g.edges_file} has {graph.num_nodes} nodes, config says {g.num_nodes}")
    return graph


def _save_node_order(config: Config, graph: ConnectomeGraph, path: Path) -> None:
    """Record which node ID (e.g. neuron bodyId) each node index / feature index refers to."""
    order = pd.DataFrame({"node_index": np.arange(graph.num_nodes), config.graph.node_column: graph.node_ids})
    nodes = pd.read_csv(_resolve(config.graph.nodes_file), dtype={config.graph.node_column: "int64"})
    extra = [c for c in ("rank", "type", "instance") if c in nodes.columns]
    order = order.merge(nodes[[config.graph.node_column, *extra]], on=config.graph.node_column, how="left")
    order.to_csv(path, index=False, lineterminator="\n")


# Thresholds for "collapsed": deliberately loose. They only catch a model whose
# output (or last graph layer) no longer depends on the image at all.
MIN_PROB_STD = 1e-6
MIN_LAYER_STD = 1e-6


def collapse_check(val_probs: np.ndarray, val_auc: float | None, diagnostics: dict[str, float]) -> dict[str, Any]:
    """Decide on VALIDATION data whether the selected model has collapsed.

    Fails if: predictions are (nearly) constant, validation ROC AUC is undefined
    or exactly 0.5, the last graph layer has no image-dependent variation, or all
    of its nodes are dead (0 for every image).
    """
    failures = []
    prob_std = float(np.std(val_probs))
    if prob_std < MIN_PROB_STD or len(np.unique(np.round(val_probs, 10))) < 2:
        failures.append(f"constant predictions (std {prob_std:.2e})")
    if val_auc is None or val_auc == 0.5:
        failures.append(f"validation ROC AUC is {val_auc}")
    layer_keys = sorted(k for k in diagnostics if k.startswith("graph_layer") and k.endswith("_std"))
    if layer_keys:
        last = layer_keys[-1].removesuffix("_std")
        if diagnostics[f"{last}_std"] < MIN_LAYER_STD:
            failures.append(f"{last} has no image-dependent variation")
        if diagnostics[f"{last}_dead_nodes_frac"] >= 1.0:
            failures.append(f"all {last} nodes are dead")
    return {"passed": not failures, "failures": failures, "val_prob_std": prob_std, "val_roc_auc": val_auc}


def model_name(config: Config) -> str:
    return "cnn_baseline" if config.graph.type == "none" else f"graph_{config.graph.type}"


def build_model(config: Config, graph: ConnectomeGraph | None = None) -> nn.Module:
    """CNN baseline if graph.type is "none", otherwise the connectome-constrained classifier."""
    m = config.model
    if config.graph.type == "none":
        return CNNBaseline(1, tuple(m.encoder_channels), m.feature_dim)
    if graph is None:
        raise ValueError("a graph is required for graph models")
    return GraphClassifier(graph.edges, graph.num_nodes, m.graph_layers, 1, tuple(m.encoder_channels), m.feature_dim)


def to_tensor_dataset(dataset: MRIDataset) -> TensorDataset:
    """Preprocess every image once (deterministic) and keep the tensors in memory."""
    images = torch.stack([dataset[i][0] for i in range(len(dataset))])
    labels = torch.tensor([int(label) for label in dataset.records["label_index"]], dtype=torch.long)
    return TensorDataset(images, labels)


def _loader(dataset: MRIDataset, config: Config, shuffle: bool) -> DataLoader:
    data = to_tensor_dataset(dataset) if config.data.cache_in_memory else dataset
    generator = make_generator(config.experiment.seed) if shuffle else None
    # num_workers=0: single process, fully deterministic order on every OS.
    return DataLoader(data, batch_size=config.training.batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def _prediction_table(dataset: MRIDataset, preds: Predictions, threshold: float) -> pd.DataFrame:
    records = dataset.records
    if not np.array_equal(records["label_index"].to_numpy(), preds.labels):
        raise RuntimeError("prediction order does not match the manifest order")
    columns = [c for c in ("image_path", "patient_id", "patient_link_key", "label", "label_index") if c in records]
    table = records[columns].copy()
    table["prob"] = preds.probs
    table["pred"] = (preds.probs >= threshold).astype(int)
    return table


def _patient_table(slices: pd.DataFrame, threshold: float, label_names: dict[int, str]) -> pd.DataFrame:
    patients = aggregate_by_patient(slices)
    patients["label"] = patients["label_index"].map(label_names)
    patients["pred"] = (patients["prob"] >= threshold).astype(int)
    return patients[["patient_id", "label", "label_index", "n_slices", "prob", "pred"]]


def run_training_experiment(config: Config, run_dir: Path, device: torch.device, logger: logging.Logger) -> dict[str, Any]:
    """Train and evaluate; write all outputs to ``run_dir``; return the metrics dict."""
    prepared_dir = PROJECT_ROOT / config.data.prepared_dir
    dataset_meta = json.loads((prepared_dir / METADATA_NAME).read_text(encoding="utf-8"))
    label_names = {int(v): k for k, v in dataset_meta["task"]["classes"].items()}
    threshold = config.training.decision_threshold

    # 1-2. train/val only; class weighting from TRAIN labels only.
    train_set = MRIDataset.from_prepared(prepared_dir, "train")
    val_set = MRIDataset.from_prepared(prepared_dir, "val")
    train_labels = train_set.records["label_index"].to_numpy()
    pos_weight = compute_pos_weight(train_labels) if config.training.class_weighting == "pos_weight_from_train" else 1.0
    logger.info("Train: %d slices / %d patients, Val: %d slices / %d patients",
                len(train_set), train_set.records["patient_id"].nunique(),
                len(val_set), val_set.records["patient_id"].nunique())
    logger.info("pos_weight = #negatives / #positives (train) = %d / %d = %.4f",
                int((train_labels == 0).sum()), int((train_labels == 1).sum()), pos_weight)

    graph = None if config.graph.type == "none" else load_graph(config)
    if graph is not None:
        logger.info("Graph %s: %d nodes, %d edges (%s)", config.graph.type, graph.num_nodes, graph.num_edges,
                    config.graph.edges_file)

    # 3. seed + model (seeded immediately before construction: identical initial weights per seed).
    set_seed(config.experiment.seed, config.experiment.deterministic)
    model = build_model(config, graph).to(device)
    n_params = count_parameters(model)
    logger.info("Model:\n%s", model)
    logger.info("Trainable parameters: %d", n_params)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    train_loader = _loader(train_set, config, shuffle=True)
    val_loader = _loader(val_set, config, shuffle=False)

    # 4. train with validation-based selection.
    checkpoint_path = run_dir / CHECKPOINT_NAME
    result = fit(model, train_loader, val_loader, loss_fn, config.training, device, checkpoint_path)
    pd.DataFrame(result.history).to_csv(run_dir / "history.csv", index=False, lineterminator="\n")
    plot_training_curves(result.history, result.best_epoch, run_dir / "training_curves.png",
                         f"{model_name(config)}, seed {config.experiment.seed}")

    # 5. best checkpoint.
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state"])
    val_preds = predict(model, val_loader, device, loss_fn)
    val_slices = _prediction_table(val_set, val_preds, threshold)
    val_slices.to_csv(run_dir / "val_predictions_slice.csv", index=False, lineterminator="\n")

    # 5b. Validation gate: the test set is only evaluated if the selected model did not collapse.
    val_diagnostics = collect_diagnostics(model, val_loader, device)
    val_slice_metrics = binary_metrics(val_slices["label_index"], val_slices["prob"], threshold)
    gate = collapse_check(val_preds.probs, val_slice_metrics["roc_auc"], val_diagnostics)
    logger.info("Validation diagnostics (best checkpoint): %s",
                ", ".join(f"{k}={v:.4g}" for k, v in val_diagnostics.items()))
    if not gate["passed"]:
        logger.error("Validation gate FAILED (%s). The TEST split is NOT evaluated.", "; ".join(gate["failures"]))
        failed = {
            "model": model_name(config),
            "seed": config.experiment.seed,
            "status": "collapsed_on_validation",
            "validation_gate": gate,
            "validation_diagnostics": val_diagnostics,
            "validation": {"slice": val_slice_metrics},
            "parameter_breakdown": parameter_breakdown(model),
            "training": {"best_epoch": result.best_epoch, "epochs_run": len(result.history),
                         "training_seconds": result.training_seconds},
            "test": None,
        }
        (run_dir / "metrics.json").write_text(json.dumps(failed, indent=2), encoding="utf-8")
        return failed
    logger.info("Validation gate passed: predictions vary, graph layers carry image-dependent signal.")

    # 6. TEST: loaded only now, after training and model selection are finished.
    logger.info("Training finished (best epoch %d). Loading the TEST split for the first time.", result.best_epoch)
    test_set = MRIDataset.from_prepared(prepared_dir, "test")
    test_loader = _loader(test_set, config, shuffle=False)
    test_preds = predict(model, test_loader, device, loss_fn)
    test_slices = _prediction_table(test_set, test_preds, threshold)
    test_patients = _patient_table(test_slices, threshold, label_names)
    val_patients = _patient_table(val_slices, threshold, label_names)
    test_slices.to_csv(run_dir / "test_predictions_slice.csv", index=False, lineterminator="\n")
    test_patients.to_csv(run_dir / "test_predictions_patient.csv", index=False, lineterminator="\n")

    # 7. metrics.
    best_row = result.history[result.best_epoch - 1]
    graph_info = None
    if graph is not None:
        _save_node_order(config, graph, run_dir / "node_order.csv")
        graph_info = {
            "type": config.graph.type,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "edges_file": config.graph.edges_file,
            "edges_sha256": sha256_file(_resolve(config.graph.edges_file)),
            "nodes_file": config.graph.nodes_file,
            "nodes_sha256": sha256_file(_resolve(config.graph.nodes_file)),
            "layers": config.model.graph_layers,
            "edge_weights": "binary mask (biological weights not used); one trainable weight per edge per layer",
        }
    metrics = {
        "model": model_name(config),
        "status": "ok",
        "validation_gate": gate,
        "validation_diagnostics": val_diagnostics,
        "graph": graph_info,
        "parameter_breakdown": parameter_breakdown(model),
        "seed": config.experiment.seed,
        "note": "single-seed pipeline validation run; not evidence of model quality",
        "dataset": {
            "prepared_dir": config.data.prepared_dir,
            "name": dataset_meta["dataset"]["name"],
            "manifest_sha256": sha256_file(prepared_dir / "manifest.csv"),
            "splits": {name: {"slices": len(ds), "patients": int(ds.records["patient_id"].nunique()),
                              "class_counts": ds.class_counts()}
                       for name, ds in (("train", train_set), ("val", val_set), ("test", test_set))},
            "positive_class": dataset_meta["task"]["positive_class"],
        },
        "class_weighting": {"method": config.training.class_weighting, "pos_weight": pos_weight,
                            "computed_from": "train split only"},
        "parameters": n_params,
        "training": {
            "best_epoch": result.best_epoch,
            "epochs_run": len(result.history),
            "stopped_early": result.stopped_early,
            "selection_metric": config.training.selection_metric,
            "best_val_score": result.best_score,
            "training_seconds": result.training_seconds,
            "best_epoch_val": {k: best_row[k] for k in ("val_loss", "val_roc_auc", "val_balanced_accuracy")},
        },
        "decision_threshold": threshold,
        "validation": {
            "slice": binary_metrics(val_slices["label_index"], val_slices["prob"], threshold),
            "patient": binary_metrics(val_patients["label_index"], val_patients["prob"], threshold),
        },
        "test": {
            "slice": binary_metrics(test_slices["label_index"], test_slices["prob"], threshold),
            "patient": binary_metrics(test_patients["label_index"], test_patients["prob"], threshold),
            "loss": test_preds.loss,
        },
        "inference": {
            "test_slices": len(test_set),
            "forward_seconds_total": test_preds.seconds,
            "forward_ms_per_slice": 1000 * test_preds.seconds / len(test_set),
            "batch_size": config.training.batch_size,
            "device": str(device),
            "note": "model forward passes only (preprocessed tensors in memory)",
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
