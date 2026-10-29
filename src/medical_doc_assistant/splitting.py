"""Freeze ingredient- and duplicate-aware partitions before model selection."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from medical_doc_assistant.rxnorm import canonical_ingredient_tokens, collect_ingredient_reference
from medical_doc_assistant.storage import file_sha256, utc_now, write_json

LOGGER = logging.getLogger(__name__)
SPLIT_CONFIG_KEYS = (
    "random_state", "split_fractions", "cv_folds", "maximum_minimum_holdout_groups_per_class",
    "split_solver_time_limit_seconds", "class_group_balance_weight",
    "minimum_holdout_size_multiplier", "maximum_holdout_size_multiplier", "near_duplicate_audit",
)


def config_digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def load_phase2_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    fractions = config["split_fractions"]
    if config.get("schema_version") != 1 or set(fractions) != {"train", "validation", "test"}:
        raise ValueError("Invalid Phase 2 configuration schema or partition names.")
    if any(value <= 0 for value in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("Train/validation/test fractions must be positive and sum to one.")
    if config["cv_folds"] < 2 or config["maximum_minimum_holdout_groups_per_class"] < 1:
        raise ValueError("At least two CV folds and one holdout group per class are required.")
    if not 0 < config["near_duplicate_audit"]["similarity_threshold"] <= 1:
        raise ValueError("Near-duplicate similarity must lie in (0, 1].")
    if set(config["parameter_grids"]) != {"logistic_regression", "linear_svm", "random_forest"}:
        raise ValueError("Define the three planned baseline classifier families.")
    return config


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def build_benchmark_groups(
    frame: pd.DataFrame, reference: dict[str, Any], config: dict[str, Any]
) -> tuple[list[str], pd.DataFrame, dict[str, Any]]:
    rows = frame[["record_id", "class_label", "leakage_group_id", "substance_names"]].to_dict("records")
    components = DisjointSet(len(rows))
    owners: dict[str, int] = {}
    for index, row in enumerate(rows):
        tokens = canonical_ingredient_tokens(row, reference) | {"phase1:" + row["leakage_group_id"]}
        for token in sorted(tokens):
            if token in owners:
                components.union(index, owners[token])
            else:
                owners[token] = index
    rxnorm_roots = np.array([components.find(index) for index in range(len(rows))])
    audit = {
        "phase1_groups": int(frame["leakage_group_id"].nunique()),
        "groups_after_rxnorm": len(set(rxnorm_roots)),
        "rxnorm_version": reference["rxnorm_version"],
        "rxnorm_names_matched": sum(entry["status"] == "matched" for entry in reference["entries"].values()),
        "rxnorm_names_total": len(reference["entries"]),
    }
    near = config["near_duplicate_audit"]
    # Fixed hashing has no learned vocabulary/IDF. It is used only to identify evaluation groups.
    hasher = HashingVectorizer(
        ngram_range=(near["ngram_size"], near["ngram_size"]), n_features=near["hash_features"],
        alternate_sign=False, binary=True, norm="l2", dtype=np.float32,
    )
    features = hasher.transform(frame["document_text"])
    word_counts = frame["word_count"].to_numpy()
    pairs = []
    for start in range(0, len(frame), near["chunk_size"]):
        stop = min(len(frame), start + near["chunk_size"])
        similarities = cosine_similarity(features[start:stop], features)
        left, right = np.where(similarities >= near["similarity_threshold"])
        for local_left, right_index in zip(left.tolist(), right.tolist()):
            left_index = start + local_left
            if left_index >= right_index or rxnorm_roots[left_index] == rxnorm_roots[right_index]:
                continue
            length_ratio = min(word_counts[left_index], word_counts[right_index]) / max(word_counts[left_index], word_counts[right_index])
            if length_ratio < near["minimum_word_count_ratio"]:
                continue
            components.union(left_index, right_index)
            pairs.append({
                "left_record_id": rows[left_index]["record_id"], "right_record_id": rows[right_index]["record_id"],
                "left_class": rows[left_index]["class_label"], "right_class": rows[right_index]["class_label"],
                "similarity": float(similarities[local_left, right_index]), "word_count_ratio": float(length_ratio),
            })
        LOGGER.info("Near-duplicate audit: %s / %s documents", stop, len(frame))
    members: dict[int, list[str]] = {}
    for index, row in enumerate(rows):
        members.setdefault(components.find(index), []).append(row["record_id"])
    identifiers = {
        component: "benchmark_" + hashlib.sha256("|".join(sorted(ids)).encode("utf-8")).hexdigest()[:16]
        for component, ids in members.items()
    }
    group_ids = [identifiers[components.find(index)] for index in range(len(rows))]
    audit.update(benchmark_groups=len(identifiers), near_duplicate_pairs=len(pairs),
                 similarity_threshold=near["similarity_threshold"])
    pair_frame = pd.DataFrame(pairs, columns=[
        "left_record_id", "right_record_id", "left_class", "right_class", "similarity", "word_count_ratio",
    ])
    return group_ids, pair_frame, audit


def allocate_groups(
    counts: pd.DataFrame,
    fractions: dict[str, float],
    minimum_groups: dict[str, dict[str, int]],
    config: dict[str, Any],
    *,
    seed: int,
    holdout_bounds: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Minimize class-size/group-size deviations subject to whole-group coverage."""
    partitions = list(fractions)
    group_count, partition_count = len(counts), len(partitions)
    binary_count = group_count * partition_count
    rng = np.random.default_rng(seed)
    objective = list(rng.uniform(0, 1e-8 / max(1, binary_count), binary_count))
    constraints: list[tuple[dict[int, float], float, float]] = []

    def add_constraint(coefficients: dict[int, float], low: float, high: float) -> None:
        constraints.append((coefficients, low, high))

    def absolute_deviation(weights: np.ndarray, partition: int, target: float, penalty: float) -> None:
        slack = len(objective)
        objective.append(penalty)
        coefficients = {group * partition_count + partition: float(weight)
                        for group, weight in enumerate(weights) if weight}
        add_constraint({**coefficients, slack: -1.0}, -np.inf, target)
        add_constraint({**{index: -value for index, value in coefficients.items()}, slack: -1.0}, -np.inf, -target)

    documents = counts.sum(axis=1).to_numpy(dtype=float)
    for group in range(group_count):
        add_constraint({group * partition_count + partition: 1.0 for partition in range(partition_count)}, 1, 1)
    for partition, name in enumerate(partitions):
        fraction = fractions[name]
        for label in counts.columns:
            class_documents = counts[label].to_numpy(dtype=float)
            presence = (class_documents > 0).astype(float)
            minimum = minimum_groups[name][label]
            if presence.sum() < minimum:
                raise ValueError(f"Insufficient groups for {label} in {name}.")
            add_constraint({group * partition_count + partition: float(value)
                            for group, value in enumerate(presence) if value}, minimum, np.inf)
            absolute_deviation(class_documents, partition, fraction * class_documents.sum(),
                               1.0 / (len(counts.columns) * class_documents.sum()))
            absolute_deviation(presence, partition, fraction * presence.sum(),
                               config["class_group_balance_weight"] / (len(counts.columns) * presence.sum()))
        absolute_deviation(documents, partition, fraction * documents.sum(), 0.25 / documents.sum())
        absolute_deviation(np.ones(group_count), partition, fraction * group_count, 0.1 / group_count)
        if holdout_bounds and name != "train":
            add_constraint({group * partition_count + partition: float(value) for group, value in enumerate(documents)},
                           fraction * documents.sum() * config["minimum_holdout_size_multiplier"],
                           fraction * documents.sum() * config["maximum_holdout_size_multiplier"])

    matrix_rows, matrix_columns, matrix_values = [], [], []
    low, high = [], []
    for row_index, (coefficients, lower, upper) in enumerate(constraints):
        for column_index, value in coefficients.items():
            matrix_rows.append(row_index)
            matrix_columns.append(column_index)
            matrix_values.append(value)
        low.append(lower)
        high.append(upper)
    matrix = coo_matrix((matrix_values, (matrix_rows, matrix_columns)), shape=(len(constraints), len(objective))).tocsc()
    upper_bounds = np.full(len(objective), np.inf)
    upper_bounds[:binary_count] = 1
    integrality = np.zeros(len(objective), dtype=int)
    integrality[:binary_count] = 1
    result = milp(
        np.asarray(objective), integrality=integrality, bounds=Bounds(np.zeros(len(objective)), upper_bounds),
        constraints=LinearConstraint(matrix, np.asarray(low), np.asarray(high)),
        options={"time_limit": config["split_solver_time_limit_seconds"], "mip_rel_gap": 0.01},
    )
    if result.x is None:
        raise ValueError("No group-separated allocation met the class-coverage constraints: " + result.message)
    chosen = result.x[:binary_count].reshape(group_count, partition_count)
    rounded = np.rint(chosen)
    if not np.allclose(chosen, rounded, atol=1e-6) or not np.all(rounded.sum(axis=1) == 1):
        raise ValueError("The split solver did not return a whole-group allocation.")
    assignment = {group: partitions[int(np.argmax(row))] for group, row in zip(counts.index, rounded)}
    for name in partitions:
        selected = counts.loc[[group for group in counts.index if assignment[group] == name]]
        for label in counts.columns:
            if (selected[label] > 0).sum() < minimum_groups[name][label]:
                raise ValueError("The split solver returned insufficient class coverage.")
    return assignment, {
        "method": "SciPy/HiGHS mixed-integer whole-group allocation",
        "random_state": seed, "status": int(result.status), "message": result.message,
        "objective": float(result.fun), "mip_gap": float(result.mip_gap),
        "time_limit_seconds": config["split_solver_time_limit_seconds"],
    }


def create_partitions(frame: pd.DataFrame, config: dict[str, Any], labels: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = pd.crosstab(frame["benchmark_group_id"], frame["class_label"]).reindex(columns=labels, fill_value=0)
    class_groups = (counts > 0).sum()
    minimum: dict[str, dict[str, int]] = {name: {} for name in config["split_fractions"]}
    for label in labels:
        total = int(class_groups[label])
        holdout = min(config["maximum_minimum_holdout_groups_per_class"], (total - config["cv_folds"]) // 2)
        if holdout < 1:
            raise ValueError(f"{label} needs at least {config['cv_folds'] + 2} independent groups for this evaluation protocol.")
        minimum["train"][label] = max(config["cv_folds"], min(total // 2, total - 2 * holdout))
        minimum["validation"][label] = minimum["test"][label] = holdout
    assignment, split_solver = allocate_groups(counts, config["split_fractions"], minimum, config,
                                               seed=config["random_state"], holdout_bounds=True)
    columns = ["record_id", "class_label", "document_hash", "leakage_group_id", "benchmark_group_id"]
    partitions = frame[columns].copy()
    partitions["split"] = partitions["benchmark_group_id"].map(assignment)
    train_counts = counts.loc[[group for group in counts.index if assignment[group] == "train"]]
    fold_names = [f"fold_{index}" for index in range(config["cv_folds"])]
    fold_minimum = {name: {label: 1 for label in labels} for name in fold_names}
    folds, cv_solver = allocate_groups(train_counts, {name: 1 / len(fold_names) for name in fold_names},
                                      fold_minimum, config, seed=config["random_state"] + 1)
    partitions["cv_fold"] = partitions["benchmark_group_id"].map(
        {group: int(name.split("_")[1]) for group, name in folds.items()}
    ).fillna(-1).astype(int)
    return partitions, {"minimum_groups_per_class": minimum, "split_solver": split_solver, "cv_solver": cv_solver}


def prepare_phase2(root: Path, config: dict[str, Any], labels: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = root / "data" / "processed" / "medicine_labels.parquet"
    checksum = file_sha256(dataset)
    cleaning = json.loads((root / "reports" / "cleaning_summary.json").read_text(encoding="utf-8"))
    if checksum != cleaning["outputs"][dataset.name]["sha256"]:
        raise ValueError("The approved Phase 1 dataset differs from its cleaning manifest.")
    frame = pd.read_parquet(dataset).reset_index(drop=True)
    if set(frame["class_label"]) != set(labels):
        raise ValueError("The Phase 1 corpus does not match the configured classification targets.")
    report_dir = root / "reports" / "phase2"
    split_dir = root / "data" / "splits"
    split_path = split_dir / "phase2_split.csv"
    manifest_path = report_dir / "split_manifest.json"
    settings = {key: config[key] for key in SPLIT_CONFIG_KEYS}
    reference_path = root / "data" / "reference" / "rxnorm_ingredient_map.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["dataset_sha256"] != checksum or manifest["split_config"] != settings:
            raise ValueError("The frozen Phase 2 split uses different data/settings. Preserve it before starting a new experiment.")
        if file_sha256(split_path) != manifest["split_sha256"] or file_sha256(reference_path) != manifest["reference_sha256"]:
            raise ValueError("The frozen split or ingredient-reference checksum does not match its manifest.")
        assignments = pd.read_csv(split_path)
        LOGGER.info("Using frozen Phase 2 split: %s documents, %s groups", len(assignments), manifest["grouping_audit"]["benchmark_groups"])
    else:
        reference = collect_ingredient_reference(frame, reference_path, config["rxnorm_workers"])
        group_ids, pairs, audit = build_benchmark_groups(frame, reference, config)
        frame["benchmark_group_id"] = group_ids
        assignments, split_details = create_partitions(frame, config, labels)
        report_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(split_path, index=False)
        pairs.to_csv(report_dir / "near_duplicate_pairs.csv", index=False)
        pd.DataFrame([
            {"source_name": name, "status": entry["status"], "ingredient_rxcui": entry.get("ingredient_rxcui", ""),
             "ingredient_name": entry.get("ingredient_name", "")}
            for name, entry in reference["entries"].items()
        ]).to_csv(report_dir / "rxnorm_matches.csv", index=False)
        balance = assignments.groupby(["split", "class_label"]).agg(
            documents=("record_id", "size"), groups=("benchmark_group_id", "nunique")
        ).reset_index()
        balance.to_csv(report_dir / "split_class_balance.csv", index=False)
        cv_balance = []
        training = assignments.loc[assignments["split"] == "train"]
        for fold in range(config["cv_folds"]):
            for label in labels:
                category = training.loc[training["class_label"] == label]
                for subset, mask in (("fit", category["cv_fold"] != fold), ("validation", category["cv_fold"] == fold)):
                    cv_balance.append({"fold": fold, "subset": subset, "class_label": label,
                                       "documents": int(mask.sum()), "groups": int(category.loc[mask, "benchmark_group_id"].nunique())})
        pd.DataFrame(cv_balance).to_csv(report_dir / "cv_class_balance.csv", index=False)
        manifest = {
            "created_at_utc": utc_now(), "schema_version": 1, "dataset_sha256": checksum,
            "documents": len(frame), "class_labels": labels, "split_config": settings,
            "split_sha256": file_sha256(split_path), "reference_sha256": file_sha256(reference_path),
            "near_duplicate_pairs_sha256": file_sha256(report_dir / "near_duplicate_pairs.csv"),
            "grouping_audit": audit, "split_balance": balance.to_dict("records"), **split_details,
        }
        write_json(manifest_path, manifest)
        LOGGER.info("Frozen Phase 2 split: %s documents, %s benchmark groups", len(frame), audit["benchmark_groups"])
    frame = frame.drop(columns=["benchmark_group_id"], errors="ignore").merge(
        assignments[["record_id", "benchmark_group_id", "split", "cv_fold"]], on="record_id", validate="one_to_one"
    )
    if len(frame) != manifest["documents"]:
        raise ValueError("The saved split does not cover every corpus document.")
    return frame, manifest


def training_cv_indices(frame: pd.DataFrame, folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if not frame["split"].eq("train").all():
        raise ValueError("Hyperparameter cross-validation accepts only the training partition.")
    return [(np.flatnonzero(frame["cv_fold"].to_numpy() != fold),
             np.flatnonzero(frame["cv_fold"].to_numpy() == fold)) for fold in range(folds)]
