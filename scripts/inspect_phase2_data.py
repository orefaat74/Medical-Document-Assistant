"""Inspect the approved Phase 1 corpus before fixing Phase 2 evaluation groups."""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from medical_doc_assistant.storage import file_sha256, write_json


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset = root / "data" / "processed" / "medicine_labels.parquet"
    cleaning = json.loads((root / "reports" / "cleaning_summary.json").read_text(encoding="utf-8"))
    checksum = file_sha256(dataset)
    if checksum != cleaning["outputs"][dataset.name]["sha256"]:
        raise ValueError("The Phase 1 dataset differs from its cleaning manifest.")
    frame = pd.read_parquet(dataset)
    output = root / "reports" / "phase2"
    output.mkdir(parents=True, exist_ok=True)
    groups = pd.crosstab(frame["leakage_group_id"], frame["class_label"])
    groups["documents"] = groups.sum(axis=1)
    groups.sort_values("documents", ascending=False).to_csv(output / "phase1_group_class_counts.csv")
    names = defaultdict(lambda: {"documents": 0, "groups": set(), "classes": set()})
    for row in frame[["substance_names", "leakage_group_id", "class_label"]].itertuples(index=False):
        for name in json.loads(row.substance_names):
            names[name]["documents"] += 1
            names[name]["groups"].add(row.leakage_group_id)
            names[name]["classes"].add(row.class_label)
    inventory = pd.DataFrame([
        {"substance_name": name, "documents": values["documents"], "phase1_groups": len(values["groups"]),
         "class_labels": ", ".join(sorted(values["classes"]))}
        for name, values in sorted(names.items())
    ])
    inventory.to_csv(output / "ingredient_inventory.csv", index=False)
    summary = {
        "dataset_sha256": checksum,
        "documents": len(frame),
        "phase1_groups": len(groups),
        "substance_names": len(names),
        "class_documents": frame["class_label"].value_counts().sort_index().to_dict(),
        "class_group_counts": (groups.drop(columns="documents") > 0).sum().to_dict(),
        "largest_group": groups.sort_values("documents", ascending=False).head(1).reset_index().to_dict("records"),
    }
    write_json(output / "input_inventory.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
