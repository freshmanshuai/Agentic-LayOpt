import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

from tool.assets.sync_rag_dataset import synchronize_dataset


DEFAULT_SEED = 20260715
PROTOCOL_ID = "engineering-rag-clearance-annotation-v1"
FIELD_SOURCE = "experimental_size_stratified_annotation"
FIELD_CONFIDENCE = 0.5
CLEARANCE_FIELDS = [
    "maintenance_clearance_m",
    "safety_clearance_m",
]
EXCLUDED_SCENE_IDS = {
    "asset/fbx/industrial_kitbash",
    "asset/max/industrial_factory_showroom_machine_015",
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Add deterministic size-stratified experimental clearance annotations."
    )
    parser.add_argument("--dataset-dir", default="rag_dataset")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    result = annotate_clearances(Path(args.dataset_dir), seed=args.seed)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def annotate_clearances(dataset_dir, *, seed=DEFAULT_SEED):
    assets_path = dataset_dir / "model_assets.json"
    assets_jsonl_path = dataset_dir / "model_assets.jsonl"
    chunks_path = dataset_dir / "rag_chunks.jsonl"
    manifest_path = dataset_dir / "build_manifest.json"

    assets = _read_json(assets_path)
    chunks = _read_jsonl(chunks_path)
    eligible_ids = sorted(
        asset["model_asset_id"]
        for asset in assets
        if asset["model_asset_id"] not in EXCLUDED_SCENE_IDS
    )
    rank_by_id = {asset_id: index for index, asset_id in enumerate(eligible_ids)}

    for asset in assets:
        asset_id = asset["model_asset_id"]
        planning = asset.setdefault("planning_fields", {})
        provenance = [
            item
            for item in asset.get("provenance", [])
            if not (
                item.get("source_id") == PROTOCOL_ID
                and item.get("field_path")
                in {f"planning_fields.{field}" for field in CLEARANCE_FIELDS}
            )
        ]
        if asset_id in EXCLUDED_SCENE_IDS:
            assignments = {field: False for field in CLEARANCE_FIELDS}
        else:
            rank = rank_by_id[asset_id]
            assignments = {
                "maintenance_clearance_m": rank % 2 == 0,
                "safety_clearance_m": rank % 4 in {1, 2},
            }

        for field, assigned in assignments.items():
            existing = planning.get(field)
            replaceable = _is_missing(existing) or (
                isinstance(existing, dict)
                and existing.get("source") in {"computed", FIELD_SOURCE}
            )
            if assigned and replaceable:
                planning[field] = _clearance_value(asset, field, seed)
                provenance.append(_provenance(asset_id, field))
            elif not assigned and isinstance(existing, dict) and existing.get("source") == FIELD_SOURCE:
                planning[field] = _empty_clearance()
        asset["provenance"] = provenance

    _write_json(assets_path, assets)
    assets_jsonl_path.write_text(
        "".join(json.dumps(asset, ensure_ascii=False) + "\n" for asset in assets),
        encoding="utf-8",
    )

    assets_by_id = {asset["model_asset_id"]: asset for asset in assets}
    for chunk in chunks:
        asset = assets_by_id[chunk["model_asset_id"]]
        chunk.setdefault("structured_facts", {})["planning_fields"] = copy.deepcopy(
            asset.get("planning_fields", {})
        )
        chunk["provenance"] = copy.deepcopy(asset.get("provenance", []))
        text = re.sub(r"\s+Clearance annotations:.*$", "", chunk.get("retrieval_text", ""))
        maintenance = "available" if not _is_missing(
            asset["planning_fields"]["maintenance_clearance_m"]
        ) else "missing"
        safety = "available" if not _is_missing(
            asset["planning_fields"]["safety_clearance_m"]
        ) else "missing"
        chunk["retrieval_text"] = (
            f"{text} Clearance annotations: maintenance={maintenance}; safety={safety}."
        )
    chunks_path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )

    counts = {
        field: {
            "available": sum(
                not _is_missing(asset["planning_fields"][field]) for asset in assets
            ),
            "missing": sum(
                _is_missing(asset["planning_fields"][field]) for asset in assets
            ),
        }
        for field in CLEARANCE_FIELDS
    }
    manifest = _read_json(manifest_path)
    manifest["clearance_annotation"] = {
        "protocol_id": PROTOCOL_ID,
        "random_seed": seed,
        "assignment_method": "deterministic_asset_rank_with_size_stratified_values",
        "small_asset_range_m": [1.0, 2.0],
        "large_asset_range_m": [3.0, 5.0],
        "large_asset_rule": "max(width_m, depth_m) >= 8 or footprint_area_m2 >= 40",
        "field_confidence": FIELD_CONFIDENCE,
        "excluded_scene_ids": sorted(EXCLUDED_SCENE_IDS),
        "counts": counts,
    }
    _write_json(manifest_path, manifest)

    synchronized = synchronize_dataset(dataset_dir)
    return {
        "status": "OK",
        "asset_count": len(assets),
        "clearance_counts": counts,
        "unresolved_item_count": synchronized["unresolved_item_count"],
    }


def _clearance_value(asset, field, seed):
    dimensions = asset.get("geometry", {}).get("dimensions_m", {})
    width = float(dimensions.get("width_m") or 0.0)
    depth = float(dimensions.get("depth_m") or 0.0)
    large = max(width, depth) >= 8.0 or width * depth >= 40.0
    low, high = (3.0, 5.0) if large else (1.0, 2.0)
    values = {
        side: _stable_random_value(seed, asset["model_asset_id"], field, side, low, high)
        for side in ["front_m", "rear_m", "left_m", "right_m"]
    }
    return {
        **values,
        "top_m": None,
        "bottom_m": None,
        "source": FIELD_SOURCE,
        "confidence": FIELD_CONFIDENCE,
    }


def _stable_random_value(seed, asset_id, field, side, low, high):
    digest = hashlib.sha256(
        f"{seed}:{asset_id}:{field}:{side}".encode("utf-8")
    ).digest()
    fraction = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
    return round(low + fraction * (high - low), 1)


def _provenance(asset_id, field):
    return {
        "field_path": f"planning_fields.{field}",
        "source_type": "experimental_annotation",
        "source_id": PROTOCOL_ID,
        "source_field": "deterministic_size_stratified_assignment",
        "confidence": FIELD_CONFIDENCE,
        "note": (
            f"User-authorized experimental clearance annotation for {asset_id}; "
            "not a manufacturer or regulatory requirement."
        ),
    }


def _empty_clearance():
    return {
        "front_m": None,
        "rear_m": None,
        "left_m": None,
        "right_m": None,
        "top_m": None,
        "bottom_m": None,
        "source": "computed",
        "confidence": None,
    }


def _is_missing(value):
    if value in (None, [], {}):
        return True
    if isinstance(value, dict):
        return not any(
            item is not None
            for key, item in value.items()
            if key not in {"source", "confidence"}
        )
    return False


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
