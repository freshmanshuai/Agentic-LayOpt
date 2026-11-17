import argparse
import json
import re
from datetime import datetime
from pathlib import Path


PLANNING_FIELDS = {
    "planning_fields.ports": "Ports require manual annotation or consistent model node naming conventions.",
    "planning_fields.maintenance_clearance_m": (
        "Maintenance clearance needs a rule library or manual annotation."
    ),
    "planning_fields.safety_clearance_m": (
        "Safety clearance needs a rule library or manual annotation."
    ),
    "planning_fields.process_capabilities": (
        "Process capabilities need authoritative equipment documentation or manual annotation."
    ),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synchronize derived Engineering RAG dataset metadata.")
    parser.add_argument("--dataset-dir", default="rag_dataset")
    args = parser.parse_args(argv)
    result = synchronize_dataset(Path(args.dataset_dir))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def synchronize_dataset(dataset_dir):
    assets_path = dataset_dir / "model_assets.json"
    chunks_path = dataset_dir / "rag_chunks.jsonl"
    unresolved_path = dataset_dir / "unresolved_items.jsonl"
    manifest_path = dataset_dir / "build_manifest.json"
    report_path = dataset_dir / "DATASET_BUILD_REPORT.md"

    assets = _read_json(assets_path)
    chunks = _read_jsonl(chunks_path)
    unresolved = _read_jsonl(unresolved_path)
    asset_ids = [asset["model_asset_id"] for asset in assets]
    chunk_ids = [chunk["model_asset_id"] for chunk in chunks]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("model_assets.json contains duplicate model_asset_id values")
    if set(asset_ids) != set(chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("rag_chunks.jsonl must contain exactly one chunk for every model asset")

    by_item_id = {
        item["item_id"]: item
        for item in unresolved
        if not _is_managed_missing_field(item)
    }
    for asset in assets:
        for field_path, description in PLANNING_FIELDS.items():
            if not _is_missing(_get_path(asset, field_path)):
                continue
            item_id = f"unresolved/{asset['model_asset_id']}/{field_path.replace('.', '_')}"
            by_item_id[item_id] = {
                "item_id": item_id,
                "model_asset_id": asset["model_asset_id"],
                "item_type": "missing_field",
                "field_path": field_path,
                "description": description,
                "severity": "warning",
                "candidates": [],
                "resolution_policy": "manual_review",
            }

    synchronized = sorted(by_item_id.values(), key=lambda item: item["item_id"])
    unresolved_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in synchronized),
        encoding="utf-8",
    )

    manifest = _read_json(manifest_path)
    manifest["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest["asset_count"] = len(assets)
    manifest["rag_chunk_count"] = len(chunks)
    manifest["unresolved_item_count"] = len(synchronized)
    manifest["asset_ids"] = asset_ids
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _update_report(report_path, assets, len(chunks), len(synchronized))
    return {
        "status": "OK",
        "asset_count": len(assets),
        "rag_chunk_count": len(chunks),
        "unresolved_item_count": len(synchronized),
    }


def _update_report(path, assets, chunk_count, unresolved_count):
    text = path.read_text(encoding="utf-8")
    summary = [
        f"- Assets: {len(assets)}",
        f"- RAG chunks: {chunk_count}",
        f"- Unresolved items: {unresolved_count}",
        "",
    ]
    for asset in assets:
        source_format = ((asset.get("model_files") or [{}])[0].get("asset_format") or "unknown")
        status = (asset.get("extraction") or {}).get("extraction_status") or "unknown"
        quality = ((asset.get("extraction") or {}).get("extraction_quality") or {})
        confidence = quality.get("geometry_confidence")
        summary.append(
            f"- `{asset['model_asset_id']}`: {asset.get('asset_name')} | "
            f"format={source_format} | status={status} | geometry_confidence={confidence}"
        )
    replacement = "\n".join(summary).rstrip() + "\n\n"
    text = re.sub(
        r"(?s)(## Asset Summary\s*\n\s*).*?(?=## Next Data Work)",
        r"\1" + replacement,
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def _get_path(value, field_path):
    current = value
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


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


def _is_managed_missing_field(item):
    return (
        item.get("item_type") == "missing_field"
        and item.get("field_path") in PLANNING_FIELDS
        and str(item.get("item_id", "")).startswith("unresolved/")
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    raise SystemExit(main())
