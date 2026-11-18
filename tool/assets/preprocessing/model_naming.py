"""Use the public model index for English labels and stable file names."""
from functools import lru_cache
import json
from pathlib import Path
import re


CATALOG_PATH = Path(__file__).resolve().parents[3] / "models" / "catalog.json"
HAN = re.compile(r"[\u3400-\u9fff]")


@lru_cache(maxsize=1)
def model_catalog():
    if not CATALOG_PATH.is_file():
        return {}
    records = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["models"]
    return {record["model_asset_id"]: record for record in records}


def model_filename(model_asset_id, extension=".glb"):
    record = model_catalog().get(model_asset_id)
    stem = record["filename_stem"] if record else model_asset_id.rsplit("/", 1)[-1]
    return stem + extension


def normalize_asset_names(assets):
    changed = set()
    for asset in assets:
        record = model_catalog().get(asset["model_asset_id"])
        if record is None:
            continue
        aliases = list(dict.fromkeys([record["name"], *(
            alias for alias in asset.get("aliases", [])
            if isinstance(alias, str) and not HAN.search(alias)
        )]))
        if asset.get("asset_name") != record["name"] or asset.get("aliases") != aliases:
            changed.add(asset["model_asset_id"])
            asset["asset_name"] = record["name"]
            asset["aliases"] = aliases
    return changed
