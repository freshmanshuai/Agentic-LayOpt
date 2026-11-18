"""Build auditable low-poly rack proxies and the V2 viewport asset manifest.

The source rack meshes are too large for repeatable browser experiments and
their catalog bounds are not usable as metre-scale planning dimensions.  This
script keeps the source asset IDs, records that the generated GLBs are render
proxies, and never overwrites the source asset catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import trimesh

from tool.assets.model_naming import model_filename


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME_ROOT = REPO_ROOT / "models" / "runtime-models"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "rag_dataset" / "three_viewport_assets_v2.json"

SINGLE_RACK_ID = "asset/fbx/fbx_be827a77"
FIVE_RACK_ID = "asset/fbx/5"
RACK_DIMENSIONS_M: Mapping[str, Tuple[float, float, float]] = {
    SINGLE_RACK_ID: (0.6, 1.2, 2.0),
    FIVE_RACK_ID: (3.0, 1.2, 2.0),
}
RACK_FILENAMES = {
    SINGLE_RACK_ID: model_filename(SINGLE_RACK_ID),
    FIVE_RACK_ID: model_filename(FIVE_RACK_ID),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _box(
    scene: trimesh.Scene,
    *,
    extents: Tuple[float, float, float],
    center: Tuple[float, float, float],
    color: Tuple[int, int, int, int],
    name: str,
) -> None:
    mesh = trimesh.creation.box(extents=extents)
    mesh.visual.face_colors = np.asarray(color, dtype=np.uint8)
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(center, dtype=float)
    scene.add_geometry(mesh, node_name=name, geom_name=name, transform=transform)


def _add_single_rack(scene: trimesh.Scene, x_offset: float, prefix: str) -> None:
    # Coordinates use X width, Y height and Z depth, matching the viewport.
    _box(
        scene,
        extents=(0.58, 1.98, 1.18),
        center=(x_offset, 0.99, 0.0),
        color=(38, 48, 60, 255),
        name=f"{prefix}_body",
    )
    _box(
        scene,
        extents=(0.54, 1.88, 0.025),
        center=(x_offset, 0.99, 0.6025),
        color=(20, 27, 35, 255),
        name=f"{prefix}_front",
    )
    for slot in range(10):
        _box(
            scene,
            extents=(0.48, 0.115, 0.018),
            center=(x_offset, 0.19 + slot * 0.165, 0.625),
            color=(43, 105, 138, 255) if slot % 2 else (55, 133, 93, 255),
            name=f"{prefix}_slot_{slot:02d}",
        )
    _box(
        scene,
        extents=(0.50, 0.045, 0.02),
        center=(x_offset, 1.86, 0.626),
        color=(213, 178, 67, 255),
        name=f"{prefix}_status",
    )


def _build_proxy(path: Path, rack_count: int) -> None:
    scene = trimesh.Scene()
    offsets = [(index - (rack_count - 1) / 2.0) * 0.6 for index in range(rack_count)]
    for index, offset in enumerate(offsets):
        _add_single_rack(scene, offset, f"rack_{index + 1:02d}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = scene.export(file_type="glb")
    if not isinstance(payload, (bytes, bytearray)):
        raise RuntimeError("trimesh did not return a binary GLB payload")
    path.write_bytes(bytes(payload))


def _catalog_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(catalog, list):
        raise ValueError(f"Asset catalog must be an array: {path}")
    return {
        str(item["model_asset_id"]): item
        for item in catalog
        if isinstance(item, dict) and item.get("model_asset_id")
    }


def _proxy_entry(
    asset_id: str,
    catalog: Mapping[str, Mapping[str, Any]],
    glb_path: Path,
) -> Dict[str, Any]:
    width_m, depth_m, height_m = RACK_DIMENSIONS_M[asset_id]
    metadata = catalog.get(asset_id, {})
    return {
        "model_asset_id": asset_id,
        "asset_name": metadata.get("asset_name") or metadata.get("name") or asset_id,
        "equipment_type": metadata.get("equipment_type") or "server_rack",
        "glb_url": f"/runtime-models/glb/{RACK_FILENAMES[asset_id]}",
        "glb_bytes": glb_path.stat().st_size,
        "glb_sha256": _sha256(glb_path),
        "rendering_role": "project_low_poly_proxy",
        "planning_dimensions_m": {
            "width": width_m,
            "depth": depth_m,
            "height": height_m,
        },
        "dimension_provenance": {
            "type": "project_manual_proxy_calibration",
            "status": "pending_human_engineering_review",
            "note": "Used only for rough-layout planning; source catalog bounds are not treated as physical dimensions.",
        },
        "source_asset_preserved": True,
    }


def _runtime_path(runtime_root: Path, glb_url: str) -> Path:
    prefix = "/runtime-models/"
    if not glb_url.startswith(prefix):
        raise ValueError(f"Unsupported GLB URL: {glb_url}")
    return runtime_root / glb_url[len(prefix) :]


def _canonical_entries(
    assets: Iterable[Mapping[str, Any]], runtime_root: Path
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for source in assets:
        asset_id = str(source.get("model_asset_id") or "")
        glb_url = str(source.get("glb_url") or "")
        if not asset_id or not glb_url:
            raise ValueError("Every viewport asset requires model_asset_id and glb_url")
        path = _runtime_path(runtime_root, glb_url)
        if not path.is_file():
            raise FileNotFoundError(path)
        entry = dict(source)
        entry["glb_bytes"] = path.stat().st_size
        entry["glb_sha256"] = _sha256(path)
        entry["runtime_relative_path"] = glb_url.removeprefix("/runtime-models/")
        entry.setdefault("rendering_role", "source_derived_runtime_model")
        entries.append(entry)
    entries.sort(key=lambda item: str(item["model_asset_id"]))
    if len({entry["model_asset_id"] for entry in entries}) != len(entries):
        raise ValueError("Viewport manifest contains duplicate model_asset_id values")
    return entries


def build(runtime_root: Path, output_manifest: Path) -> Dict[str, Any]:
    external_manifest_path = runtime_root / "glb_manifest.json"
    external = json.loads(external_manifest_path.read_text(encoding="utf-8"))
    assets = external.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError(f"Invalid runtime manifest: {external_manifest_path}")
    catalog = _catalog_by_id(REPO_ROOT / "rag_dataset" / "model_assets.json")
    missing_racks = sorted(set(RACK_DIMENSIONS_M) - set(catalog))
    if missing_racks:
        raise ValueError(f"Rack IDs missing from source catalog: {missing_racks}")

    rack_paths = {
        SINGLE_RACK_ID: runtime_root / "glb" / RACK_FILENAMES[SINGLE_RACK_ID],
        FIVE_RACK_ID: runtime_root / "glb" / RACK_FILENAMES[FIVE_RACK_ID],
    }
    _build_proxy(rack_paths[SINGLE_RACK_ID], rack_count=1)
    _build_proxy(rack_paths[FIVE_RACK_ID], rack_count=5)

    retained = [
        item
        for item in assets
        if isinstance(item, dict) and item.get("model_asset_id") not in RACK_DIMENSIONS_M
    ]
    retained.extend(
        _proxy_entry(asset_id, catalog, rack_paths[asset_id])
        for asset_id in (SINGLE_RACK_ID, FIVE_RACK_ID)
    )
    retained.sort(key=lambda item: str(item.get("model_asset_id") or ""))
    external["assets"] = retained
    external["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(external_manifest_path, external)

    canonical_assets = _canonical_entries(retained, runtime_root)
    manifest: Dict[str, Any] = {
        "schema_version": "three_viewport_assets_v2.1",
        "status": "CANDIDATE_PENDING_USER_REVIEW",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_url_root": "/runtime-models/",
        "source_runtime_manifest": str(external_manifest_path),
        "source_runtime_manifest_sha256": _sha256(external_manifest_path),
        "asset_count": len(canonical_assets),
        "assets": canonical_assets,
    }
    manifest["asset_set_sha256"] = _canonical_hash(canonical_assets)
    _write_json(output_manifest, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    args = parser.parse_args()
    manifest = build(args.runtime_root.resolve(), args.output_manifest.resolve())
    print(
        json.dumps(
            {
                "status": "OK",
                "asset_count": manifest["asset_count"],
                "asset_set_sha256": manifest["asset_set_sha256"],
                "output_manifest": str(args.output_manifest.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
