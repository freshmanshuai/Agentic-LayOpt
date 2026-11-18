import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import trimesh
from trimesh.exchange.gltf import export_glb

from tool.assets.model_naming import model_catalog, model_filename


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_STORAGE_ROOT = Path(
    os.environ.get("NLP_MODEL_STORAGE_DIR", REPO_ROOT / "models")
)
DEFAULT_DATASET_DIR = REPO_ROOT / "rag_dataset"
DEFAULT_RUNTIME_DIR = DEFAULT_MODEL_STORAGE_ROOT / "runtime-models"
DEFAULT_SOURCE_ROOT = DEFAULT_MODEL_STORAGE_ROOT / "production-model-library"
FBX_TO_OBJ_HELPER = REPO_ROOT / "tool" / "assets" / "fbx_to_obj.mjs"


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    runtime_dir = Path(args.runtime_dir).resolve()
    output_dir = runtime_dir / "glb"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_dir / "glb_manifest.json"

    assets = json.loads((dataset_dir / "model_assets.json").read_text(encoding="utf-8"))
    assets_by_id = {item["model_asset_id"]: item for item in assets}
    viewport_manifest = json.loads(
        (dataset_dir / "three_viewport_assets.json").read_text(encoding="utf-8")
    )
    obj_manifest_path = runtime_dir / "obj_manifest.json"
    obj_manifest = load_obj_manifest(obj_manifest_path, runtime_dir, Path(args.fbx_source_root))
    existing_manifest = load_existing_manifest(manifest_path)
    existing_by_id = {
        item["model_asset_id"]: item for item in existing_manifest.get("assets", [])
    }

    asset_ids = [item["model_asset_id"] for item in viewport_manifest.get("assets", [])]
    asset_ids.extend(
        sorted(
            model_asset_id
            for model_asset_id in assets_by_id
            if model_asset_id.startswith("asset/fbx/factory_unit_")
        )
    )
    if args.all_assets:
        asset_ids = [
            asset["model_asset_id"]
            for asset in assets
            if has_obj_mapping(asset, obj_manifest.get("items", []))
        ]
    elif args.asset_id:
        asset_ids = args.asset_id
    asset_ids = list(dict.fromkeys(asset_ids))

    converted_by_id = dict(existing_by_id)
    failures = []
    for index, model_asset_id in enumerate(asset_ids, start=1):
        asset = assets_by_id.get(model_asset_id)
        if asset is None:
            failures.append({"model_asset_id": model_asset_id, "error": "asset not found"})
            continue
        try:
            try:
                obj_item = match_obj_item(asset, obj_manifest.get("items", []))
            except ValueError:
                if not args.export_missing_obj:
                    raise
                obj_item = export_missing_obj(asset, runtime_dir, Path(args.fbx_source_root))
                obj_manifest.setdefault("items", []).append(obj_item)
                write_obj_manifest(obj_manifest_path, obj_manifest)
            output_path = output_dir / model_filename(model_asset_id)
            existing = existing_by_id.get(model_asset_id)
            if not args.force and is_current(existing, obj_item, runtime_dir, output_path):
                item = existing
                print(f"[{index}/{len(asset_ids)}] {model_asset_id}: cached", flush=True)
            else:
                item = convert_one(
                    asset,
                    obj_item,
                    runtime_dir,
                    output_path,
                    meshopt=not args.skip_meshopt,
                )
                print(
                    f"[{index}/{len(asset_ids)}] {model_asset_id}: "
                    f"{item['source_obj_bytes']} -> {item['glb_bytes']} bytes",
                    flush=True,
                )
            converted_by_id[model_asset_id] = item
        except Exception as exc:
            failures.append({"model_asset_id": model_asset_id, "error": str(exc)})
            print(f"[{index}/{len(asset_ids)}] {model_asset_id}: failed: {exc}", flush=True)

    payload = {
        "format": "agentic_layopt_glb_manifest_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "assets": [converted_by_id[key] for key in sorted(converted_by_id)],
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}", flush=True)
    if failures:
        raise SystemExit(2)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert selected runtime OBJ assets to GLB.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR))
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--asset-id", action="append", default=[])
    scope.add_argument("--all-assets", action="store_true")
    parser.add_argument("--skip-meshopt", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--export-missing-obj", action="store_true")
    parser.add_argument("--fbx-source-root", default=str(DEFAULT_SOURCE_ROOT))
    return parser.parse_args()


def load_existing_manifest(manifest_path):
    if not manifest_path.exists():
        return {"assets": [], "failures": []}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_obj_manifest(manifest_path, runtime_dir, source_root):
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "format": "agentic_layopt_obj_manifest_v1",
        "source_root": str(source_root),
        "output_root": str(runtime_dir / "obj"),
        "generated_at_unix": int(datetime.now(timezone.utc).timestamp()),
        "items": [],
    }


def write_obj_manifest(manifest_path, manifest):
    manifest["generated_at_unix"] = int(datetime.now(timezone.utc).timestamp())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def has_obj_mapping(asset, obj_items):
    try:
        match_obj_item(asset, obj_items)
    except (KeyError, ValueError):
        return False
    return True


def is_current(existing, obj_item, runtime_dir, output_path):
    if not existing or not output_path.is_file():
        return False
    source_path = resolve_obj_path(obj_item, runtime_dir)
    if not source_path.is_file():
        return False
    return (
        existing.get("source_obj_bytes") == source_path.stat().st_size
        and existing.get("glb_bytes") == output_path.stat().st_size
        and output_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    )


def match_obj_item(asset, obj_items):
    model_files = asset.get("model_files") or []
    if not model_files or not model_files[0].get("file_uri"):
        raise ValueError("asset has no source file URI")

    source_relative = normalize_source_relative(model_files[0]["file_uri"])
    exact = [
        item
        for item in obj_items
        if item.get("status") == "exported"
        and normalize_source_relative(item.get("source_relative_path", "")) == source_relative
    ]
    if len(exact) == 1:
        return exact[0]

    source_stem = PurePosixPath(source_relative).stem.casefold()
    by_stem = [
        item
        for item in obj_items
        if item.get("status") == "exported"
        and PurePosixPath(normalize_source_relative(item.get("source_relative_path", ""))).stem.casefold()
        == source_stem
    ]
    if len(by_stem) == 1:
        return by_stem[0]
    raise ValueError(f"OBJ mapping is not unique for {model_files[0]['file_uri']}")


def normalize_source_relative(value):
    normalized = str(value or "").replace("\\", "/").strip("/")
    prefix = "产线模型库/"
    if normalized.startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized.casefold()


def convert_one(asset, obj_item, runtime_dir, output_path, *, meshopt):
    obj_path = resolve_obj_path(obj_item, runtime_dir)
    if not obj_path.exists():
        raise FileNotFoundError(obj_path)

    scene = trimesh.load_scene(obj_path, process=True)
    if not scene.geometry:
        raise ValueError("OBJ contains no geometry")
    raw_output_path = output_path.with_suffix(".raw.glb") if meshopt else output_path
    raw_output_path.write_bytes(export_glb(scene, include_normals=True))
    if meshopt:
        optimize_meshopt(raw_output_path, output_path)
        raw_output_path.unlink(missing_ok=True)

    bounds = scene.bounds.tolist() if scene.bounds is not None else None
    vertices = sum(len(mesh.vertices) for mesh in scene.geometry.values())
    triangles = sum(len(mesh.faces) for mesh in scene.geometry.values())
    return {
        "model_asset_id": asset["model_asset_id"],
        "asset_name": asset.get("asset_name"),
        "equipment_type": asset.get("equipment_type"),
        "glb_url": "/runtime-models/glb/" + output_path.name,
        "glb_bytes": output_path.stat().st_size,
        "source_obj_url": obj_item.get("obj_url"),
        "source_obj_bytes": obj_path.stat().st_size,
        "vertices": vertices,
        "triangles": triangles,
        "bounds": bounds,
    }


def optimize_meshopt(source_path, output_path):
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        raise RuntimeError("npx is required for Meshopt compression")
    command = [
        npx,
        "--no-install",
        "gltf-transform",
        "meshopt",
        str(source_path),
        str(output_path),
        "--level",
        "medium",
    ]
    subprocess.run(
        subprocess.list2cmdline(command) if os.name == "nt" else command,
        check=True,
        shell=os.name == "nt",
    )


def normalize_source_relative(value):
    normalized = str(value or "").replace("\\", "/").strip("/")
    for prefix in ("产线模型库/", "浜х嚎妯″瀷搴?"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized.casefold()


def export_missing_obj(asset, runtime_dir, source_root):
    source_path = resolve_fbx_source_path(asset, source_root)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not FBX_TO_OBJ_HELPER.is_file():
        raise FileNotFoundError(FBX_TO_OBJ_HELPER)

    slug = asset["model_asset_id"].rsplit("/", 1)[-1]
    source_hash = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:8]
    output_dir = runtime_dir / "obj" / f"{source_hash}_{slug}"
    output_path = output_dir / "model.obj"
    node = shutil.which("node") or shutil.which("node.exe")
    if not node:
        raise RuntimeError("node is required to export FBX assets to OBJ")

    started = datetime.now(timezone.utc)
    result = subprocess.run(
        [node, str(FBX_TO_OBJ_HELPER), str(source_path), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    runtime_s = (datetime.now(timezone.utc) - started).total_seconds()
    stats = json.loads(result.stdout.strip().splitlines()[-1])
    relative_obj = output_path.relative_to(runtime_dir).as_posix()
    return {
        "source_path": str(source_path),
        "source_relative_path": (
            source_path.relative_to(source_root).as_posix()
            if source_path.is_relative_to(source_root)
            else source_path.relative_to(REPO_ROOT).as_posix()
            if source_path.is_relative_to(REPO_ROOT)
            else source_path.name
        ),
        "status": "exported",
        "obj_path": str(output_path),
        "obj_url": "/runtime-models/" + relative_obj,
        "object_count": stats.get("object_count", 0),
        "mesh_object_count": stats.get("mesh_object_count", 0),
        "vertices": stats.get("vertices", 0),
        "faces": stats.get("faces", 0),
        "obj_bytes": output_path.stat().st_size,
        "runtime_s": round(runtime_s, 3),
        "raw_bounds": stats.get("bounds"),
    }


def resolve_fbx_source_path(asset, source_root):
    record = model_catalog().get(asset.get("model_asset_id"), {})
    for relative in record.get("files", []):
        if Path(relative).suffix.lower() != ".fbx":
            continue
        packaged_source = DEFAULT_MODEL_STORAGE_ROOT / relative
        if packaged_source.is_file():
            return packaged_source.resolve()
    model_files = asset.get("model_files") or []
    if not model_files or not model_files[0].get("file_uri"):
        raise ValueError("asset has no source file URI")
    file_uri = str(model_files[0]["file_uri"]).replace("\\", "/").strip()
    candidate = Path(file_uri)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    normalized = file_uri.lstrip("/")
    packaged_source = REPO_ROOT / normalized
    if packaged_source.is_file():
        return packaged_source.resolve()
    if normalized.startswith("产线模型库/"):
        normalized = normalized.removeprefix("产线模型库/")
    candidate = Path(source_root) / normalized
    return candidate.resolve()


def resolve_obj_path(obj_item, runtime_dir):
    obj_url = obj_item.get("obj_url")
    if obj_url and obj_url.startswith("/runtime-models/"):
        return runtime_dir / obj_url.removeprefix("/runtime-models/")
    return Path(obj_item["obj_path"])


if __name__ == "__main__":
    main()
