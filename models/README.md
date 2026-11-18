# Model library

Download the complete 105-model library from [Hugging Face](https://huggingface.co/datasets/RenmingLiu/Agentic-LayOpt-Model-Library). Extract `Agentic-LayOpt-Model-Library-v1.0.zip` into the project root, alongside `README.md` and `tool/`. The archive supplies `models/`, including native source files and all runtime GLBs. The runtime manifest must be at `models/runtime-models/glb_manifest.json`.

Model files use `NNN-english-model-name.extension`, for example:

```text
runtime-models/glb/003-horizontal-centrifugal-pump-set.glb
runtime-models/glb/calibrated/046-tool-cabinet.glb
runtime-models/glb/source_raw/093-fixture-workstation-source.glb
source-models/025-h5-chain-automatic-assembly-line.fbx
source-models/107-assembly-model-asm1.stl
unity_models/093-fixture-workstation.fbx
```

The number comes from the existing model inventory. Keep it when changing a display name or adding another format. Numbering has gaps because the inventory also contains alternate source formats. New models can use the next unused inventory number.

Use lowercase ASCII letters, digits, and hyphens in filenames. Keep model codes and useful variant labels in the name. Model identifiers and engineering annotations retain their existing values. Runtime GLBs use glTF's Y-up convention; native files retain their source coordinates.

The [CSV catalog](catalog.csv) and [JSON catalog](catalog.json) map each number and English name to its stable `model_asset_id` and distributed files. All 105 catalog records have a runtime GLB. File paths in these catalogs are relative to this directory. `runtime_glb_url` identifies the file selected by the workbench.

`source-models/` contains the recovered native files for 25 models, using the same numbered English names. Their GLBs were exported from the original FBX or STL geometry with Blender 4.5.3. FBX axis and declared-unit conversion follows the importer; STL uses the catalog's declared unit scale. Geometry was retained without simplification, and image data is embedded in the GLB. Existing planning geometry and engineering annotations remain separate from source render geometry.

Calibrated and source geometry occupy separate directories. The `-source`, `-raw`, and `-before-...` suffixes distinguish retained variants of the same model. Native FBX files share the model number and English name.

The application and reference annotations continue to bind models by `model_asset_id`. Source-language CAD node labels and historical source references remain in the original model content and provenance records. Public filenames, display names, and retrieval aliases use English.

Conversion and catalog import use `tool/assets/model_naming.py` so regenerated files retain the same names. Add or update a model in `catalog.json` before using a new naming entry. Model terms and attribution remain as described in `../THIRD_PARTY_NOTICES.md`.
