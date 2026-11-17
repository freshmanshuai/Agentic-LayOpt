# Agentic LayOpt

Agentic LayOpt turns natural-language factory-layout requirements into engineering-grounded layout tasks. It combines asset retrieval, layout generation and optimization, explicit editing, constraint validation, recovery, and a Three.js workbench.

| Directory | Contents |
|---|---|
| `tool/` | Implementation organized by service and responsibility |
| `web_frontend/` | Workbench and vendored Three.js runtime |
| `rag_dataset/` | Asset catalog, engineering evidence, schemas, and annotations |
| `models/` | Downloaded 3D model library: runtime GLBs, native FBX/STL files, and model bindings |
| `examples/electronics_workshop/` | The single published, recorded Agentic LayOpt workshop session |

## Downloads

| Resource | Download |
|---|---|
| GenFLP-Bench | [Dataset download page](https://agentic-layopt-datasets.pages.dev/#genflp) |
| LayOpt RAG Benchmark | [Dataset download page](https://agentic-layopt-datasets.pages.dev/#rag) |
| 3D model library | [Hugging Face model library](https://huggingface.co/datasets/RenmingLiu/Agentic-LayOpt-Model-Library) |

The dataset packages contain user tasks, context, and reference annotations. They are distributed separately from this code repository.

### Install the 3D model library

Download [Agentic-LayOpt-Model-Library-v1.0.zip (3.11 GiB)](https://huggingface.co/datasets/RenmingLiu/Agentic-LayOpt-Model-Library/resolve/main/Agentic-LayOpt-Model-Library-v1.0.zip?download=true) from Hugging Face. It contains 105 cataloged models, including 136 GLB files, 39 FBX files, and one STL file.

The archive already contains a top-level `models/` directory. Extract its contents into the project root, alongside `README.md` and `tool/`, preserving the archive's directory structure:

```text
Agentic-LayOpt/
  README.md
  tool/
  rag_dataset/
  models/
    catalog.json
    catalog.csv
    README.md
    runtime-models/
      glb_manifest.json
      glb/
    source-models/
    unity_models/
```

For example, place the downloaded ZIP in the project root and run:

```sh
python -m zipfile -e Agentic-LayOpt-Model-Library-v1.0.zip .
```

After extraction, the runtime manifest is at `models/runtime-models/glb_manifest.json`. The server uses `models/runtime-models/` by default.

Models use a stable three-digit number followed by an English name, such as `003-horizontal-centrifugal-pump-set.glb`. The downloaded `models/catalog.csv` and `models/catalog.json` map names and files to stable `model_asset_id` values; `models/README.md` explains naming and file variants.

## Run

Install the model library as described above before starting the server. Media files tracked by Git LFS use `git lfs install` and `git lfs pull` in an existing checkout.

Run commands from this directory. Install the Python dependencies first:

```sh
python -m pip install -r requirements.txt
```

For natural-language operations, configure `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` together in your shell. Intent understanding and RAG share these settings. The service must support Chat Completions, Bearer authentication, and `response_format={"type":"json_object"}`. `LLM_BASE_URL` is the API prefix, including `/v1` when required; the client appends `/chat/completions`. Native provider APIs with different request formats require an adapter.

Set an available `BGE_M3_MODEL_PATH`, then build the local BGE-M3/Qdrant index:

```sh
python -m tool.rag.engineering_rag index --dataset-dir rag_dataset --index-dir .rag_index/engineering_rag_qdrant
```

After configuring the service and index, start the server:

```sh
python -m tool.api.rough_design_api_server
```

On the same computer, open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to use the workbench. This address refers to your own computer.

`.env.example` is a configuration reference and is not loaded automatically. The server uses the extracted `models/runtime-models/` by default. Server startup does not imply that provider access or retrieval is configured.

To reuse an existing local index, set `ENGINEERING_RAG_INDEX_DIR` to its directory before starting the server. Use the asset catalog and embedding model that were used to build that index. Set `BGE_M3_MODEL_PATH` to the local embedding-model directory to avoid downloading its weights again.

Legacy `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, and `DEEPSEEK_BASE_URL` settings remain supported when the shared `LLM_*` settings are unset. `FLP_INTENT_LLM_MODEL` and `FLP_INTENT_LLM_BASE_URL` can override the intent stage while using the same key. Partial shared settings produce a configuration error rather than inheriting credentials from the legacy provider.

Supply your own credentials through the environment. On Windows, the client also reads user and system environment settings. Keep credentials and personal endpoint settings outside the distributed project; `.env.example` contains no credentials. Model calls go to the configured service; the embedding model, vector index, and 3D assets can run locally.

## Code layout

| Package | Responsibility |
|---|---|
| `tool/api/` | HTTP API, static files, and stream entry points |
| `tool/agents/` | Agent orchestration and agent output contracts |
| `tool/clients/` | LLM client connections and provider errors |
| `tool/context/` | Authoritative context, session memory, changes, and instance names |
| `tool/rag/` | Retrieval, embeddings, vector storage, and read-only proxy |
| `tool/representations/` | Task contracts, FLP-IR, and edit representations |
| `tool/layout/` | Solvers, site constraints, workstation organization, and orientation |
| `tool/logistics/` | Facility endpoints, AGV cost, relocation, and transfer cells |
| `tool/validation/` | Constraint checks, viewport prechecks, and receipts |
| `tool/execution/` | Exact edit execution, validation-driven recovery, and violation resolution |
| `tool/assets/` | Catalog import, annotations, conversion, proxies, and frontend asset vendoring |

GenFLP-Bench and the LayOpt RAG Benchmark are released separately through the [dataset download page](https://agentic-layopt-datasets.pages.dev/). `rag_dataset/` contains the runtime asset catalog and engineering evidence, not the formal RAG test set.

## Recorded workshop example

The [electronic-control equipment workshop](examples/electronics_workshop/README.md) is the only case example. It contains the actual natural-language inputs, DeepSeek intent-understanding records, and four accepted stage results for the 23-object, 20/20/50-batch session. Clarifications, a rejected proposal, restarts, and the final Edit deviation are disclosed rather than hidden.

To view its recorded Repair result without inference credentials:

```sh
python -m tool.api.rough_design_api_server --recording-only --workspace-snapshot examples/electronics_workshop/stages/repair/workspace.json
```

Open [http://127.0.0.1:8000/?restore=captured](http://127.0.0.1:8000/?restore=captured). This is a read-only view of historical results, not a new execution. See the example README for the other stages and for new live requests through the normal workflow.

Unit tests, benchmark ZIP files, other experiments, other case examples, paper materials, and historical Git objects are excluded from this code repository. The selected workshop recording is the explicit exception for case inputs, conversations, and results.

Project code uses the [MIT license](LICENSE). Externally authored software and models retain their source terms; see [third-party notices](THIRD_PARTY_NOTICES.md).
