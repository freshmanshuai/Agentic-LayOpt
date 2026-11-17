import argparse
import copy
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from tool.clients.llm_clients import IntentLLMClient
from tool.rag.engineering_rag import EngineeringRAGService
from tool.layout.layout_solvers import DEFAULT_SOLVER_NAME, solver_catalog
from tool.agents.rough_design_agents import AgenticLayOptOrchestrator, RoughDesignOrchestrator
from tool.validation.viewport_receipts import ViewportReceiptError


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_WEB_DIR = Path(__file__).resolve().parents[2] / "web_frontend"
DEFAULT_MODEL_STORAGE_ROOT = Path(
    os.environ.get(
        "NLP_MODEL_STORAGE_DIR",
        Path(__file__).resolve().parents[2] / "models",
    )
)
DEFAULT_RUNTIME_MODELS_DIR = DEFAULT_MODEL_STORAGE_ROOT / "runtime-models"


def make_handler(
    dataset_dir,
    web_dir=None,
    runtime_models_dir=None,
    rag_index_dir=None,
    *,
    rag_service=None,
    intent_llm_client=None,
    workspace_snapshot_path=None,
    initial_sessions=None,
    recording_only=False,
):
    dataset_dir = Path(dataset_dir)
    web_dir = Path(web_dir or DEFAULT_WEB_DIR).resolve()
    runtime_models_dir = Path(runtime_models_dir or DEFAULT_RUNTIME_MODELS_DIR).resolve()
    workspace_snapshot_path = (
        Path(workspace_snapshot_path).resolve()
        if workspace_snapshot_path
        else None
    )
    orchestrator = None
    stream_orchestrator = None
    if not recording_only:
        if rag_service is None:
            rag_service = EngineeringRAGService(
                dataset_dir,
                index_dir=rag_index_dir or Path(".rag_index") / "engineering_rag_qdrant",
            )
        if intent_llm_client is None:
            intent_llm_client = _make_intent_llm_client()
        orchestrator = RoughDesignOrchestrator(
            dataset_dir,
            rag_service=rag_service,
            intent_llm_client=intent_llm_client,
        )
        stream_orchestrator = AgenticLayOptOrchestrator(
            dataset_dir,
            rag_service=rag_service,
            intent_llm_client=intent_llm_client,
        )
        stream_orchestrator.sessions.update(copy.deepcopy(initial_sessions or {}))
    def load_workspace_snapshot():
        """Load an explicitly configured captured normal-workbench state."""

        if workspace_snapshot_path is None or not workspace_snapshot_path.is_file():
            return None
        try:
            payload = json.loads(workspace_snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    class RoughDesignApiHandler(BaseHTTPRequestHandler):
        server_version = "RoughDesignApi/0.1"

        def do_OPTIONS(self):
            self._send_json({"status": "OK"})

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json({"status": "OK", "service": "rough_design_api", "recording_only": recording_only})
                return
            if path == "/api/viewport/config":
                self._send_json(_viewport_config(runtime_models_dir))
                return
            if path == "/api/solver/config":
                self._send_json(solver_catalog())
                return
            if path == "/api/workspace-snapshot":
                snapshot = load_workspace_snapshot()
                if snapshot is None:
                    self._send_json(
                        {"status": "NOT_FOUND", "error": "No captured workspace is available."},
                        status_code=404,
                    )
                else:
                    self._send_json(snapshot)
                return
            if path == "/":
                self._serve_file(web_dir / "index.html")
                return
            if path.startswith("/web/"):
                self._serve_static(web_dir, path.removeprefix("/web/"))
                return
            if path.startswith("/runtime-models/"):
                self._serve_static(runtime_models_dir, path.removeprefix("/runtime-models/"))
                return
            self._send_json({"status": "NOT_FOUND", "path": path}, status_code=404)

        def do_POST(self):
            path = urlparse(self.path).path
            if recording_only:
                self._send_json(
                    {"status": "READ_ONLY_RECORDING", "error": "This server displays recorded results. Start the normal server for a new run."},
                    status_code=405,
                )
                return
            if path == "/api/agentic-layopt/stream":
                self._handle_agentic_stream()
                return
            if path == "/api/agentic-layopt/viewport-ack":
                self._handle_viewport_ack()
                return
            if path != "/api/rough-design/run":
                self._send_json({"status": "NOT_FOUND", "path": path}, status_code=404)
                return

            try:
                request = self._read_json()
                pipeline_message = _build_pipeline_message(request)
                scene_context = _scene_context_with_solver(request)
                active_layout_id = scene_context.get("active_layout_id") or None
                result = orchestrator.run(
                    pipeline_message,
                    active_layout_id=active_layout_id,
                    scene_context=scene_context,
                )
                response = {
                    "layout_response": _to_layout_response(result),
                    "backend_request": {
                        "raw_user_message": request.get("raw_user_message", ""),
                        "pipeline_message": pipeline_message,
                        "solver_requested": scene_context.get("solver") or DEFAULT_SOLVER_NAME,
                        "solver_selection_source": scene_context.get("solver_selection_source") or "default",
                    },
                    "agent_result": result,
                }
                self._send_json(response)
            except Exception as exc:  # pragma: no cover - surfaced to the frontend.
                self._send_json(
                    {
                        "layout_response": {
                            "status": "BACKEND_ERROR",
                            "unsupported_reason": str(exc),
                            "agent_trace": [
                                {
                                    "agent": "PythonBackend",
                                    "status": "error",
                                    "summary": str(exc),
                                }
                            ],
                        },
                        "error": str(exc),
                    },
                    status_code=500,
                )

        def _handle_agentic_stream(self):
            try:
                request = self._read_json()
                session_id = request.get("session_id") or "default"
                scene_context = _scene_context_with_solver(request)
                active_layout_id = scene_context.get("active_layout_id") or None
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.close_connection = True
                for event in stream_orchestrator.stream(
                    _build_pipeline_message(request),
                    session_id=session_id,
                    active_layout_id=active_layout_id,
                    scene_context=scene_context,
                    engineering_context=(
                        request["engineering_context"]
                        if "engineering_context" in request
                        else None
                    ),
                ):
                    self.wfile.write(_sse_encode(event))
                    self.wfile.flush()
            except Exception as exc:  # pragma: no cover - browser integration path.
                self.wfile.write(
                    _sse_encode(
                        {
                            "event": "turn_blocked",
                            "session_id": "unknown",
                            "turn_id": 0,
                            "agent_name": "PythonBackend",
                            "status": "error",
                            "summary": str(exc),
                            "payload": {"error": str(exc), "details": getattr(exc, "details", {}), "exception_type": type(exc).__name__},
                        }
                    )
                )
                self.wfile.flush()

        def _handle_viewport_ack(self):
            try:
                request = self._read_json()
                session_id = str(request.get("session_id") or "").strip()
                if not session_id:
                    raise ViewportReceiptError("session_id is required.")
                verified = stream_orchestrator.record_viewport_receipt(
                    session_id,
                    request.get("receipt"),
                )
                self._send_json({"viewport_receipt": verified})
            except ViewportReceiptError as exc:
                self._send_json(
                    {"status": "VIEWPORT_RECEIPT_REJECTED", "error": str(exc)},
                    status_code=409,
                )

        def log_message(self, format, *args):
            print("%s - %s" % (self.address_string(), format % args), flush=True)

        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            return json.loads(raw)

        def _send_json(self, payload, status_code=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body, content_type, status_code=200):
            self.send_response(status_code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, root, relative_path):
            try:
                target = _safe_join(root, relative_path)
            except ValueError:
                self._send_json({"status": "FORBIDDEN"}, status_code=403)
                return
            self._serve_file(target)

        def _serve_file(self, path):
            path = Path(path)
            if not path.exists() or not path.is_file():
                self._send_json({"status": "NOT_FOUND", "path": str(path)}, status_code=404)
                return

            content_type = _content_type(path)
            content_length = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.send_header("Access-Control-Allow-Origin", "*")
            if (
                path.suffix.lower() in {".html", ".css"}
                or (
                    path.suffix.lower() == ".js"
                    and path.resolve().parent == web_dir
                )
            ):
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
            elif path.name == "glb_manifest.json":
                self.send_header("Cache-Control", "no-cache")
            elif path.suffix.lower() in {".glb", ".js"}:
                self.send_header("Cache-Control", "public, max-age=86400")
            if path.suffix == ".gz":
                self.send_header("Content-Encoding", "gzip")
            elif path.suffix == ".br":
                self.send_header("Content-Encoding", "br")
            self.end_headers()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    self.wfile.write(chunk)

    return RoughDesignApiHandler


def _viewport_config(runtime_models_dir):
    manifest_path = Path(runtime_models_dir) / "glb_manifest.json"
    if not manifest_path.exists():
        return {
            "available": False,
            "renderer": "threejs",
            "manifest_url": None,
            "asset_count": 0,
            "error": "GLB runtime manifest not found",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "available": False,
            "renderer": "threejs",
            "manifest_url": None,
            "asset_count": 0,
            "error": f"Invalid GLB runtime manifest: {exc}",
        }
    assets = manifest.get("assets") or []
    return {
        "available": bool(assets),
        "renderer": "threejs",
        "manifest_url": "/runtime-models/glb_manifest.json",
        "asset_count": len(assets),
        "failure_count": len(manifest.get("failures") or []),
    }


def _safe_join(root, relative_path):
    root = Path(root).resolve()
    relative = unquote(relative_path).replace("\\", "/").lstrip("/")
    target = (root / relative).resolve()
    if os.path.commonpath([str(root), str(target)]) != str(root):
        raise ValueError("path escapes static root")
    return target


def _content_type(path):
    suffixes = path.suffixes
    effective_suffix = suffixes[-2] if suffixes and suffixes[-1] in {".gz", ".br"} and len(suffixes) >= 2 else path.suffix
    if effective_suffix == ".wasm":
        return "application/wasm"
    if effective_suffix == ".glb":
        return "model/gltf-binary"
    if effective_suffix == ".js":
        return "text/javascript"
    if effective_suffix == ".data":
        return "application/octet-stream"
    if effective_suffix == ".symbols":
        return "application/octet-stream"
    content_type, _encoding = mimetypes.guess_type(str(path))
    return content_type or "application/octet-stream"


def _build_pipeline_message(request):
    parts = [request.get("raw_user_message", "").strip()]
    structured = request.get("structured_constraints") or {}
    width = structured.get("site_width_m")
    depth = structured.get("site_depth_m")
    if width and depth:
        parts.append(f"Site {float(width):g}m x {float(depth):g}m.")

    selected_assets = request.get("selected_assets") or []
    asset_ids = [asset.get("model_asset_id") for asset in selected_assets if asset.get("model_asset_id")]
    if asset_ids:
        parts.append("Selected model_asset_id " + " ".join(asset_ids) + ".")

    hard_constraints = structured.get("hard_constraints") or []
    if any(str(item).lower() == "no_overlap" for item in hard_constraints):
        parts.append("Hard constraint: no overlap.")

    objectives = structured.get("objectives") or []
    if objectives:
        parts.append("Objectives: " + ", ".join(str(item) for item in objectives) + ".")
    if any("distance" in str(item).lower() for item in objectives):
        parts.append("Minimize material handling distance.")

    return " ".join(part for part in parts if part)


def _scene_context_with_solver(request):
    scene_context = dict(request.get("scene_context") or {})
    if "engineering_context" in request:
        if not isinstance(request["engineering_context"], str):
            raise ValueError("engineering_context must be a natural-language string.")
        scene_context["engineering_context"] = request["engineering_context"]
    structured = request.get("structured_constraints") or {}
    requested_solver = next(
        (
            value
            for value in (
                request.get("solver"),
                request.get("solver_name"),
                request.get("optimizer"),
                request.get("method"),
                structured.get("solver"),
                structured.get("optimizer"),
                structured.get("method"),
                scene_context.get("solver"),
                scene_context.get("optimizer"),
                scene_context.get("method"),
            )
            if value is not None and str(value).strip()
        ),
        None,
    )
    if requested_solver is not None:
        scene_context["solver"] = requested_solver
        scene_context["solver_selection_source"] = "user"
    return scene_context


def _to_layout_response(result):
    validation_report = result.get("validation", {}).get("validation_report") or {}
    validation_report = dict(validation_report)
    validation_report["solver_ready"] = validation_report.get("status") in {
        "VALID_SOLVER_READY",
        "VALID_WITH_WARNINGS",
    }

    evaluation = result.get("evaluation") or {}
    metrics = evaluation.get("metrics") or {}
    layout_evaluation = {
        "status": evaluation.get("evaluation_status"),
        "summary": "; ".join(evaluation.get("notes") or []),
        "metrics": {
            "estimated_area_utilization": metrics.get("site_utilization", 0.0),
            "overlap_count": 0 if metrics.get("no_overlap") else 1,
            "missing_reference_count": 0,
            "unresolved_item_count": len((result.get("design", {}).get("flp_ir") or {}).get("unresolved_items", [])),
            **metrics,
        },
    }

    intent = result.get("intent") or {}
    semantic_intent = result.get("semantic_intent") or {}
    repair = result.get("validation", {}).get("repair") or {}
    return {
        "status": result.get("status"),
        "unsupported_reason": repair.get("unsupported_reason"),
        "intent": {
            "intent": intent.get("intent_type"),
            "task_type": intent.get("intent_type"),
            "confidence": intent.get("confidence", 0.0),
            "reason": semantic_intent.get("reasoning_summary") or "Parsed by Python IntentUnderstandingAgent.",
        },
        "semantic_intent": semantic_intent,
        "layout_edit_ir": result.get("design", {}).get("layout_edit_ir"),
        "layout_diff": result.get("design", {}).get("layout_diff"),
        "flp_ir": result.get("design", {}).get("flp_ir"),
        "rough_layout": result.get("design", {}).get("rough_layout"),
        "solver_report": result.get("design", {}).get("solver_report"),
        "solver_used": result.get("design", {}).get("solver_used") or result.get("solver_used"),
        "validation_report": validation_report,
        "clarification_questions": repair.get("clarification_questions") or [],
        "evaluation": layout_evaluation,
        "agent_trace": [_to_layout_trace(item) for item in result.get("agent_trace", [])],
        "loop_report": result.get("loop_report") or {},
        "revision_history": result.get("revision_history") or [],
        "agent_contracts": {
            name: outputs[0].get("contract") if outputs else None
            for name, outputs in (result.get("agent_outputs") or {}).items()
        },
        "assistant_message": result.get("assistant_message"),
        "assistant_message_source": result.get("assistant_message_source"),
        "assistant_message_error": result.get("assistant_message_error"),
    }


def _to_layout_trace(trace):
    return {
        "agent": trace.get("agent_name"),
        "status": "done",
        "summary": f"{trace.get('input_ref')} -> {trace.get('output_ref')}",
        "iteration": trace.get("iteration"),
        "decision": trace.get("decision"),
        "input_artifacts": trace.get("input_artifacts") or [],
        "output_artifacts": trace.get("output_artifacts") or [],
        "evidence": trace.get("evidence") or [],
    }


def _make_intent_llm_client():
    return IntentLLMClient.from_env()


def _sse_encode(event):
    payload = json.dumps(event, ensure_ascii=False)
    return f"event: {event.get('event', 'message')}\ndata: {payload}\n\n".encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the FLP rough-design JSON API for the Three.js frontend.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dataset-dir", default=str(Path(__file__).resolve().parents[2] / "rag_dataset"))
    parser.add_argument("--web-dir", default=str(DEFAULT_WEB_DIR))
    parser.add_argument("--runtime-models-dir", default=str(DEFAULT_RUNTIME_MODELS_DIR))
    parser.add_argument("--workspace-snapshot", help="Display an explicitly selected recorded workspace.")
    parser.add_argument("--recording-only", action="store_true", help="Serve a read-only recording without initializing LLM or retrieval clients.")
    parser.add_argument(
        "--rag-index-dir",
        default=os.environ.get(
            "ENGINEERING_RAG_INDEX_DIR",
            str(Path(".rag_index") / "engineering_rag_qdrant"),
        ),
    )
    args = parser.parse_args()

    handler = make_handler(
        args.dataset_dir,
        web_dir=args.web_dir,
        runtime_models_dir=args.runtime_models_dir,
        rag_index_dir=args.rag_index_dir,
        workspace_snapshot_path=args.workspace_snapshot,
        recording_only=args.recording_only,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Rough design API listening on http://{args.host}:{args.port}", flush=True)
    print(f"Dataset: {Path(args.dataset_dir).resolve()}", flush=True)
    print(f"Web frontend: {Path(args.web_dir).resolve()}", flush=True)
    print(f"Runtime models: {Path(args.runtime_models_dir).resolve()}", flush=True)
    print(f"Engineering RAG index: {Path(args.rag_index_dir).resolve()}", flush=True)
    if args.recording_only:
        print("Mode: read-only recording; inference and execution are disabled.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
