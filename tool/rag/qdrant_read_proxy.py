import argparse
import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tool.clients.llm_clients import RAGBlockedError
from tool.rag.engineering_rag import _canonical_sha256
from tool.rag.vector_store import QdrantVectorStore


ALLOWED_FILTERS = {
    "doc_type",
    "evidence_id",
    "field_path",
    "model_asset_id",
    "source_id",
    "unresolved_id",
}
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_TOP_K = 200


class ReadOnlyQdrantProxy:
    def __init__(self, index_dir, token):
        if not token:
            raise RAGBlockedError(
                "qdrant_proxy_token_missing",
                "A random session token is required for the Qdrant proxy.",
            )
        self.index_dir = Path(index_dir)
        self.token = token
        self.manifest = json.loads(
            (self.index_dir / "index_manifest.json").read_text(encoding="utf-8")
        )
        self.vector_size = int(self.manifest["vector_size"])
        self.collection_name = self.manifest["collection_name"]
        self.store = QdrantVectorStore(
            self.index_dir, collection_name=self.collection_name
        )
        self.search_lock = threading.Lock()
        self._validate_local_collection()

    def close(self):
        self.store.close()

    def authorized(self, authorization):
        expected = f"Bearer {self.token}"
        return hmac.compare_digest(str(authorization or ""), expected)

    def search(self, payload):
        if payload.get("collection_name") != self.collection_name:
            raise RequestRejected(400, "collection_name_mismatch")
        vector = payload.get("vector")
        if not isinstance(vector, list) or len(vector) != self.vector_size:
            raise RequestRejected(400, "vector_size_mismatch")
        top_k = payload.get("top_k")
        if not isinstance(top_k, int) or not 1 <= top_k <= MAX_TOP_K:
            raise RequestRejected(400, "top_k_out_of_range")
        filters = payload.get("filters") or {}
        if not isinstance(filters, dict):
            raise RequestRejected(400, "filters_must_be_object")
        unsupported = sorted(set(filters) - ALLOWED_FILTERS)
        if unsupported:
            raise RequestRejected(400, "filter_not_allowlisted", {"filters": unsupported})
        try:
            vector = [float(item) for item in vector]
        except (TypeError, ValueError) as exc:
            raise RequestRejected(400, "vector_must_be_numeric") from exc
        with self.search_lock:
            hits = self.store.search(vector, top_k, filters=filters)
        return {"hits": hits}

    def _validate_local_collection(self):
        client = self.store._load_client()
        collection = client.get_collection(self.collection_name)
        actual_size = getattr(collection.config.params.vectors, "size", None)
        if collection.points_count != self.manifest["evidence_count"]:
            raise RAGBlockedError(
                "qdrant_proxy_point_count_mismatch",
                "Local Qdrant point count does not match the index manifest.",
                details={
                    "expected": self.manifest["evidence_count"],
                    "actual": collection.points_count,
                },
            )
        if actual_size != self.vector_size:
            raise RAGBlockedError(
                "qdrant_proxy_vector_size_mismatch",
                "Local Qdrant vector size does not match the index manifest.",
                details={"expected": self.vector_size, "actual": actual_size},
            )
        payloads = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            payloads.extend(dict(point.payload or {}) for point in points)
            if offset is None:
                break
        actual_hash = _canonical_sha256(payloads)
        if actual_hash != self.manifest["evidence_sha256"]:
            raise RAGBlockedError(
                "qdrant_proxy_evidence_hash_mismatch",
                "Local Qdrant payloads do not match the frozen evidence hash.",
                details={
                    "expected": self.manifest["evidence_sha256"],
                    "actual": actual_hash,
                },
            )


class RequestRejected(ValueError):
    def __init__(self, status_code, code, details=None):
        super().__init__(code)
        self.status_code = status_code
        self.code = code
        self.details = details or {}


def build_handler(proxy):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EngineeringRAGQdrantProxy/1.0"

        def do_GET(self):
            if not self._authenticate():
                return
            if self.path == "/health":
                self._json_response(200, {"status": "OK", "mode": "read_only"})
            elif self.path == "/manifest":
                self._json_response(200, proxy.manifest)
            else:
                self._json_response(404, {"error": "not_found"})

        def do_POST(self):
            if not self._authenticate():
                return
            if self.path != "/search":
                self._json_response(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_BODY_BYTES:
                    raise RequestRejected(413, "invalid_body_size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise RequestRejected(400, "body_must_be_object")
                self._json_response(200, proxy.search(payload))
            except RequestRejected as exc:
                self._json_response(
                    exc.status_code, {"error": exc.code, "details": exc.details}
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json_response(400, {"error": "invalid_json"})
            except RAGBlockedError as exc:
                self._json_response(503, exc.to_report())

        def _authenticate(self):
            if proxy.authorized(self.headers.get("Authorization")):
                return True
            self._json_response(401, {"error": "unauthorized"})
            return False

        def _json_response(self, status_code, payload):
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    return Handler


def serve(index_dir, host, port, token):
    proxy = ReadOnlyQdrantProxy(index_dir, token)
    server = ThreadingHTTPServer((host, port), build_handler(proxy))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        proxy.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serve a read-only local Qdrant search proxy.")
    parser.add_argument("--index-dir", default=".rag_index/engineering_rag_qdrant")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-env", default="RAG_QDRANT_PROXY_TOKEN")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env)
    try:
        serve(args.index_dir, args.host, args.port, token)
    except RAGBlockedError as exc:
        print(json.dumps(exc.to_report(), ensure_ascii=False, indent=2))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
