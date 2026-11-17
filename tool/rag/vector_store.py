import uuid
import threading
import time
from pathlib import Path

from tool.clients.llm_clients import RAGBlockedError


class QdrantVectorStore:
    def __init__(self, index_dir=".rag_index/engineering_rag_qdrant", *, collection_name="engineering_rag"):
        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self._client = None

    def recreate_collection(self, vector_size):
        client = self._load_client()
        try:
            from qdrant_client.models import Distance, VectorParams

            if client.collection_exists(self.collection_name):
                client.delete_collection(self.collection_name)
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=int(vector_size), distance=Distance.COSINE),
            )
        except Exception as exc:
            raise RAGBlockedError(
                "qdrant_collection_error",
                "Qdrant collection could not be recreated.",
                details={"collection_name": self.collection_name, "index_dir": str(self.index_dir)},
                suggestions=["Check local filesystem permissions and qdrant-client installation."],
            ) from exc

    def upsert(self, points):
        client = self._load_client()
        try:
            from qdrant_client.models import PointStruct

            qdrant_points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, point["id"])),
                    vector=point["vector"],
                    payload=point["payload"],
                )
                for point in points
            ]
            client.upsert(collection_name=self.collection_name, points=qdrant_points)
        except Exception as exc:
            raise RAGBlockedError(
                "qdrant_upsert_error",
                "Evidence vectors could not be written to Qdrant.",
                details={"collection_name": self.collection_name, "point_count": len(points)},
                suggestions=["Check Qdrant local index directory and payload serializability."],
            ) from exc

    def search(self, vector, top_k, filters=None):
        client = self._load_client()
        query_filter = self._build_filter(filters)
        try:
            if hasattr(client, "search"):
                hits = client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    query_filter=query_filter,
                    limit=int(top_k),
                )
            else:
                result = client.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    query_filter=query_filter,
                    limit=int(top_k),
                )
                hits = result.points
        except Exception as exc:
            raise RAGBlockedError(
                "qdrant_search_error",
                "Qdrant search failed.",
                details={"collection_name": self.collection_name, "index_dir": str(self.index_dir)},
                suggestions=["Run the RAG index command before querying and verify the local index path."],
            ) from exc
        return [{"score": float(hit.score), "payload": dict(hit.payload or {})} for hit in hits]

    def _build_filter(self, filters):
        if not filters:
            return None
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
        except ImportError as exc:
            raise RAGBlockedError(
                "missing_dependency",
                "qdrant-client filter models are unavailable.",
                details={"dependency": "qdrant-client"},
            ) from exc
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
            if value is not None
        ]
        return Filter(must=conditions) if conditions else None

    def _load_client(self):
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            raise RAGBlockedError(
                "missing_dependency",
                "qdrant-client is required for the Engineering RAG vector store.",
                details={"dependency": "qdrant-client"},
                suggestions=["Install qdrant-client before running RAG commands."],
            ) from exc
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=str(self.index_dir))
        return self._client

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None


class TunnelVectorStore:
    MAX_TOP_K = 200
    ALLOWED_FILTERS = {
        "doc_type",
        "evidence_id",
        "field_path",
        "model_asset_id",
        "source_id",
        "unresolved_id",
    }
    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url,
        token,
        *,
        collection_name="engineering_rag",
        timeout_s=60,
        max_connections=4,
        max_attempts=3,
        expected_vector_size=1024,
        expected_evidence_count=755,
        expected_evidence_sha256=None,
        sleep_fn=time.sleep,
    ):
        if not base_url or not token:
            raise RAGBlockedError(
                "qdrant_tunnel_configuration_missing",
                "The Qdrant tunnel URL and session token are required.",
            )
        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.timeout_s = float(timeout_s)
        self.max_connections = int(max_connections)
        self.max_attempts = int(max_attempts)
        self.expected_vector_size = int(expected_vector_size)
        self.expected_evidence_count = int(expected_evidence_count)
        self.expected_evidence_sha256 = expected_evidence_sha256
        self._token = token
        self._sleep = sleep_fn
        self._client = None
        self._client_lock = threading.Lock()
        self._manifest = None

    def validate(self):
        manifest = self._request("GET", "/manifest")
        mismatches = {}
        expected = {
            "collection_name": self.collection_name,
            "evidence_count": self.expected_evidence_count,
            "vector_size": self.expected_vector_size,
        }
        if self.expected_evidence_sha256:
            expected["evidence_sha256"] = self.expected_evidence_sha256
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                mismatches[key] = {
                    "expected": expected_value,
                    "actual": manifest.get(key),
                }
        if mismatches:
            raise RAGBlockedError(
                "qdrant_tunnel_manifest_mismatch",
                "The tunneled Qdrant manifest does not match the frozen experiment.",
                details={"mismatches": mismatches},
            )
        self._manifest = manifest
        return manifest

    def search(self, vector, top_k, filters=None):
        if len(vector) != self.expected_vector_size:
            raise RAGBlockedError(
                "qdrant_tunnel_vector_size_mismatch",
                "The query vector does not match the frozen vector size.",
                details={"expected": self.expected_vector_size, "actual": len(vector)},
            )
        if not 1 <= int(top_k) <= self.MAX_TOP_K:
            raise RAGBlockedError(
                "qdrant_tunnel_top_k_invalid",
                f"The read-only Qdrant proxy accepts top_k from 1 through {self.MAX_TOP_K}.",
                details={"top_k": top_k},
            )
        unsupported = sorted(set((filters or {}).keys()) - self.ALLOWED_FILTERS)
        if unsupported:
            raise RAGBlockedError(
                "qdrant_tunnel_filter_rejected",
                "The Qdrant proxy rejected non-allowlisted filters.",
                details={"filters": unsupported},
            )
        payload = {
            "collection_name": self.collection_name,
            "vector": [float(item) for item in vector],
            "top_k": int(top_k),
            "filters": filters or {},
        }
        result = self._request("POST", "/search", json_payload=payload)
        return result.get("hits", [])

    def close(self):
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def _request(self, method, path, *, json_payload=None):
        try:
            import httpx
        except ImportError as exc:
            raise RAGBlockedError(
                "missing_dependency",
                "httpx is required for the tunneled Qdrant vector store.",
                details={"dependency": "httpx"},
            ) from exc
        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._http_client(httpx).request(
                    method,
                    f"{self.base_url}{path}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    json=json_payload,
                )
                if response.status_code >= 400:
                    if (
                        response.status_code in self.RETRYABLE_STATUS_CODES
                        and attempts < self.max_attempts
                    ):
                        self._sleep(0.5 * (2 ** (attempts - 1)))
                        continue
                    response.raise_for_status()
                result = response.json()
                if not isinstance(result, dict):
                    raise ValueError("Proxy response must be a JSON object")
                return result
            except httpx.TransportError as exc:
                if attempts < self.max_attempts:
                    self._sleep(0.5 * (2 ** (attempts - 1)))
                    continue
                raise RAGBlockedError(
                    "qdrant_tunnel_transport_error",
                    "The Qdrant reverse tunnel failed after three attempts.",
                    details={"attempts": attempts, "endpoint": path},
                ) from exc
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                blocker = (
                    "qdrant_tunnel_authentication_error"
                    if status in {401, 403}
                    else "qdrant_tunnel_http_error"
                )
                raise RAGBlockedError(
                    blocker,
                    "The read-only Qdrant proxy rejected the request.",
                    details={"http_status": status, "attempts": attempts, "endpoint": path},
                ) from exc
            except (TypeError, ValueError) as exc:
                raise RAGBlockedError(
                    "qdrant_tunnel_invalid_response",
                    "The read-only Qdrant proxy returned invalid JSON.",
                    details={"attempts": attempts, "endpoint": path},
                ) from exc

    def _http_client(self, httpx):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is None:
                limits = httpx.Limits(
                    max_connections=self.max_connections,
                    max_keepalive_connections=self.max_connections,
                )
                self._client = httpx.Client(timeout=self.timeout_s, limits=limits)
        return self._client
