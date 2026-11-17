from tool.clients.llm_clients import RAGBlockedError, _env_value


class BGEM3EmbeddingProvider:
    def __init__(self, model_name="BAAI/bge-m3", *, use_fp16=False, batch_size=12):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self._model = None

    @classmethod
    def from_env(cls, *, env=None, default_model_name="BAAI/bge-m3", use_fp16=False, batch_size=12):
        import os

        env = os.environ if env is None else env
        model_name = (
            _env_value(env, "BGE_M3_MODEL_PATH")
            or _env_value(env, "BGE_M3_MODEL_NAME")
            or default_model_name
        )
        return cls(model_name=model_name, use_fp16=use_fp16, batch_size=batch_size)

    def embed_texts(self, texts):
        if not texts:
            return []
        model = self._load_model()
        try:
            encoded = model.encode(
                list(texts),
                batch_size=self.batch_size,
                max_length=8192,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except Exception as exc:
            raise RAGBlockedError(
                "embedding_failed",
                "BGE-M3 embedding failed.",
                details={"model_name": self.model_name},
                suggestions=["Check model download/cache availability and local torch/transformers setup."],
            ) from exc
        dense = encoded.get("dense_vecs") if isinstance(encoded, dict) else encoded
        return [_to_float_list(vector) for vector in dense]

    def ensure_ready(self):
        self._load_model()

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            # On Windows, loading Torch/Transformers before Arrow can make
            # arrow.dll fail with an access violation.  Load Arrow first so
            # FlagEmbedding's datasets dependency reuses the initialized DLL.
            import pyarrow  # noqa: F401

            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RAGBlockedError(
                "missing_dependency",
                "FlagEmbedding is required for BGE-M3 embeddings.",
                details={"dependency": "FlagEmbedding", "model_name": self.model_name},
                suggestions=["Install FlagEmbedding, transformers, and torch before running RAG commands."],
            ) from exc
        try:
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
        except Exception as exc:
            raise RAGBlockedError(
                "embedding_model_unavailable",
                "BGE-M3 model could not be loaded.",
                details={"model_name": self.model_name},
                suggestions=["Ensure network access for first download or a valid local model cache."],
            ) from exc
        return self._model


def _to_float_list(vector):
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(item) for item in vector]
