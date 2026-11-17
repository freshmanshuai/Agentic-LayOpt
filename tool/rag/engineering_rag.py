import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from tool.clients.llm_clients import DeepSeekClient, RAGBlockedError


_DEFAULT = object()
DOMAIN_KEYWORDS = {
    "semiconductor": [
        "半导体",
        "芯片",
        "晶圆",
        "硅片",
        "贴片",
        "编带",
        "cleanroom",
        "semiconductor",
        "chip",
        "wafer",
        "pcb",
        "smd",
        "packaging",
        "placing",
    ],
    "server_room": ["服务器", "机柜", "机房", "server", "rack", "data center", "datacenter"],
    "electrical": ["电气", "变压器", "接线盒", "electrical", "transformer", "junction"],
    "fluid_mechanical": ["泵", "风机", "罐", "料斗", "pump", "blower", "tank", "hopper"],
    "machine_tool": [
        "机床",
        "车床",
        "铣床",
        "切割机",
        "machine tool",
        "lathe",
        "milling",
        "cutting",
        "machining",
        "cnc",
        "machining center",
    ],
    "general_factory": [
        "工厂",
        "车间",
        "产线",
        "工业",
        "生产区",
        "生产",
        "factory",
        "workshop",
        "production line",
        "production",
        "plant",
        "industrial",
        "manufacturing",
    ],
}
DOMAIN_EXCLUSION_KEYWORDS = {
    "server_room": ["不要服务器", "不要机柜", "without server", "no server", "exclude server", "without rack"],
    "fluid_mechanical": ["不要泵", "without pump", "no pump", "exclude pump"],
    "electrical": ["不要变压器", "without transformer", "no transformer", "exclude transformer"],
}
DOMAIN_CASE_TYPES = {"domain_intent_layout"}
DOMAIN_PRIORITY = {
    "semiconductor": 0,
    "server_room": 1,
    "fluid_mechanical": 2,
    "electrical": 3,
    "machine_tool": 4,
    "general_factory": 99,
    "unknown": 100,
}


class EngineeringRAGService:
    def __init__(
        self,
        dataset_dir,
        *,
        index_dir=Path(".rag_index") / "engineering_rag_qdrant",
        llm_client=_DEFAULT,
        embedding_provider=_DEFAULT,
        vector_store=_DEFAULT,
        resolved_score=0.5,
        non_exact_resolved_score=0.75,
        ambiguity_margin=0.05,
        retrieval_top_k=200,
        rerank_vector_weight=0.25,
        exhaustive_candidate_ceiling=100,
        identity_plausibility_floor=0.75,
        identity_plausibility_max_score_gap=0.10,
        catalog_decision=False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.index_dir = Path(index_dir)
        self.resolved_score = resolved_score
        self.non_exact_resolved_score = non_exact_resolved_score
        self.ambiguity_margin = ambiguity_margin
        self.retrieval_top_k = int(retrieval_top_k)
        self.rerank_vector_weight = float(rerank_vector_weight)
        self.exhaustive_candidate_ceiling = int(exhaustive_candidate_ceiling)
        self.identity_plausibility_floor = float(identity_plausibility_floor)
        self.identity_plausibility_max_score_gap = float(
            identity_plausibility_max_score_gap
        )
        self.catalog_decision = bool(catalog_decision)
        self.llm_client = self._default_llm() if llm_client is _DEFAULT else llm_client
        self.embedding_provider = (
            self._default_embedding_provider() if embedding_provider is _DEFAULT else embedding_provider
        )
        self.vector_store = self._default_vector_store() if vector_store is _DEFAULT else vector_store
        self._docs_by_id = {}

    def index(self, *, progress=None):
        self._ensure_runtime_ready()
        _progress(progress, "building evidence documents")
        docs = build_evidence_documents(self.dataset_dir)
        if not docs:
            raise RAGBlockedError(
                "empty_evidence_set",
                "No evidence documents were built from the dataset.",
                details={"dataset_dir": str(self.dataset_dir)},
            )
        _progress(progress, f"built {len(docs)} evidence documents")
        _progress(progress, "embedding evidence documents with BGE-M3")
        vectors = self.embedding_provider.embed_texts([doc["text"] for doc in docs])
        if not vectors or len(vectors) != len(docs):
            raise RAGBlockedError(
                "embedding_count_mismatch",
                "Embedding provider returned an unexpected number of vectors.",
                details={"document_count": len(docs), "vector_count": len(vectors)},
            )
        vector_size = len(vectors[0])
        if not vector_size:
            raise RAGBlockedError("empty_embedding_vector", "Embedding provider returned empty vectors.")
        _progress(progress, f"embedding complete; vector_size={vector_size}")
        _progress(progress, "recreating Qdrant collection")
        self.vector_store.recreate_collection(vector_size)
        points = [
            {"id": doc["evidence_id"], "vector": vector, "payload": doc}
            for doc, vector in zip(docs, vectors)
        ]
        _progress(progress, f"upserting {len(points)} evidence vectors")
        self.vector_store.upsert(points)
        self._docs_by_id = {doc["evidence_id"]: doc for doc in docs}
        manifest = {
            "schema_version": "1.0",
            "collection_name": self.vector_store.collection_name,
            "evidence_count": len(docs),
            "evidence_sha256": _canonical_sha256(docs),
            "vector_size": vector_size,
            "embedding_model": _model_identifier(
                getattr(self.embedding_provider, "model_name", None)
            ),
        }
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            (self.index_dir / "index_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise RAGBlockedError(
                "index_manifest_write_error",
                "The Engineering RAG index manifest could not be written.",
                details={"index_dir": str(self.index_dir)},
            ) from exc
        _progress(progress, "index complete")
        return {
            "status": "OK",
            "indexed_evidence_count": len(docs),
            "vector_size": vector_size,
            "evidence_sha256": manifest["evidence_sha256"],
        }

    def query(self, query, *, top_k=5, filters=None, asset_mentions=None):
        self._ensure_runtime_ready()
        catalog_plan = (
            self.llm_client.decide_grounding(
                query,
                catalog=self._decision_catalog(),
            )
            if self.catalog_decision
            else None
        )
        if asset_mentions is None:
            if catalog_plan is not None:
                raw_mentions = [
                    item.get("surface_text", "")
                    for item in catalog_plan.get("mentions", [])
                    if isinstance(item, dict) and str(item.get("surface_text") or "").strip()
                ]
                raw_mentions = _unique_coreferential_surfaces(raw_mentions)
                extraction = {
                    "normalized_query": query,
                    "asset_mentions": raw_mentions,
                    "constraint_terms": [],
                    "objective_terms": [],
                    "missing_field_terms": [],
                    "language": "unknown",
                }
            else:
                extraction = self.llm_client.extract_query(query)
                raw_mentions = extraction.get("asset_mentions") or []
        else:
            raw_mentions = [str(item) for item in asset_mentions if str(item).strip()]
            extraction = {
                "normalized_query": query,
                "asset_mentions": raw_mentions,
                "constraint_terms": [],
                "objective_terms": [],
                "missing_field_terms": [],
                "language": "unknown",
            }
        repaired_mentions = _repair_asset_mentions(query, raw_mentions)
        mention_specs, rejected_mentions = _prepare_asset_mentions(query, repaired_mentions)
        fallback_extraction = None
        if not mention_specs and self.catalog_decision and asset_mentions is None:
            fallback_extraction = self.llm_client.extract_query(query)
            fallback_mentions = _repair_asset_mentions(
                query, fallback_extraction.get("asset_mentions") or []
            )
            mention_specs, fallback_rejected = _prepare_asset_mentions(
                query, fallback_mentions
            )
            rejected_mentions.extend(fallback_rejected)
        if not mention_specs:
            mention_specs = _prepare_asset_mentions(
                query, [extraction.get("normalized_query") or query], allow_context=True
            )[0]
        extraction = dict(extraction)
        extraction["raw_asset_mentions"] = list(raw_mentions)
        extraction["asset_mentions"] = [item["surface_text"] for item in mention_specs]
        extraction["mention_specs"] = mention_specs
        extraction["rejected_asset_mentions"] = rejected_mentions
        extraction["fallback_query_parse"] = fallback_extraction
        mention_results = {}
        rerank_decisions = {}
        summaries = []
        for mention_spec in mention_specs:
            search_text = mention_spec["surface_text"]
            query_vector = self.embedding_provider.embed_texts([search_text])[0]
            raw_hits = self.vector_store.search(
                query_vector,
                top_k=max(self.retrieval_top_k, top_k * 8),
                filters=filters,
            )
            candidates = self._asset_identity_candidates(raw_hits)
            ranked = self.llm_client.rerank(search_text, candidates) if candidates else {
                "ranked_evidence": [],
                "summary": "",
            }
            if ranked.get("summary"):
                summaries.append(str(ranked["summary"]))
            if isinstance(ranked.get("identity_decision"), dict):
                rerank_decisions[mention_spec["mention_id"]] = {
                    **ranked["identity_decision"],
                    "surface_text": search_text,
                    "decision_source": "retrieval_constrained_llm",
                }
            ranked_results = sorted(
                _apply_llm_rerank(
                    candidates,
                    ranked,
                    vector_weight=self.rerank_vector_weight,
                ),
                key=lambda item: (-item["score"], item["evidence_id"]),
            )
            best_by_asset = {}
            for item in ranked_results:
                asset_id = item.get("model_asset_id")
                if asset_id and asset_id not in best_by_asset:
                    best_by_asset[asset_id] = item
            # Keep the full exhaustive set until the identity decision has been
            # applied.  Truncating by the blended score here can discard the ID
            # explicitly selected by the constrained reranker (most visibly in
            # the surface ablation, where there is no catalog plan).
            candidate_limit = (
                len(self._assets())
                if catalog_plan or mention_spec["mention_id"] in rerank_decisions
                else top_k
            )
            mention_results[mention_spec["mention_id"]] = list(best_by_asset.values())[
                :candidate_limit
            ]

        asset_matches = self._asset_matches_from_results(
            mention_specs,
            mention_results,
            decision_plan=catalog_plan,
            rerank_decisions=rerank_decisions,
            top_k=top_k,
        )
        retained_asset_ids = {
            item.get("model_asset_id")
            for match in asset_matches
            for item in match.get("candidate_model_asset_ids", [])
            if item.get("model_asset_id")
        }
        retained_asset_ids.update(
            match.get("selected_model_asset_id")
            for match in asset_matches
            if match.get("selected_model_asset_id")
        )
        all_results = {}
        for items in mention_results.values():
            for item in items:
                if item.get("model_asset_id") not in retained_asset_ids:
                    continue
                current = all_results.get(item["evidence_id"])
                if current is None or item["score"] > current["score"]:
                    all_results[item["evidence_id"]] = item
        results = sorted(
            all_results.values(),
            key=lambda item: (-item["score"], item["evidence_id"]),
        )
        excluded_domains = infer_excluded_domains(query)
        domain_intent = infer_domain_intent(query, extraction)
        domain_asset_matches = self._domain_asset_matches(
            domain_intent,
            excluded_domains=excluded_domains,
            top_k=top_k,
        )
        negative_domain_hits = self._negative_domain_hits(query, domain_asset_matches)
        return {
            "status": "OK",
            "query": query,
            "query_parse": extraction,
            "domain_intent": domain_intent,
            "domain_asset_matches": domain_asset_matches,
            "negative_domain_hits": negative_domain_hits,
            "results": results,
            "asset_matches": asset_matches,
            "catalog_decision_plan": catalog_plan,
            "summary": "\n".join(summaries),
        }

    def query_exact_assets(self, query, model_asset_ids, *, top_k=24):
        """Ground a frozen contract without re-deciding its exact asset identities."""
        self._ensure_runtime_ready()
        requested_ids = list(dict.fromkeys(str(item) for item in model_asset_ids if item))
        assets_by_id = {item["model_asset_id"]: item for item in self._assets()}
        missing_ids = [item for item in requested_ids if item not in assets_by_id]
        if missing_ids:
            raise RAGBlockedError(
                "unknown_exact_asset_id",
                "The frozen contract references asset identities absent from the catalog.",
                details={"model_asset_ids": missing_ids},
            )

        query_vector = self.embedding_provider.embed_texts([query])[0]
        raw_hits = self.vector_store.search(
            query_vector,
            top_k=max(1, min(self.retrieval_top_k, int(top_k) * 8)),
        )
        requested_id_set = set(requested_ids)
        retrieved_results = []
        for raw in raw_hits:
            payload = raw.get("payload") or {}
            if payload.get("model_asset_id") not in requested_id_set:
                continue
            result = _payload_to_doc(payload)
            result["score"] = float(raw.get("score", 0.0))
            retrieved_results.append(result)

        summary_results = []
        matches = []
        for index, asset_id in enumerate(requested_ids, start=1):
            asset = assets_by_id[asset_id]
            summary = _asset_summary_doc(asset, None)
            summary["score"] = 1.0
            summary_results.append(summary)
            matches.append(
                {
                    "mention_id": f"exact_asset_{index}",
                    "surface_text": asset_id,
                    "start_char": None,
                    "end_char": None,
                    "decision": "RESOLVED",
                    "selected_model_asset_id": asset_id,
                    "candidate_model_asset_ids": [self._exact_candidate(asset_id)],
                    "source_provenance": [
                        {
                            "mention_id": f"exact_asset_{index}",
                            "evidence": [
                                {
                                    "evidence_id": summary["evidence_id"],
                                    "source_id": asset_id,
                                }
                            ],
                        }
                    ],
                    "decision_source": "authoritative_contract_exact_id",
                }
            )

        results_by_id = {
            item["evidence_id"]: item for item in summary_results + retrieved_results
        }
        results = sorted(
            results_by_id.values(),
            key=lambda item: (-float(item.get("score", 0.0)), item["evidence_id"]),
        )
        return {
            "status": "OK",
            "query": query,
            "query_parse": {
                "normalized_query": query,
                "asset_mentions": requested_ids,
                "constraint_terms": [],
                "objective_terms": [],
                "missing_field_terms": [],
                "language": "unknown",
            },
            "domain_intent": {},
            "domain_asset_matches": [],
            "negative_domain_hits": [],
            "results": results,
            "asset_matches": matches,
            "catalog_decision_plan": None,
            "summary": (
                f"Resolved {len(requested_ids)} exact asset identities from the "
                "authoritative frozen contract; one semantic evidence retrieval executed."
            ),
            "provenance": {
                "mode": "exact_contract_binding",
                "identity_source": "model_assets.json",
                "semantic_retrieval_count": 1,
            },
        }

    def find_asset_candidates(self, query, *, top_k=5):
        result = self.query(query, top_k=top_k)
        candidates = []
        for match in result.get("asset_matches", []):
            candidates.extend(match.get("candidate_model_asset_ids", []))
        candidates.sort(key=lambda item: (-item["score"], item["model_asset_id"]))
        return candidates

    def _asset_matches_from_results(
        self,
        mention_specs,
        mention_results,
        *,
        decision_plan=None,
        rerank_decisions=None,
        top_k=None,
    ):
        mention_specs = [
            item
            if isinstance(item, dict)
            else {
                "mention_id": str(item),
                "surface_text": str(item),
                "start_char": None,
                "end_char": None,
            }
            for item in mention_specs
        ]
        plan_mentions = [
            item
            for item in (decision_plan or {}).get("mentions", [])
            if isinstance(item, dict)
        ]
        used_plan_indexes = set()
        matches = []
        for mention_spec in mention_specs:
            search_text = mention_spec["surface_text"]
            rerank_mention = (rerank_decisions or {}).get(
                mention_spec["mention_id"]
            )
            catalog_mention, plan_index = _matching_plan_mention(
                search_text,
                plan_mentions,
                used_plan_indexes,
            )
            if plan_index is not None:
                used_plan_indexes.add(plan_index)
            plan_mention = rerank_mention or catalog_mention
            catalog_decision = str(
                (catalog_mention or {}).get("decision") or ""
            ).upper()
            catalog_candidate_ids = {
                str(item)
                for item in (catalog_mention or {}).get(
                    "candidate_model_asset_ids", []
                )
                if item
            }
            catalog_selected_id = (catalog_mention or {}).get(
                "selected_model_asset_id"
            )
            generic_surface = _is_generic_asset_surface(search_text)
            valid_catalog_clarify = (
                catalog_decision == "CLARIFY" and generic_surface
            )
            strong_catalog_resolution = (
                catalog_decision == "RESOLVE"
                and bool(catalog_selected_id)
                and catalog_candidate_ids == {str(catalog_selected_id)}
                and not generic_surface
            )
            # The catalog decision sees the complete query and all catalog identities,
            # whereas reranking sees one mention at a time.  Do not let the narrower
            # call override an explicit ambiguity decision or a high-confidence,
            # single-identity catalog resolution.
            if valid_catalog_clarify or strong_catalog_resolution:
                plan_mention = catalog_mention
            results = mention_results.get(
                mention_spec["mention_id"], mention_results.get(search_text, [])
            )
            grouped = {}
            for item in results:
                model_asset_id = item.get("model_asset_id")
                if not model_asset_id:
                    continue
                current = grouped.get(model_asset_id)
                if current is None or item["score"] > current["score"]:
                    candidate = {
                        "model_asset_id": model_asset_id,
                        "library_id": item.get("library_id"),
                        "site_id": item.get("site_id"),
                        "score": item["score"],
                        "match_type": "semantic",
                    }
                    for key in ("raw_vector_score", "vector_score", "rerank_score"):
                        if key in item:
                            candidate[key] = item.get(key)
                    grouped[model_asset_id] = candidate
            candidates = sorted(grouped.values(), key=lambda item: (-item["score"], item["model_asset_id"]))
            exact_ids = self._exact_asset_ids(search_text)
            if exact_ids:
                candidates_by_id = {item["model_asset_id"]: item for item in candidates}
                candidates = [
                    candidates_by_id.get(asset_id) or self._exact_candidate(asset_id)
                    for asset_id in exact_ids
                ] + [
                    item for item in candidates if item["model_asset_id"] not in set(exact_ids)
                ]
                candidates = _limit_candidates(candidates, top_k)
                if len(exact_ids) == 1:
                    top = candidates[0]
                    matches.append(
                        _asset_match(
                            mention_spec,
                            candidates,
                            "resolved",
                            top,
                            decision_source="exact_identifier",
                        )
                    )
                    continue
                matches.append(
                    _asset_match(
                        mention_spec,
                        candidates,
                        "ambiguous",
                        decision_prior="CLARIFY",
                        decision_source="multiple_exact_identifiers",
                    )
                )
                continue
            top_candidate_score = (
                float(candidates[0].get("score") or 0.0) if candidates else 0.0
            )
            plausible_candidates = [
                item
                for item in candidates
                if float(item.get("rerank_score") or 0.0)
                >= getattr(self, "identity_plausibility_floor", 0.75)
                and top_candidate_score - float(item.get("score") or 0.0)
                <= getattr(self, "identity_plausibility_max_score_gap", 0.10)
            ]
            if len(plausible_candidates) >= 2 and not strong_catalog_resolution:
                matches.append(
                    _asset_match(
                        mention_spec,
                        _limit_candidates(candidates, top_k),
                        "ambiguous",
                        decision_prior="CLARIFY",
                        decision_source="multiple_plausible_identity_candidates",
                    )
                )
                continue
            if plan_mention is not None:
                plan_decision = str(plan_mention.get("decision") or "UNRESOLVED").upper()
                plan_source = plan_mention.get("decision_source") or "catalog_constrained_llm"
                preferred_ids = [
                    plan_mention.get("selected_model_asset_id"),
                    *(plan_mention.get("candidate_model_asset_ids") or []),
                ]
                candidates = _prioritize_candidates(candidates, preferred_ids)
                if plan_decision == "CLARIFY":
                    plan_candidate_ids = {
                        str(item)
                        for item in plan_mention.get(
                            "candidate_model_asset_ids", []
                        )
                        if item
                    }
                    if len(plan_candidate_ids) < 2 and not generic_surface:
                        plan_decision = "INVALID_CLARIFY"
                    else:
                        matches.append(
                            _asset_match(
                                mention_spec,
                                _limit_candidates(candidates, top_k),
                                "ambiguous",
                                decision_prior="CLARIFY",
                                decision_source=plan_source,
                            )
                        )
                        continue
                if plan_decision == "INVALID_CLARIFY":
                    plan_mention = None
                elif plan_decision == "UNRESOLVED":
                    matches.append(
                        _asset_match(
                            mention_spec,
                            _limit_candidates(candidates, top_k),
                            "unresolved",
                            decision_prior="UNRESOLVED",
                            decision_source=plan_source,
                        )
                    )
                    continue
                if plan_mention is not None:
                    selected_id = plan_mention.get("selected_model_asset_id")
                    selected = next(
                        (item for item in candidates if item["model_asset_id"] == selected_id),
                        None,
                    )
                    if plan_decision != "RESOLVE" or selected is None:
                        matches.append(
                            _asset_match(
                                mention_spec,
                                _limit_candidates(candidates, top_k),
                                "unresolved",
                                decision_prior="UNRESOLVED",
                                decision_source="invalid_catalog_decision",
                            )
                        )
                        continue
                    candidates = _limit_candidates(candidates, top_k)
                    if selected["score"] < max(self.resolved_score, self.non_exact_resolved_score):
                        matches.append(
                            _asset_match(
                                mention_spec,
                                candidates,
                                "unresolved",
                                decision_source="catalog_decision_below_threshold",
                            )
                        )
                    else:
                        matches.append(
                            _asset_match(
                                mention_spec,
                                candidates,
                                "resolved",
                                selected,
                                decision_source=plan_source,
                            )
                        )
                    continue
            candidates = _limit_candidates(candidates, top_k)
            if not candidates or candidates[0]["score"] < self.resolved_score:
                matches.append(_asset_match(mention_spec, candidates, "unresolved"))
                continue
            if len(candidates) > 1 and _is_generic_asset_surface(search_text):
                matches.append(
                    _asset_match(
                        mention_spec,
                        candidates,
                        "ambiguous",
                        decision_prior="CLARIFY",
                        decision_source="generic_surface",
                    )
                )
                continue
            if candidates[0]["score"] < self.non_exact_resolved_score:
                matches.append(_asset_match(mention_spec, candidates, "unresolved"))
                continue
            second_score = candidates[1]["score"] if len(candidates) > 1 else -math.inf
            top = candidates[0]
            if top["score"] - second_score < self.ambiguity_margin:
                matches.append(_asset_match(mention_spec, candidates, "ambiguous"))
            else:
                matches.append(_asset_match(mention_spec, candidates, "resolved", top))
        return matches

    def _asset_identity_candidates(self, raw_hits):
        best_raw_by_asset = {}
        for raw in raw_hits:
            payload = raw.get("payload") or {}
            asset_id = payload.get("model_asset_id")
            if not asset_id:
                continue
            score = float(raw.get("score", 0.0))
            current = best_raw_by_asset.get(asset_id)
            if current is None or score > current["score"]:
                best_raw_by_asset[asset_id] = {
                    "score": score,
                    "payload": payload,
                }

        assets = self._assets()
        assets_by_id = {item["model_asset_id"]: item for item in assets}
        exhaustive = bool(assets) and len(assets) <= self.exhaustive_candidate_ceiling
        asset_ids = (
            [item["model_asset_id"] for item in assets]
            if exhaustive
            else list(best_raw_by_asset)
        )
        candidates = []
        for asset_id in asset_ids:
            asset = assets_by_id.get(asset_id)
            raw = best_raw_by_asset.get(asset_id)
            if asset is not None:
                candidate = _asset_summary_doc(asset, None)
                candidate["text"] = " ".join(
                    str(value)
                    for value in [
                        f"model_asset_id={asset_id}",
                        f"equipment_type={asset.get('equipment_type')}",
                        f"asset_name={asset.get('asset_name')}",
                        f"aliases={' | '.join(asset.get('aliases') or [])}",
                        f"category={asset.get('category')}",
                        f"domain_tags={' | '.join(derive_domain_tags(asset))}",
                    ]
                    if value and not value.endswith("=None")
                )
            elif raw is not None:
                candidate = _payload_to_doc(raw["payload"])
            else:
                continue
            candidate["vector_score"] = raw["score"] if raw is not None else 0.0
            if raw is not None:
                candidate["retrieved_evidence_id"] = raw["payload"].get("evidence_id")
                candidate["retrieved_field_path"] = raw["payload"].get("field_path")
            candidates.append(candidate)
        return sorted(
            candidates,
            key=lambda item: (-item["vector_score"], item["model_asset_id"]),
        )

    def _exact_asset_ids(self, query):
        return _exact_asset_ids_from_index(query, self._exact_identifier_index())

    def _exact_identifier_index(self):
        if hasattr(self, "_exact_identifier_index_cache"):
            return self._exact_identifier_index_cache
        self._exact_identifier_index_cache = _build_exact_identifier_index(self._assets())
        return self._exact_identifier_index_cache

    def _exact_candidate(self, asset_id):
        asset = next(item for item in self._assets() if item["model_asset_id"] == asset_id)
        return {
            "model_asset_id": asset_id,
            "library_id": asset.get("library_id"),
            "site_id": asset.get("site_id"),
            "score": 1.0,
            "match_type": "exact",
        }

    def _ensure_runtime_ready(self):
        if self.llm_client is None:
            raise RAGBlockedError(
                "missing_llm_client",
                "A configured LLM client is required; no fallback is allowed.",
            )
        if self.embedding_provider is None:
            raise RAGBlockedError(
                "missing_embedding_provider",
                "A BGE-M3 embedding provider is required; no fallback is allowed.",
            )
        if hasattr(self.embedding_provider, "ensure_ready"):
            self.embedding_provider.ensure_ready()
        if self.vector_store is None:
            raise RAGBlockedError(
                "missing_vector_store",
                "A Qdrant vector store is required; no fallback is allowed.",
            )

    def _domain_asset_matches(self, domain_intent, *, excluded_domains=None, top_k):
        domain = domain_intent.get("domain")
        if not domain:
            return []
        excluded_domains = set(excluded_domains or [])
        matches = []
        for asset in self._assets():
            tags = derive_domain_tags(asset)
            if domain not in tags:
                continue
            if excluded_domains & set(tags):
                continue
            matches.append(
                {
                    "model_asset_id": asset["model_asset_id"],
                    "library_id": asset.get("library_id"),
                    "site_id": asset.get("site_id"),
                    "asset_name": asset.get("asset_name"),
                    "equipment_type": asset.get("equipment_type"),
                    "domain_tags": tags,
                    "score": domain_intent.get("confidence", 0.8),
                    "match_type": "domain_intent",
                }
            )
        matches.sort(key=lambda item: (-item["score"], item["model_asset_id"]))
        return matches[:top_k]

    def _negative_domain_hits(self, query, domain_asset_matches):
        excluded_domains = infer_excluded_domains(query)
        if not excluded_domains:
            return []
        hits = []
        selected_ids = {item["model_asset_id"] for item in domain_asset_matches}
        for asset in self._assets():
            if asset["model_asset_id"] in selected_ids:
                continue
            tags = derive_domain_tags(asset)
            if any(domain in tags for domain in excluded_domains):
                hits.append(
                    {
                        "model_asset_id": asset["model_asset_id"],
                        "asset_name": asset.get("asset_name"),
                        "domain_tags": tags,
                        "score": 0.0,
                        "match_type": "excluded_domain",
                    }
                )
        return hits

    def _assets(self):
        if not hasattr(self, "_assets_cache"):
            self._assets_cache = _read_json(self.dataset_dir / "model_assets.json")
        return self._assets_cache

    def _decision_catalog(self):
        return [
            {
                "model_asset_id": asset["model_asset_id"],
                "asset_scope": asset.get("asset_scope"),
                "asset_name": asset.get("asset_name"),
                "asset_description": asset.get("asset_description"),
                "equipment_type": asset.get("equipment_type"),
                "aliases": asset.get("aliases") or [],
                "category": asset.get("category"),
                "process_capabilities": (
                    (asset.get("planning_fields") or {}).get(
                        "process_capabilities"
                    )
                    or []
                ),
            }
            for asset in self._assets()
        ]

    def _default_llm(self):
        return DeepSeekClient.from_env()

    def _default_embedding_provider(self):
        from tool.rag.embedding_providers import BGEM3EmbeddingProvider

        return BGEM3EmbeddingProvider.from_env()

    def _default_vector_store(self):
        from tool.rag.vector_store import QdrantVectorStore

        return QdrantVectorStore(self.index_dir)


def build_evidence_documents(dataset_dir):
    dataset_dir = Path(dataset_dir)
    assets = _read_json(dataset_dir / "model_assets.json")
    chunks = _read_jsonl(dataset_dir / "rag_chunks.jsonl")
    unresolved_items = _read_jsonl(dataset_dir / "unresolved_items.jsonl")
    chunk_by_asset = {}
    for chunk in chunks:
        asset_ids = chunk.get("model_asset_ids") or [chunk.get("model_asset_id")]
        for asset_id in asset_ids:
            if asset_id:
                chunk_by_asset[asset_id] = chunk

    docs = []
    for asset in assets:
        docs.append(_asset_summary_doc(asset, chunk_by_asset.get(asset["model_asset_id"])))
        docs.append(_asset_field_doc(asset, "geometry", "geometry"))
        docs.append(_asset_field_doc(asset, "extraction.extraction_quality", "extraction_quality"))
        for field_path in [
            "planning_fields.ports",
            "planning_fields.maintenance_clearance_m",
            "planning_fields.safety_clearance_m",
            "planning_fields.process_capabilities",
        ]:
            docs.append(_planning_field_doc(asset, field_path))

    for item in unresolved_items:
        docs.append(_unresolved_item_doc(item))
    return docs


def evaluate_cases(service, cases, *, top_k=5):
    results = []
    failures = []
    asset_cases = 0
    hit_at_1 = 0
    recall_at_3 = 0
    evidence_items = 0
    evidence_complete = 0
    unknown_false_positive = 0
    missing_field_hallucination = 0
    ambiguous_resolution_errors = 0
    flp_ir_status_errors = 0
    domain_cases = 0
    domain_precision_total = 0.0
    domain_recall_total = 0.0
    cross_domain_false_positive = 0
    layout_candidate_available = 0

    for case in cases:
        query_result = service.query(case["query"], top_k=top_k)
        case_result = {"case_id": case["case_id"], "status": query_result["status"], "query_result": query_result}
        results.append(case_result)
        for item in query_result.get("results", []):
            evidence_items += 1
            if item.get("evidence_id") and item.get("source_id") and item.get("field_path") and item.get("provenance"):
                evidence_complete += 1

        if case["case_type"] == "asset_lookup":
            asset_cases += 1
            ranked_asset_ids = _ranked_asset_ids(query_result)
            expected = case["expected_model_asset_id"]
            if ranked_asset_ids[:1] == [expected]:
                hit_at_1 += 1
            else:
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "failure_type": "asset_hit_at_1",
                        "expected": expected,
                        "actual": ranked_asset_ids[:1],
                    }
                )
            if expected in ranked_asset_ids[:3]:
                recall_at_3 += 1
            else:
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "failure_type": "asset_recall_at_3",
                        "expected": expected,
                        "actual": ranked_asset_ids[:3],
                    }
                )
        elif case["case_type"] == "unknown_asset":
            if any(match.get("selected_model_asset_id") for match in query_result.get("asset_matches", [])):
                unknown_false_positive += 1
                failures.append({"case_id": case["case_id"], "failure_type": "unknown_false_positive"})
        elif case["case_type"] == "missing_field":
            expected_field = case["expected_field_path"]
            if not _has_missing_field_evidence(query_result, expected_field):
                missing_field_hallucination += 1
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "failure_type": "missing_field_hallucination",
                        "expected_field_path": expected_field,
                    }
                )
        elif case["case_type"] == "ambiguous_asset":
            if not any(match.get("ambiguity_status") == "ambiguous" for match in query_result.get("asset_matches", [])):
                ambiguous_resolution_errors += 1
                failures.append({"case_id": case["case_id"], "failure_type": "ambiguous_asset_not_reported"})
        elif case["case_type"] == "flp_ir_status":
            from tool.representations.flp_ir_layer import run_flp_ir_pipeline

            expected_status = case["expected_validation_status"]
            pipeline_result = run_flp_ir_pipeline(case["query"], service.dataset_dir, rag_service=service)
            actual_status = pipeline_result.get("validation_report", {}).get("status")
            case_result["flp_ir_status"] = actual_status
            if actual_status != expected_status:
                flp_ir_status_errors += 1
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "failure_type": "flp_ir_status",
                        "expected": expected_status,
                        "actual": actual_status,
                    }
                )
        elif case["case_type"] in DOMAIN_CASE_TYPES:
            domain_cases += 1
            expected_domain = case["expected_domain"]
            excluded_domains = set(case.get("excluded_domains") or [])
            matches = query_result.get("domain_asset_matches", [])
            expected_ids = _asset_ids_for_domain(service.dataset_dir, expected_domain)
            matched_ids = [item.get("model_asset_id") for item in matches if item.get("model_asset_id")]
            if matches:
                layout_candidate_available += 1
            precision_denominator = len(matches[:top_k]) or 1
            precision = sum(
                1
                for item in matches[:top_k]
                if expected_domain in (item.get("domain_tags") or [])
            ) / precision_denominator
            recall = len(set(matched_ids[:top_k]) & expected_ids) / len(expected_ids) if expected_ids else 1.0
            domain_precision_total += precision
            domain_recall_total += recall
            false_hits = [
                item
                for item in matches[:top_k]
                if excluded_domains & set(item.get("domain_tags") or [])
            ]
            if false_hits:
                cross_domain_false_positive += len(false_hits)
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "failure_type": "cross_domain_false_positive",
                        "actual": [item.get("model_asset_id") for item in false_hits],
                    }
                )
            if query_result.get("domain_intent", {}).get("domain") != expected_domain:
                failures.append(
                    {
                        "case_id": case["case_id"],
                        "failure_type": "domain_intent",
                        "expected": expected_domain,
                        "actual": query_result.get("domain_intent", {}).get("domain"),
                    }
                )

    metrics = {
        "asset_hit_at_1": hit_at_1 / asset_cases if asset_cases else None,
        "asset_recall_at_3": recall_at_3 / asset_cases if asset_cases else None,
        "evidence_coverage": evidence_complete / evidence_items if evidence_items else 1.0,
        "unknown_false_positive": unknown_false_positive,
        "missing_field_hallucination": missing_field_hallucination,
        "ambiguous_resolution_errors": ambiguous_resolution_errors,
        "flp_ir_status_errors": flp_ir_status_errors,
        "domain_precision_at_k": domain_precision_total / domain_cases if domain_cases else None,
        "domain_recall_at_k": domain_recall_total / domain_cases if domain_cases else None,
        "cross_domain_false_positive_rate": (
            cross_domain_false_positive / domain_cases if domain_cases else None
        ),
        "layout_candidate_availability": (
            layout_candidate_available / domain_cases if domain_cases else None
        ),
        "default_site_success_rate": None,
        "default_constraint_success_rate": None,
    }
    return {"status": "OK", "metrics": metrics, "failures": failures, "case_results": results}


def _asset_summary_doc(asset, chunk):
    dims = asset["geometry"]["dimensions_m"]
    text = " ".join(
        str(part)
        for part in [
            asset["model_asset_id"],
            asset.get("asset_name"),
            asset.get("asset_description"),
            asset.get("equipment_type"),
            asset.get("category"),
            " ".join(asset.get("aliases", [])),
            f"dimensions {dims.get('width_m')}m x {dims.get('depth_m')}m x {dims.get('height_m')}m",
            chunk.get("retrieval_text") if chunk else "",
        ]
        if part
    )
    return _doc(
        evidence_id=f"evidence/model_asset/{_slug(asset['model_asset_id'])}/summary",
        source_type="model_asset",
        source_id=asset["model_asset_id"],
        field_path="asset_summary",
        text=text,
        structured_facts={
            "model_asset_id": asset["model_asset_id"],
            "asset_scope": asset.get("asset_scope"),
            "asset_name": asset.get("asset_name"),
            "asset_description": asset.get("asset_description"),
            "equipment_type": asset.get("equipment_type"),
            "category": asset.get("category"),
            "aliases": asset.get("aliases", []),
            "process_capabilities": (
                (asset.get("planning_fields") or {}).get(
                    "process_capabilities"
                )
                or []
            ),
            "geometry": asset.get("geometry"),
            "domain_tags": derive_domain_tags(asset),
        },
        provenance=asset.get("provenance", []),
        model_asset_id=asset["model_asset_id"],
        library_id=asset.get("library_id"),
        site_id=asset.get("site_id"),
        tags=["asset", "summary", asset.get("asset_scope") or "", *derive_domain_tags(asset)],
    )


def _asset_field_doc(asset, field_path, tag):
    value = _get_path(asset, field_path)
    return _doc(
        evidence_id=f"evidence/model_asset/{_slug(asset['model_asset_id'])}/{_slug(field_path)}",
        source_type="model_asset_field",
        source_id=asset["model_asset_id"],
        field_path=field_path,
        text=f"{asset['model_asset_id']} {asset.get('asset_name')} {field_path}: {json.dumps(value, ensure_ascii=False)}",
        structured_facts={field_path: value},
        provenance=asset.get("provenance", []),
        model_asset_id=asset["model_asset_id"],
        library_id=asset.get("library_id"),
        site_id=asset.get("site_id"),
        tags=["asset", tag, *derive_domain_tags(asset)],
    )


def _planning_field_doc(asset, field_path):
    value = _get_path(asset, field_path)
    missing = _is_missing_planning_value(value)
    source_type = "missing_planning_field" if missing else "model_asset_field"
    tags = ["planning_field", field_path.split(".")[-1], *derive_domain_tags(asset)]
    if missing:
        tags.append("missing")
    if missing:
        provenance = [
            {
                "field_path": field_path,
                "source_type": "asset_library",
                "source_id": asset["model_asset_id"],
                "source_field": field_path,
                "confidence": None,
                "note": "Field is empty or unresolved in the current dataset.",
            }
        ]
    else:
        provenance = [
            item
            for item in asset.get("provenance", [])
            if item.get("field_path") == field_path
        ] or [
            {
                "field_path": field_path,
                "source_type": "dataset_annotation",
                "source_id": asset["model_asset_id"],
                "source_field": field_path,
                "confidence": value.get("confidence") if isinstance(value, dict) else None,
                "note": "Field value is present in the current dataset.",
            }
        ]
    return _doc(
        evidence_id=f"evidence/model_asset/{_slug(asset['model_asset_id'])}/{_slug(field_path)}",
        source_type=source_type,
        source_id=asset["model_asset_id"],
        field_path=field_path,
        text=(
            f"{asset['model_asset_id']} {asset.get('asset_name')} {asset.get('equipment_type')} "
            f"{field_path} is {'missing' if missing else 'available'}: "
            f"{json.dumps(value, ensure_ascii=False)}"
        ),
        structured_facts={field_path: value, "is_missing": missing},
        provenance=provenance,
        model_asset_id=asset["model_asset_id"],
        library_id=asset.get("library_id"),
        site_id=asset.get("site_id"),
        tags=tags,
    )


def _unresolved_item_doc(item):
    field_path = item.get("field_path")
    model_asset_id = item.get("model_asset_id")
    return _doc(
        evidence_id=f"evidence/unresolved/{_slug(item['item_id'])}",
        source_type="unresolved_item",
        source_id=model_asset_id,
        field_path=field_path,
        text=f"{model_asset_id} unresolved {field_path}: {item.get('description')}",
        structured_facts=item,
        provenance=[
            {
                "field_path": field_path,
                "source_type": "asset_library",
                "source_id": model_asset_id,
                "source_field": field_path,
                "confidence": None,
                "note": item.get("description"),
            }
        ],
        model_asset_id=model_asset_id,
        library_id=None,
        site_id=None,
        tags=["unresolved", "missing", str(item.get("item_type") or "")],
    )


def derive_domain_tags(asset):
    text = _domain_text(asset)
    tags = [
        domain
        for domain, keywords in DOMAIN_KEYWORDS.items()
        if any(keyword.lower() in text for keyword in keywords)
    ]
    if not tags and (asset.get("asset_scope") == "equipment" or asset.get("category") == "factory_unit"):
        tags.append("general_factory")
    return sorted(set(tags or ["unknown"]))


def infer_domain_intent(query, extraction=None):
    text = _domain_text(
        {
            "asset_name": query,
            "equipment_type": " ".join((extraction or {}).get("asset_mentions") or []),
            "aliases": (extraction or {}).get("constraint_terms", []) + (extraction or {}).get("objective_terms", []),
        }
    )
    excluded_domains = set(infer_excluded_domains(query))
    domain_scores = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if domain in excluded_domains:
            continue
        score = sum(1 for keyword in keywords if keyword.lower() in text)
        if score:
            domain_scores.append((domain, score))
    if not domain_scores:
        return {"domain": None, "confidence": 0.0, "source_text": query}
    domain_scores.sort(key=lambda item: (-item[1], DOMAIN_PRIORITY.get(item[0], 50), item[0]))
    domain, score = domain_scores[0]
    confidence = min(0.95, 0.55 + 0.1 * score)
    return {"domain": domain, "confidence": confidence, "source_text": query}


def infer_excluded_domains(query):
    text = str(query or "").lower()
    domains = []
    for domain, keywords in DOMAIN_EXCLUSION_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            domains.append(domain)
            continue
        if any(marker in text for marker in ["不要", "排除", "without", "no ", "exclude"]):
            domain_keywords = DOMAIN_KEYWORDS.get(domain, [])
            if any(keyword.lower() in text for keyword in domain_keywords):
                domains.append(domain)
    return domains


def _domain_text(asset):
    values = [
        asset.get("asset_name", ""),
        asset.get("equipment_type", ""),
        asset.get("category", ""),
        asset.get("asset_scope", ""),
    ]
    values.extend(asset.get("aliases") or [])
    return " ".join(str(value).lower() for value in values if value)


def _asset_ids_for_domain(dataset_dir, domain):
    return {
        asset["model_asset_id"]
        for asset in _read_json(Path(dataset_dir) / "model_assets.json")
        if domain in derive_domain_tags(asset)
    }


def _doc(
    *,
    evidence_id,
    source_type,
    source_id,
    field_path,
    text,
    structured_facts,
    provenance,
    model_asset_id,
    library_id,
    site_id,
    tags,
):
    return {
        "evidence_id": evidence_id,
        "source_type": source_type,
        "source_id": source_id,
        "field_path": field_path,
        "text": text,
        "structured_facts": structured_facts,
        "provenance": provenance,
        "confidence": _confidence_from_provenance(provenance),
        "model_asset_id": model_asset_id,
        "library_id": library_id,
        "site_id": site_id,
        "tags": [tag for tag in tags if tag],
    }


def _canonical_sha256(items):
    ordered = sorted(items, key=lambda item: item.get("evidence_id", ""))
    payload = json.dumps(
        ordered,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_identifier(value):
    if not value:
        return None
    path = Path(str(value))
    return str(path.resolve()) if path.exists() else str(value)


def _apply_llm_rerank(candidates, ranked, *, vector_weight=0.7):
    vector_weight = max(0.0, min(1.0, float(vector_weight)))
    llm_weight = 1.0 - vector_weight
    by_id = {item["evidence_id"]: item for item in candidates}
    relevance_by_id = {
        item["evidence_id"]: item
        for item in ranked.get("ranked_evidence", [])
        if item.get("evidence_id") in by_id
    }
    results = []
    for evidence_id, candidate in by_id.items():
        llm_item = relevance_by_id.get(evidence_id, {})
        raw_vector_score = float(candidate.get("vector_score", 0.0))
        vector_score = max(0.0, min(1.0, raw_vector_score))
        llm_relevance = float(llm_item.get("relevance", 0.0))
        result = dict(candidate)
        result["raw_vector_score"] = raw_vector_score
        result["rerank_score"] = llm_relevance
        result["score"] = round(
            vector_weight * vector_score + llm_weight * llm_relevance,
            6,
        )
        result["llm_rationale"] = llm_item.get("rationale", "")
        results.append(result)
    return results


def _payload_to_doc(payload):
    return dict(payload)


_CONTEXT_ONLY_MENTIONS = {
    "this line",
    "that line",
    "the line",
    "line",
    "this area",
    "that area",
    "the area",
    "work area",
    "equipment area",
    "available floor area",
    "floor area",
    "main aisle",
    "the main aisle",
    "operator aisle",
    "service aisle",
    "main service aisle",
    "the aisle",
    "aisle",
    "available bay",
    "the bay",
    "utility area",
    "assembly area",
    "machining cell",
    "production cell",
    "revised production cell",
    "heavy equipment bay",
    "process flow",
    "the process",
    "production step",
    "required production step",
    "upstream step",
    "downstream step",
    "next step",
    "next position",
    "the next item",
    "last position",
    "last station",
    "nearby equipment",
    "adjacent equipment",
    "related equipment",
    "next machine",
    "the next machine",
    "raised platform",
    "another station",
    "station",
    "equipment",
    "machine",
    "machinery",
    "model",
    "work",
    "this machine",
    "that machine",
    "the first one",
    "the second one",
    "it",
    "this",
    "that",
    "them",
}

_CONTEXT_ONLY_PATTERN = re.compile(
    r"^(?:(?:the|this|that|a|an) )?"
    r"(?:(?:main|service|operator|nearby|adjacent|related|revised|available|next|last|"
    r"heavy equipment|production|assembly|machining|work|utility) )+"
    r"(?:aisle|walkway|area|bay|cell|flow|step|position|station|item|work)$"
)
_FUNCTIONAL_MENTION_PATTERN = re.compile(
    r"\b(?:equipment intended to|equipment that can|a system used to)\b[^.!?;\n]*",
    re.I,
)
_FUNCTIONAL_CONTEXT_SUFFIX = re.compile(
    r"\s+beside the aisle without crowding (?:adjacent|nearby) equipment\s*$",
    re.I,
)


def _repair_asset_mentions(query, mentions):
    """Keep a functional equipment request as one mention instead of its subparts."""
    query_text = str(query)
    functional_spans = []
    for match in _FUNCTIONAL_MENTION_PATTERN.finditer(query_text):
        surface = _FUNCTIONAL_CONTEXT_SUFFIX.sub("", match.group(0)).strip(" ,")
        end = match.start() + len(surface)
        functional_spans.append((match.start(), end, surface))
    if not functional_spans:
        return list(mentions)

    repaired = [(start, surface) for start, _end, surface in functional_spans]
    query_lower = query_text.lower()
    for index, value in enumerate(mentions):
        surface = str(value or "").strip()
        if not surface:
            continue
        start = query_lower.find(surface.lower())
        end = start + len(surface) if start >= 0 else None
        overlaps_functional = start >= 0 and any(
            start < functional_end and end > functional_start
            for functional_start, functional_end, _surface in functional_spans
        )
        is_functional_component = any(
            _normalize_match_text(surface) in _normalize_match_text(functional_surface)
            for _start, _end, functional_surface in functional_spans
        )
        if overlaps_functional or is_functional_component:
            continue
        repaired.append((start if start >= 0 else len(query_text) + index, surface))
    repaired.sort(key=lambda item: (item[0], item[1].lower()))
    return [surface for _start, surface in repaired]


def _unique_coreferential_surfaces(mentions):
    unique = []
    seen = set()
    for value in mentions:
        surface = str(value or "").strip()
        normalized = _normalize_match_text(surface)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(surface)
    return unique


def _prepare_asset_mentions(query, mentions, *, allow_context=False):
    prepared = []
    rejected = []
    next_start_by_surface = {}
    query_text = str(query)
    query_lower = query_text.lower()
    for index, value in enumerate(mentions):
        surface = str(value or "").strip()
        if not surface:
            continue
        normalized = _normalize_match_text(surface)
        if not allow_context and (
            normalized in _CONTEXT_ONLY_MENTIONS
            or _CONTEXT_ONLY_PATTERN.fullmatch(normalized)
        ):
            rejected.append({"surface_text": surface, "reason": "context_only"})
            continue
        search_start = next_start_by_surface.get(surface.lower(), 0)
        start = query_lower.find(surface.lower(), search_start)
        if start < 0:
            start = query_lower.find(surface.lower())
        if start >= 0:
            end = start + len(surface)
            next_start_by_surface[surface.lower()] = end
            mention_id = f"m_{start}_{end}"
        else:
            start = None
            end = None
            mention_id = f"m_pred_{index:02d}"
        prepared.append(
            {
                "mention_id": mention_id,
                "surface_text": surface,
                "start_char": start,
                "end_char": end,
            }
        )
    prepared, nested_rejected = _suppress_nested_mentions(prepared)
    rejected.extend(nested_rejected)
    return prepared, rejected


def _suppress_nested_mentions(mentions):
    retained = []
    rejected = []
    for index, mention in enumerate(mentions):
        start = mention.get("start_char")
        end = mention.get("end_char")
        container = next(
            (
                other
                for other_index, other in enumerate(mentions)
                if other_index != index
                and start is not None
                and end is not None
                and other.get("start_char") is not None
                and other.get("end_char") is not None
                and other["start_char"] <= start
                and other["end_char"] >= end
                and (
                    other["start_char"] < start
                    or other["end_char"] > end
                )
            ),
            None,
        )
        if container is None:
            retained.append(mention)
            continue
        rejected.append(
            {
                "surface_text": mention.get("surface_text", ""),
                "reason": "nested_in_longer_mention",
                "container_surface_text": container.get("surface_text", ""),
            }
        )
    return retained, rejected


def _matching_plan_mention(surface_text, plan_mentions, used_indexes):
    normalized = _normalize_match_text(surface_text)
    fallback = None
    for index, item in enumerate(plan_mentions):
        if index in used_indexes:
            continue
        plan_surface = _normalize_match_text(item.get("surface_text") or "")
        if not plan_surface:
            continue
        if plan_surface == normalized:
            return item, index
        if normalized in plan_surface or plan_surface in normalized:
            fallback = fallback or (item, index)
    return fallback or (None, None)


def _prioritize_candidates(candidates, preferred_ids):
    preferred = []
    for value in preferred_ids:
        if value and value not in preferred:
            preferred.append(str(value))
    priority = {asset_id: index for index, asset_id in enumerate(preferred)}
    return sorted(
        candidates,
        key=lambda item: (
            priority.get(item["model_asset_id"], len(priority)),
            -item["score"],
            item["model_asset_id"],
        ),
    )


def _limit_candidates(candidates, top_k):
    return candidates[: int(top_k)] if top_k is not None else candidates


def _asset_match(
    mention_spec,
    candidates,
    status,
    selected=None,
    *,
    decision_prior=None,
    decision_source=None,
):
    match = {
        "mention_id": mention_spec.get("mention_id"),
        "surface_text": mention_spec.get("surface_text", ""),
        "start_char": mention_spec.get("start_char"),
        "end_char": mention_spec.get("end_char"),
        "candidate_model_asset_ids": candidates,
        "selected_model_asset_id": selected.get("model_asset_id") if selected else None,
        "selected_library_id": selected.get("library_id") if selected else None,
        "selected_site_id": selected.get("site_id") if selected else None,
        "ambiguity_status": status,
    }
    if decision_prior:
        match["decision_prior"] = decision_prior
    if decision_source:
        match["decision_source"] = decision_source
    return match


def _ranked_asset_ids(query_result):
    ids = []
    for item in query_result.get("results", []):
        asset_id = item.get("model_asset_id")
        if asset_id and asset_id not in ids:
            ids.append(asset_id)
    return ids


def _is_generic_asset_surface(surface_text):
    normalized = _normalize_match_text(surface_text)
    normalized = re.sub(
        r"^(?:(?:a|an|the|requested|additional)\s+)+", "", normalized
    )
    return normalized in {
        "asset",
        "automated assembly line",
        "bulk material hopper",
        "cnc milling machine",
        "control cabinet",
        "device",
        "electrical power unit",
        "enclosed machining cell",
        "equipment",
        "equipment cabinet",
        "fluid handling unit",
        "forming press",
        "industrial power system",
        "industrial press",
        "inspection equipment",
        "inspection machine",
        "inspection station",
        "lifting equipment",
        "load handling unit",
        "machine",
        "machining center",
        "machining equipment",
        "material feed hopper",
        "material hopper",
        "packaging machine",
        "packaging station",
        "packaging system",
        "power distribution equipment",
        "power equipment",
        "press machine",
        "process fluid machine",
        "process tank",
        "process vessel",
        "pump or compressor",
        "service platform",
        "spark erosion machine",
        "storage vessel",
        "test equipment",
        "testing equipment",
        "detection equipment",
        "检测设备",
        "檢測設備",
        "测试设备",
        "測試設備",
        "设备",
        "設備",
        "装置",
        "裝置",
        "机器",
        "機器",
    }


def _has_exact_asset_surface_match(surface_text, candidates, results_by_id):
    normalized = _normalize_match_text(surface_text)
    if not normalized:
        return False
    for candidate in candidates:
        for item in results_by_id.values():
            if item.get("model_asset_id") != candidate.get("model_asset_id"):
                continue
            if normalized in _normalized_asset_identifiers(item):
                return True
    return False


def _normalized_asset_identifiers(item):
    values = [
        item.get("source_id"),
        item.get("model_asset_id"),
    ]
    facts = item.get("structured_facts") or {}
    values.extend(
        [
            facts.get("model_asset_id"),
            facts.get("asset_name"),
            facts.get("equipment_type"),
        ]
    )
    values.extend(facts.get("aliases") or [])
    return {_normalize_match_text(value) for value in values if value}


def _build_exact_identifier_index(assets):
    asset_ids_by_identifier = defaultdict(set)

    def add_identifier(value, asset_id, *, allow_plural=False):
        normalized = _normalize_match_text(str(value).replace("_", " "))
        if len(normalized) < 4:
            return
        asset_ids_by_identifier[normalized].add(asset_id)
        if allow_plural:
            plural = _regular_english_plural_identifier(normalized)
            if plural and plural != normalized:
                asset_ids_by_identifier[plural].add(asset_id)

    for asset in assets:
        asset_id = asset["model_asset_id"]
        add_identifier(asset_id, asset_id)
        human_identifiers = [
            asset.get("asset_name"),
            asset.get("equipment_type"),
            *(asset.get("aliases") or []),
        ]
        for value in human_identifiers:
            if not value:
                continue
            add_identifier(
                value,
                asset_id,
                allow_plural=not _looks_like_model_code(value),
            )
            for code in re.findall(r"[A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*", str(value)):
                add_identifier(code, asset_id)
    return {
        identifier: next(iter(asset_ids))
        for identifier, asset_ids in asset_ids_by_identifier.items()
        if len(asset_ids) == 1
    }


def _exact_asset_ids_from_index(query, identifier_index):
    normalized_query = _normalize_match_text(query)
    purpose = re.search(r"\bfor\s+(?:transferring|transporting|carrying|loading|unloading|serving|feeding|handling|moving)\b", normalized_query)
    head_end = purpose.start() if purpose else len(normalized_query)
    return sorted(
        {
            asset_id
            for identifier, asset_id in identifier_index.items()
            if (match := re.search(rf"\b{re.escape(identifier)}\b", normalized_query))
            and (match.start() < head_end or _looks_like_model_code(identifier))
        }
    )


def _looks_like_model_code(value):
    raw = str(value or "").strip()
    return bool(
        "/" in raw
        or re.fullmatch(
            r"[A-Za-z]+(?:[_-][A-Za-z]+)*[_-]?\d+[A-Za-z0-9_-]*",
            raw,
        )
    )


def _regular_english_plural_identifier(identifier):
    words = identifier.split()
    if not words or not re.fullmatch(r"[a-z]+", words[-1]):
        return None
    word = words[-1]
    if word.endswith(("s", "x", "z", "ch", "sh")):
        plural = word + "es"
    elif len(word) > 1 and word.endswith("y") and word[-2] not in "aeiou":
        plural = word[:-1] + "ies"
    else:
        plural = word + "s"
    return " ".join([*words[:-1], plural])


def _normalize_match_text(value):
    return " ".join(
        "".join(char.lower() if char.isalnum() else " " for char in str(value)).split()
    )


def _has_missing_field_evidence(query_result, expected_field):
    for item in query_result.get("results", []):
        if item.get("field_path") == expected_field and item.get("source_type") in {
            "missing_planning_field",
            "unresolved_item",
        }:
            return True
    return False


def _is_missing_planning_value(value):
    if value in (None, [], {}):
        return True
    if isinstance(value, dict):
        meaningful = [
            item
            for key, item in value.items()
            if key not in {"source", "confidence"} and item is not None
        ]
        return not meaningful
    return False


def _get_path(data, field_path):
    current = data
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _confidence_from_provenance(provenance):
    values = [item.get("confidence") for item in provenance if isinstance(item.get("confidence"), (int, float))]
    return max(values) if values else None


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _slug(value):
    return "".join(char.lower() if char.isalnum() else "_" for char in str(value)).strip("_")


def _load_cases(path):
    return _read_jsonl(path)


def _progress(callback, message):
    if callback is not None:
        callback(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Engineering RAG index/query/evaluate CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--dataset-dir", default="rag_dataset")
    index_parser.add_argument("--index-dir", default=str(Path(".rag_index") / "engineering_rag_qdrant"))

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("--dataset-dir", default="rag_dataset")
    query_parser.add_argument("--index-dir", default=str(Path(".rag_index") / "engineering_rag_qdrant"))
    query_parser.add_argument("--query", required=True)
    query_parser.add_argument("--top-k", type=int, default=5)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--dataset-dir", default="rag_dataset")
    eval_parser.add_argument("--index-dir", default=str(Path(".rag_index") / "engineering_rag_qdrant"))
    eval_parser.add_argument("--cases", required=True)
    eval_parser.add_argument("--output", default=None)
    eval_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args(argv)
    try:
        service = EngineeringRAGService(args.dataset_dir, index_dir=args.index_dir)
        if args.command == "index":
            result = service.index(progress=lambda message: print(f"[engineering-rag] {message}", file=sys.stderr))
        elif args.command == "query":
            result = service.query(args.query, top_k=args.top_k)
        else:
            result = evaluate_cases(service, _load_cases(args.cases), top_k=args.top_k)
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except RAGBlockedError as exc:
        print(json.dumps(exc.to_report(), ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
