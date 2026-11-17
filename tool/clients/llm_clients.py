import json
import os
import threading
import time
from email.utils import parsedate_to_datetime


def _env_value(env, name):
    value = env.get(name)
    if value:
        return value
    if env is not os.environ or os.name != "nt":
        return value
    return _windows_environment_value(name)


def _windows_environment_value(name):
    try:
        import winreg
    except ImportError:
        return None

    locations = [
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    ]
    for root, subkey in locations:
        try:
            with winreg.OpenKey(root, subkey) as key:
                value, _value_type = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            return str(value)
    return None


DEEPSEEK_API_KEY = _env_value(os.environ, "DEEPSEEK_API_KEY")
DEFAULT_RERANK_MAX_OUTPUT_TOKENS = 32768
PLANNING_SLOT_INSTRUCTIONS = (
    " Equipment declarations may additionally contain instance_labels (the exact user-supplied names) "
    "and planning_overrides. Preserve every requested equipment declaration, including unresolved ones. "
    "planning_overrides contains only explicitly stated user planning requirements: "
    "maintenance_clearance_m or safety_clearance_m as {front_m, rear_m, left_m, right_m}, "
    "allowed_orientations_deg as a list of numeric yaw angles, and source_text quoting the exact "
    "requirement span. Convert stated length units to meters. Apply an explicit all-equipment "
    "requirement to each declaration. A stated single initial yaw for a new generated layout must "
    "be preserved as its singleton allowed_orientations_deg. Omit unstated overrides; never infer "
    "clearances or orientations. Preserve these optional fields during grounding and revision. "
    "Constraint items may additionally include parameters for explicitly stated numeric requirements. "
)


class RAGBlockedError(RuntimeError):
    def __init__(self, blocker_type, message, *, details=None, suggestions=None):
        super().__init__(message)
        self.blocker_type = blocker_type
        self.message = message
        self.details = details or {}
        self.suggestions = suggestions or []

    def to_report(self):
        return {
            "status": "BLOCKED",
            "blocker_type": self.blocker_type,
            "message": self.message,
            "details": self.details,
            "suggestions": self.suggestions,
        }


def _llm_environment_config(env=None, *, model=None, base_url=None, intent=False):
    runtime_env = os.environ if env is None else env
    shared = {
        name: _env_value(runtime_env, name)
        for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
    }
    if intent:
        model = model or _env_value(runtime_env, "FLP_INTENT_LLM_MODEL")
        base_url = base_url or _env_value(runtime_env, "FLP_INTENT_LLM_BASE_URL")
    if any(shared.values()):
        config = {
            "LLM_API_KEY": shared["LLM_API_KEY"],
            "LLM_BASE_URL": base_url or shared["LLM_BASE_URL"],
            "LLM_MODEL": model or shared["LLM_MODEL"],
        }
        missing = [name for name, value in config.items() if not value]
        if missing:
            raise RAGBlockedError(
                "missing_llm_configuration",
                "Configure LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL together.",
                details={"missing_env_vars": missing},
                suggestions=["Use .env.example as a reference and set the variables in your shell."],
            )
        # Keep provider credentials together: partial generic settings must not
        # inherit a saved DeepSeek key or send another provider's key to DeepSeek.
        return config["LLM_API_KEY"], config["LLM_MODEL"], config["LLM_BASE_URL"]
    return (
        DEEPSEEK_API_KEY if env is None else _env_value(runtime_env, "DEEPSEEK_API_KEY"),
        model or _env_value(runtime_env, "DEEPSEEK_MODEL"),
        base_url or _env_value(runtime_env, "DEEPSEEK_BASE_URL"),
    )


class DeepSeekClient:
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-v4-pro"

    RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key,
        *,
        model=None,
        base_url=None,
        timeout_s=60,
        max_tokens=None,
        rerank_candidate_text_chars=1200,
        rerank_candidate_limit=20,
        rerank_require_rationale=True,
        max_connections=4,
        max_attempts=4,
        backoff_base_s=1.0,
        sleep_fn=time.sleep,
        thinking=None,
    ):
        if not api_key:
            raise RAGBlockedError(
                "missing_deepseek_api_key",
                "An LLM API key is required for Engineering RAG.",
                suggestions=["Configure LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL, or use legacy DEEPSEEK_API_KEY settings."],
            )
        self._api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = int(max_tokens) if max_tokens is not None else None
        self.rerank_candidate_text_chars = int(rerank_candidate_text_chars)
        self.rerank_candidate_limit = int(rerank_candidate_limit)
        self.rerank_require_rationale = bool(rerank_require_rationale)
        self.max_connections = int(max_connections)
        self.max_attempts = int(max_attempts)
        self.backoff_base_s = float(backoff_base_s)
        self._sleep = sleep_fn
        self.thinking = dict(thinking) if isinstance(thinking, dict) else thinking
        self._client = None
        self._client_lock = threading.Lock()
        self._trace = threading.local()

    @classmethod
    def from_env(
        cls,
        *,
        env=None,
        model=None,
        base_url=None,
        timeout_s=60,
        max_tokens=None,
        rerank_candidate_text_chars=1200,
        rerank_candidate_limit=20,
        rerank_require_rationale=True,
        max_connections=4,
        max_attempts=4,
        backoff_base_s=1.0,
        thinking=None,
    ):
        api_key, model, base_url = _llm_environment_config(env, model=model, base_url=base_url)
        if not api_key:
            raise RAGBlockedError(
                "missing_deepseek_api_key",
                "An LLM API key is required for Engineering RAG.",
                details={"env_var": "DEEPSEEK_API_KEY"},
                suggestions=["Configure LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL, or use legacy DEEPSEEK_API_KEY settings."],
            )
        return cls(
            api_key,
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            max_tokens=max_tokens,
            rerank_candidate_text_chars=rerank_candidate_text_chars,
            rerank_candidate_limit=rerank_candidate_limit,
            rerank_require_rationale=rerank_require_rationale,
            max_connections=max_connections,
            max_attempts=max_attempts,
            backoff_base_s=backoff_base_s,
            thinking=thinking,
        )

    def begin_request_trace(self):
        self._trace.events = []

    def end_request_trace(self):
        events = list(getattr(self._trace, "events", []))
        self._trace.events = []
        return {
            "calls": len(events),
            "attempts": sum(item["attempts"] for item in events),
            "retry_count": sum(item["attempts"] - 1 for item in events),
            "cumulative_backoff_ms": round(
                sum(item["cumulative_backoff_ms"] for item in events), 3
            ),
            "status_categories": [
                category
                for item in events
                for category in item["status_categories"]
            ],
            "purposes": [item["purpose"] for item in events],
        }

    def close(self):
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()

    def extract_query(self, query):
        prompt = (
            "Extract engineering RAG query facts as strict JSON. "
            "Return keys: normalized_query, asset_mentions, constraint_terms, "
            "objective_terms, missing_field_terms, language. "
            "asset_mentions must contain only explicit tangible industrial equipment or model-asset "
            "references, in source order, including unknown, generic, ambiguous, misspelled, and "
            "repeated equipment references. Copy each mention verbatim from the query. Exclude "
            "pronouns, deictic references, locations, process positions, constraints, objectives, "
            "and requested fields. For example, in 'put the pump on this line near the aisle', "
            "asset_mentions is ['the pump']; 'this line' and 'the aisle' are not assets. A typed "
            "phrase such as 'rotary assembly line' is an asset, while bare 'this line' is not. "
            "Do not normalize, merge, infer, or omit actual equipment references. "
            "Do not invent asset IDs or engineering facts."
        )
        data = self._chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
            purpose="query_extraction",
        )
        return {
            "normalized_query": str(data.get("normalized_query") or query),
            "asset_mentions": _string_list(data.get("asset_mentions")),
            "constraint_terms": _string_list(data.get("constraint_terms")),
            "objective_terms": _string_list(data.get("objective_terms")),
            "missing_field_terms": _string_list(data.get("missing_field_terms")),
            "language": str(data.get("language") or "unknown"),
        }

    def rerank(self, query, candidates):
        compact_candidates = [
            {
                "evidence_id": item["evidence_id"],
                "model_asset_id": item.get("model_asset_id"),
                "source_id": item.get("source_id"),
                "field_path": item.get("field_path"),
                "text": item.get("text", "")[: self.rerank_candidate_text_chars],
            }
            for item in candidates[: self.rerank_candidate_limit]
        ]
        item_schema = (
            "{evidence_id, relevance, rationale}"
            if self.rerank_require_rationale
            else "{evidence_id, relevance}"
        )
        prompt = (
            "Rerank Engineering RAG evidence as strict JSON. Return keys: "
            "ranked_evidence, identity_decision, and summary. ranked_evidence must be a list of "
            f"{item_schema}. relevance is 0..1. Return at most one row for each supplied "
            "evidence_id, no more rows than supplied candidates, and no IDs outside that set. "
            "Keep each rationale to a short phrase of at most 12 words. Never copy evidence text "
            "or source passages into rationale or summary; keep summary to at most 20 words. "
            "identity_decision is {decision, selected_model_asset_id, candidate_model_asset_ids, confidence}; "
            "decision must be RESOLVE, CLARIFY, or UNRESOLVED. Use only supplied model_asset_id values. "
            "For RESOLVE, select exactly one supported ID. For CLARIFY, list every genuinely plausible ID. "
            "For UNRESOLVED, select no ID. "
            "Judge whether each candidate is the same equipment requested by the query, not merely "
            "from the same factory domain. Use 0.90-1.00 for an exact identity or clear functional "
            "paraphrase, 0.60-0.89 for a plausible but incomplete identity match, 0.35-0.59 for a "
            "weak or ambiguous match, and 0.00-0.34 for a different asset. If the requested asset "
            "is absent from the catalog, do not raise the nearest related asset above 0.34. Give "
            "multiple genuinely valid interpretations comparable scores. Only judge relevance to "
            "the query; do not add facts not present in evidence."
        )
        data = self._chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"query": query, "candidates": compact_candidates},
                        ensure_ascii=False,
                    ),
                },
            ],
            purpose="evidence_rerank",
            max_tokens=(
                self.max_tokens
                if self.max_tokens is not None
                else DEFAULT_RERANK_MAX_OUTPUT_TOKENS
            ),
        )
        ranked = []
        valid_evidence_ids = {
            str(item["evidence_id"])
            for item in compact_candidates
            if item.get("evidence_id")
        }
        seen_evidence_ids = set()
        for item in data.get("ranked_evidence", []):
            if not isinstance(item, dict) or "evidence_id" not in item:
                continue
            evidence_id = str(item["evidence_id"])
            if (
                evidence_id not in valid_evidence_ids
                or evidence_id in seen_evidence_ids
                or len(ranked) >= len(valid_evidence_ids)
            ):
                continue
            seen_evidence_ids.add(evidence_id)
            ranked.append(
                {
                    "evidence_id": evidence_id,
                    "relevance": _bounded_float(item.get("relevance")),
                    "rationale": str(item.get("rationale") or "")[:512],
                }
            )
        identity_decision = None
        raw_decision = data.get("identity_decision")
        if isinstance(raw_decision, dict):
            valid_ids = {
                str(item["model_asset_id"])
                for item in compact_candidates
                if item.get("model_asset_id")
            }
            decision = str(raw_decision.get("decision") or "UNRESOLVED").upper()
            if decision not in {"RESOLVE", "CLARIFY", "UNRESOLVED"}:
                decision = "UNRESOLVED"
            selected = (
                str(raw_decision["selected_model_asset_id"])
                if raw_decision.get("selected_model_asset_id") in valid_ids
                else None
            )
            candidate_ids = []
            seen_candidate_ids = set()
            for item in _string_list(raw_decision.get("candidate_model_asset_ids")):
                if item in valid_ids and item not in seen_candidate_ids:
                    candidate_ids.append(item)
                    seen_candidate_ids.add(item)
            if selected and selected not in candidate_ids:
                candidate_ids.insert(0, selected)
            if decision == "RESOLVE" and selected is None:
                decision = "UNRESOLVED"
            if decision != "RESOLVE":
                selected = None
            identity_decision = {
                "decision": decision,
                "selected_model_asset_id": selected,
                "candidate_model_asset_ids": candidate_ids,
                "confidence": _bounded_float(raw_decision.get("confidence")),
            }
        return {
            "ranked_evidence": ranked,
            "identity_decision": identity_decision,
            "summary": str(data.get("summary") or "")[:1000],
        }

    def decide_grounding(self, query, *, catalog=None):
        context_mode = "full_catalog" if catalog is not None else "parametric_only"
        prompt = (
            "Ground English industrial equipment references as strict JSON. Return keys: mentions, fields. "
            "Each mention is {surface_text, decision, selected_model_asset_id, candidate_model_asset_ids, "
            "confidence, evidence}. decision must be RESOLVE, CLARIFY, or UNRESOLVED. "
            "Return exactly the explicit tangible equipment/model-asset mentions in source order and copy "
            "surface_text verbatim from the query. Exclude pronouns, deictic references, locations, process "
            "positions, constraints, objectives, and requested field names. Return one entity mention per "
            "intended equipment instance: merge later coreferential repetitions that refer to an already "
            "introduced instance, but keep explicitly distinct repeated instances separate. "
            "Each field is {mention_surface_text, model_asset_id, field_path, decision, value, confidence, evidence}; "
            "mention_surface_text must copy the equipment mention to which that requested field applies. For a "
            "global request about each machine, return one field record per applicable equipment mention. Even "
            "when the identity-only catalog omits engineering values, preserve the requested field path and its "
            "mention_surface_text with decision UNRESOLVED. field decision "
            "must be ANSWER or UNRESOLVED. Evidence items are {source_id, field_path}. "
            "Use only model_asset_id values and facts explicitly present in the supplied catalog. "
            "Catalog entries may distinguish standalone equipment from composite scenes through asset_scope, "
            "and may include asset_description or process_capabilities. Treat those fields as catalog evidence, "
            "but do not infer a function that the catalog does not state. "
            "If no catalog is supplied, do not invent opaque asset IDs or engineering values. "
            "A broad reference with multiple valid catalog items must be CLARIFY and candidate_model_asset_ids "
            "must contain every genuinely plausible catalog item. An absent asset or field "
            "must be UNRESOLVED. Do not force a nearest match."
        )
        payload = {"query": query, "context_mode": context_mode}
        if catalog is not None:
            payload["catalog"] = catalog
        data = self._chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose=f"grounding_{context_mode}",
        )
        valid_catalog_ids = {
            str(item["model_asset_id"])
            for item in (catalog or [])
            if isinstance(item, dict) and item.get("model_asset_id")
        }
        mentions = []
        for item in data.get("mentions", []):
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision") or "UNRESOLVED").upper()
            if decision not in {"RESOLVE", "CLARIFY", "UNRESOLVED"}:
                decision = "UNRESOLVED"
            selected = (
                str(item["selected_model_asset_id"])
                if item.get("selected_model_asset_id")
                and str(item["selected_model_asset_id"]) in valid_catalog_ids
                else None
            )
            candidate_ids = [
                asset_id
                for asset_id in _string_list(item.get("candidate_model_asset_ids"))
                if asset_id in valid_catalog_ids
            ]
            if selected and selected not in candidate_ids:
                candidate_ids.insert(0, selected)
            if decision == "RESOLVE" and selected is None:
                decision = "UNRESOLVED"
            if decision != "RESOLVE":
                selected = None
            mentions.append(
                {
                    "surface_text": str(item.get("surface_text") or ""),
                    "decision": decision,
                    "selected_model_asset_id": selected,
                    "candidate_model_asset_ids": candidate_ids,
                    "confidence": _bounded_float(item.get("confidence")),
                    "evidence": _evidence_refs(item.get("evidence")),
                }
            )
        fields = []
        for item in data.get("fields", []):
            if not isinstance(item, dict):
                continue
            decision = str(item.get("decision") or "UNRESOLVED").upper()
            if decision not in {"ANSWER", "UNRESOLVED"}:
                decision = "UNRESOLVED"
            fields.append(
                {
                    "mention_surface_text": str(item.get("mention_surface_text") or "") or None,
                    "model_asset_id": (
                        str(item["model_asset_id"])
                        if item.get("model_asset_id")
                        and str(item["model_asset_id"]) in valid_catalog_ids
                        else None
                    ),
                    "field_path": str(item.get("field_path") or "") or None,
                    "decision": decision,
                    "value": item.get("value"),
                    "confidence": _bounded_float(item.get("confidence")),
                    "evidence": _evidence_refs(item.get("evidence")),
                }
            )
        return {"mentions": mentions, "fields": fields}

    def _chat_json(self, messages, *, purpose, max_tokens=None):
        try:
            import httpx
        except ImportError as exc:
            raise RAGBlockedError(
                "missing_dependency",
                "httpx is required to call the configured LLM service.",
                details={"dependency": "httpx", "purpose": purpose},
                suggestions=["Install dependencies with pip before running RAG commands."],
            ) from exc

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        max_output_tokens = self.max_tokens if max_tokens is None else int(max_tokens)
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        attempts = 0
        cumulative_backoff_s = 0.0
        invalid_json_retried = False
        last_status = None
        last_category = "unknown_error"
        last_error_payload = None
        status_categories = []
        while True:
            attempts += 1
            try:
                response = self._http_client(httpx).post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                last_status = response.status_code
                if response.status_code >= 400:
                    try:
                        last_error_payload = response.json()
                    except Exception:
                        response_text = getattr(response, "text", None)
                        last_error_payload = (
                            str(response_text)[:500] if response_text is not None else None
                        )
                    if (
                        response.status_code in self.RETRYABLE_STATUS_CODES
                        and attempts < self.max_attempts
                    ):
                        last_category = _http_status_category(response.status_code)
                        status_categories.append(last_category)
                        delay = self._retry_delay(response, attempts)
                        cumulative_backoff_s += delay
                        self._sleep(delay)
                        continue
                    last_category = _http_status_category(response.status_code)
                    status_categories.append(last_category)
                    response.raise_for_status()
                content = None
                finish_reason = None
                try:
                    body = response.json()
                    choice = body["choices"][0]
                    content = choice["message"]["content"]
                    finish_reason = choice.get("finish_reason")
                    if finish_reason == "length":
                        last_category = "output_truncated"
                        status_categories.append(last_category)
                        raise RAGBlockedError(
                            "llm_output_truncated",
                            "The configured LLM reached its output-token limit before returning complete JSON.",
                            details={
                                "purpose": purpose,
                                "model": self.model,
                                "http_status": last_status,
                                "attempts": attempts,
                                "finish_reason": finish_reason,
                                "content_length": len(content) if isinstance(content, str) else None,
                                "max_tokens": max_output_tokens,
                            },
                            suggestions=[
                                "Increase the output-token budget or shorten the requested JSON response before retrying."
                            ],
                        )
                    result = _parse_json_object(content)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    last_category = "invalid_json"
                    status_categories.append(last_category)
                    if not invalid_json_retried:
                        invalid_json_retried = True
                        continue
                    raise RAGBlockedError(
                        "llm_invalid_json",
                        "The configured LLM returned invalid JSON after one JSON retry.",
                        details={
                            "purpose": purpose,
                            "model": self.model,
                            "http_status": last_status,
                            "attempts": attempts,
                            "finish_reason": finish_reason,
                            "content_length": len(content) if isinstance(content, str) else None,
                            "content_preview": content[:240] if isinstance(content, str) else None,
                        },
                        suggestions=["Verify that the selected model supports JSON object responses."],
                    ) from exc
                self._record_trace(
                    purpose,
                    attempts,
                    status_categories + ["success"],
                    cumulative_backoff_s,
                )
                return result
            except RAGBlockedError:
                self._record_trace(
                    purpose,
                    attempts,
                    status_categories or [last_category],
                    cumulative_backoff_s,
                )
                raise
            except httpx.TransportError as exc:
                last_category = "transport_error"
                status_categories.append(last_category)
                if attempts < self.max_attempts:
                    delay = self.backoff_base_s * (2 ** (attempts - 1))
                    cumulative_backoff_s += delay
                    self._sleep(delay)
                    continue
                self._record_trace(
                    purpose,
                    attempts,
                    status_categories,
                    cumulative_backoff_s,
                )
                raise self._blocked_api_error(
                    purpose, last_status, attempts, last_category, last_error_payload
                ) from exc
            except httpx.HTTPStatusError as exc:
                self._record_trace(
                    purpose,
                    attempts,
                    status_categories or [last_category],
                    cumulative_backoff_s,
                )
                raise self._blocked_api_error(
                    purpose, last_status, attempts, last_category, last_error_payload
                ) from exc
            except Exception as exc:
                self._record_trace(
                    purpose,
                    attempts,
                    status_categories or [last_category],
                    cumulative_backoff_s,
                )
                raise self._blocked_api_error(
                    purpose, last_status, attempts, last_category, last_error_payload
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

    def _retry_delay(self, response, attempts):
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    return max(0.0, retry_at.timestamp() - time.time())
                except (TypeError, ValueError, OverflowError):
                    pass
        return self.backoff_base_s * (2 ** (attempts - 1))

    def _record_trace(self, purpose, attempts, status_categories, cumulative_backoff_s):
        events = getattr(self._trace, "events", None)
        if events is None:
            return
        events.append(
            {
                "purpose": purpose,
                "attempts": attempts,
                "status_categories": list(status_categories),
                "cumulative_backoff_ms": cumulative_backoff_s * 1000.0,
            }
        )

    @staticmethod
    def _blocked_api_error(
        purpose, status_code, attempts, status_category, response_error=None
    ):
        return RAGBlockedError(
            "llm_api_error",
            "The configured LLM API call failed after the configured retry policy.",
            details={
                "purpose": purpose,
                "http_status": status_code,
                "attempts": attempts,
                "status_category": status_category,
                "response_error": response_error,
            },
            suggestions=[
                "Verify the API key, network access, base URL, model name, and context limits.",
                "Check provider rate limits and service health before resuming the run.",
            ],
        )


def _parse_json_object(content):
    if not isinstance(content, str):
        raise TypeError("JSON response content must be text")
    stripped = content.strip().lstrip("\ufeff")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            stripped = stripped[first_newline + 1 :] if first_newline >= 0 else stripped[3:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()
        start = stripped.find("{")
        if start < 0:
            raise
        value, _end = json.JSONDecoder().raw_decode(stripped[start:])
    if not isinstance(value, dict):
        raise ValueError("JSON response must be an object")
    return value


class IntentLLMClient:
    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"
    CANONICAL_CONSTRAINT_GUIDANCE = (
        "Use only these canonical constraint_type values: inside_boundary, no_overlap, "
        "minimum_clearance, safety_clearance, forbidden_zone_avoidance, fixed_pose, "
        "region_membership, spatial_order, near, far_from, adjacency, "
        "forbidden_adjacency, non_adjacency, alignment, port_compatibility, "
        "corridor_preservation, entrance_exit_unblocked. For an explicit wall margin "
        "or fixed buffer, use forbidden_zone_avoidance; for an explicit central aisle, "
        "use corridor_preservation; for cardinal equipment groups, use "
        "region_membership. Every source_text must be the exact contiguous source span "
        "that supports that requirement; do not invent a paraphrase or an informal label. "
        "Use only these canonical objective_type values: minimize_material_handling_distance, "
        "minimize_weighted_flow_distance, minimize_agv_transport_cost, minimize_high_flow_distance_excess, minimize_relocation_effort, "
        "maximize_space_utilization, maximize_adjacency_satisfaction, minimize_layout_change, "
        "minimize_cross_flow, minimize_total_movement, maximize_process_order_consistency, "
        "minimize_distance_to_exit, minimize_distance_to_entrance. Use "
        "minimize_weighted_flow_distance for travel weighted by batch volume. When the request "
        "asks only to make the required Repair moves as small as possible, use minimize_layout_change; do not reinterpret that as the priced minimize_relocation_effort objective. When the request "
        "also states a flow threshold and a distance threshold for excess travel, use "
        "minimize_high_flow_distance_excess with the exact threshold clause as source_text. "
        "Use minimize_relocation_effort for equipment relocation or reconfiguration effort/cost, with the exact objective clause as source_text. Coefficients are supplied planning assumptions, not inferred equipment facts. "
        "A route list and its batch volumes are process facts, not an optimization objective "
        "by themselves. Emit an objective only when the user explicitly asks to optimize, "
        "minimize, reduce, maximize, or penalize it; otherwise return an empty objectives list. "
        "Operator-assisted material handoffs do not require or establish automatic mechanical docking or port compatibility. "
        "A carrier is the main transporter between stations; manual loading and unloading do not replace the assigned AGV or conveyor. "
    )

    def __init__(
        self,
        api_key,
        *,
        model=None,
        base_url=None,
        timeout_s=60,
        thinking=None,
        max_tokens=None,
    ):
        if not api_key:
            raise RAGBlockedError(
                "missing_intent_llm_api_key",
                "An API key is required for LLM intent understanding.",
                details={"env_var": "DEEPSEEK_API_KEY"},
                suggestions=["Configure LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL, or use legacy DEEPSEEK_API_KEY settings."],
            )
        self._api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.thinking = dict(thinking) if isinstance(thinking, dict) else thinking
        self.max_tokens = max_tokens
        self.audit_directory = None

    @classmethod
    def from_env(cls, *, env=None, timeout_s=60, thinking=None):
        api_key, model, base_url = _llm_environment_config(env, intent=True)
        return cls(
            api_key,
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
            thinking=thinking,
        )

    def extract_layout_slots(
        self,
        user_message,
        *,
        active_layout_id=None,
        conversation_history=None,
        instance_directory=None,
        engineering_context=None,
    ):
        system_prompt = (
            "You are the linguistic stage of a facility-layout Intent Understanding "
            "agent. Return strict JSON only: {intent_type, confidence, site, "
            "asset_mentions, domain_intent, constraints, objectives, "
            "process_relations, edit_operations, requires_existing_layout, "
            "missing_information, clarification_questions, reasoning_summary}. "
            "Do not retrieve or guess model_asset_id. asset_mentions are "
            "{surface_text, quantity}; all source_text and surface_text values must be "
            "exact contiguous spans of the current request, supplied conversation history, "
            "or stable_engineering_context. stable_engineering_context is a persistent "
            "reference for facility, inventory, demand, routes, and evaluation assumptions. "
            "Use only the current request to decide the requested stage operation. Never "
            "turn an operation described in stable_engineering_context into a current edit. "
            "Use equipment mentions for inventory retrieval only when they are listed in "
            "the equipment inventory section or the current request; categories mentioned "
            "only in evaluation assumptions are not inventory. For a labeled inventory "
            "such as two CNC machining centers A and B, preserve the full equipment-type "
            "declaration as the retrieval mention; do not reduce it to a row alias such as "
            "CNC A. "
            "process_relations is a flat list of directed adjacent transfers: each item uses "
            "{from_surface_text, to_surface_text, product, flow_weight, sequence_order, flow_type, source_text, transport_mode, carrier_surface_text, carrier_source_text}. "
            "source_text must be an exact contiguous route-line span from the current request or stable_engineering_context, including its product and daily volume when present. On an initial Generate, treat routes and carrier assignments in stable_engineering_context as declared process facts even when the short request does not repeat them; do not ask the user to repeat them. For a long explicit arrow list, do not duplicate every adjacent edge and its repeated source line in this response; leave those long context chains for source-backed expansion after this call. Do not return nested route groups. On a small set of direct relations, copy the stated batch or trip count and exact carrier assignment sentence. Operators are valid carriers and have no model asset. Never infer carrier assignments beyond those source statements. "
            "Process endpoints use from_surface_text/to_surface_text. Receiving/dispatch doors and named floor buffers are site areas, not model assets. Preserve them as route endpoints and do not retrieve equipment models for them. Preserve partial "
            "edit operations even when a target, distance, or angle is missing, and "
            "encode an edit as {op_type, surface_text, move_mode, dx_m, dy_m, dz_m, "
            "x_m, y_m, z_m, source_text}. Use op_type move and relative dx_m/dy_m/dz_m "
            "for a translation; do not emit operation: translate or delta_x/delta_y/delta_z. "
            "For a move, surface_text is only the literal target equipment phrase, not the full action sentence. "
            "The workshop floor uses x east-west and z north-south; y is vertical. North/south floor translations use dz_m, never dy_m. East/west translations use dx_m. "
            "For a rotation use op_type rotate with rotation_mode relative and delta_yaw_deg, or rotation_mode absolute and target_yaw_deg. Clockwise relative turns have negative yaw. Split separately named targets into separate operations. "
            "For add, use op_type add, surface_text for the new user-facing role, new_asset_surface_text for the literal equipment description, quantity, and source_text. For remove, use op_type remove and the existing role in surface_text. For replace, keep the old role in surface_text and put the replacement equipment phrase in new_asset_surface_text. Preserve phrases such as near an existing station in source_text; do not invent a coordinate for them. "
            "For an explicit centered aisle-width update that moves only obstructing units "
            "in an active layout, use constraint_repair with no per-unit edit operation; "
            "the active layout supplies the obstructors and their poses. "
            "record the missing value for clarification. Resolve pronouns only from the "
            "supplied history. Never emit layout geometry, solver settings, or hidden "
            "reasoning. "
            + self.CANONICAL_CONSTRAINT_GUIDANCE
        )
        payload = {
            "current_user_message": user_message,
            "active_layout_id": active_layout_id,
            "conversation_history": conversation_history or [],
            "current_instance_directory": instance_directory or [],
            "stable_engineering_context": str(engineering_context or ""),
        }
        return self._chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose="layout_intent_linguistic_slot_extraction",
        )

    def extract_layout_intent_with_rag(
        self,
        user_message,
        rag_grounding,
        *,
        active_layout_id=None,
        instance_directory=None,
        linguistic_frame=None,
        engineering_context=None,
    ):
        system_prompt = (
            "You are the Intent Understanding agent for a facility-layout rough-design system. "
            "A linguistic stage has extracted semantic slots and Engineering RAG has "
            "resolved asset candidates. Reconcile both inputs and return strict JSON only: "
            "{intent_type, confidence, site, asset_mentions, domain_intent, constraints, objectives, "
            "process_relations, edit_operations, requires_existing_layout, missing_information, "
            "clarification_questions, reasoning_summary}. Use the current user message to "
            "decide the requested stage operation. Use stable_engineering_context as durable "
            "requirements for facility, inventory, demand, routes, and evaluation assumptions. "
            "Do not reinterpret its wording as a new operation or replay completed-turn edits. "
            "Use the request and stable context for quantities, site, constraints, objectives, "
            "edits, and process relations. Preserve incomplete "
            "edit operations from the linguistic frame. Use RAG only to bind asset mentions. "
            "intent_type must be one of generate_feasible_layout, optimize_layout_from_scratch, "
            "optimize_existing_layout, edit_layout, constraint_repair, validate_layout, "
            "evaluate_layout, unknown_or_unsupported. "
            "Every resolved model_asset_id must occur in the supplied RAG matches. Treat "
            "asset_mentions as equipment declarations rather than every repeated textual "
            "reference: merge a quantity declaration with later numbered references to those "
            "same units, and never exceed that model's current_instance_directory count. site is null or "
            "{width_m, depth_m, source_text}; constraints are {constraint_type, severity, source_text}; "
            "objectives are {objective_type, source_text}. Copy source spans from the current "
            "request, stable_engineering_context, or conversation-derived linguistic frame; "
            "assistant questions are context, not engineering facts. On an initial Generate "
            "with no active_layout_id, treat explicit routes and carrier assignments in "
            "stable_engineering_context as declared facts even when the short request does not "
            "repeat them. If the linguistic frame omitted a long arrow-chain graph, do not "
            "duplicate every adjacent edge and its full source line in this response; leave "
            "the source-backed normalizer to expand those exact lines after this call. Preserve "
            "any compact route relations already present in the frame. Stable route and carrier "
            "sentences are engineering facts, not completed operations; they must not change "
            "the current stage operation or be presented as a reason to ask the user to repeat "
            "them. When you do return a direct relation, use exact route-line source_text and "
            "exact carrier_source_text spans from the context, with the stated daily flow_weight, "
            "sequence_order, product, flow_type, and assigned carrier. "
            "Use canonical move edit fields {op_type: move, surface_text, "
            "move_mode: relative, dx_m, dy_m, dz_m, source_text}; do not use operation, translate, "
            "or delta_x/delta_y/delta_z aliases. For a move, surface_text is the literal target equipment phrase only. "
            "For an explicit centered aisle-width update "
            "that moves only obstructing units in an active layout, use constraint_repair with "
            "no per-unit edit operation. process_relations are "
            "{from_model_asset_id, to_model_asset_id, from_quantity_index, to_quantity_index, "
            "flow_type, flow_weight, sequence_order, required_connection, source_text, transport_mode, carrier_model_asset_id, carrier_quantity_index, carrier_surface_text, carrier_source_text}. Preserve the user-assigned carrier on every transfer. Carrier model identities must come from the retrieved equipment; operators have no model identity. Keep carrier resources separate from processing endpoints. carrier_source_text quotes the exact source sentence assigning the role from the current request or stable engineering context. For a declared floor buffer or receiving/dispatch door, replace the endpoint model identity with from_site_feature_name or to_site_feature_name using the literal area name. Retain these facility edges; they are not catalog equipment. Quantity "
            "indices are 1-based and may be null only when that model has exactly one instance. "
            "For repeated models, copy the index from current_instance_directory and cite the "
            "corresponding stable_alias or ordinal verbatim in source_text; otherwise report the "
            "ambiguity as missing information. Return an empty list when no relation is explicit or "
            "reviewed. Never infer an edge from asset order. Do not invent engineering facts, geometry, "
            "coordinates, solver settings, or missing values. Map a request to repair constraint "
            "violations in an existing layout to optimize_existing_layout; preserve unaffected "
            "requirements and instances. "
            + self.CANONICAL_CONSTRAINT_GUIDANCE
        )
        payload = {
            "user_message": user_message,
            "stable_engineering_context": str(engineering_context or ""),
            "active_layout_id": active_layout_id,
            "linguistic_frame": linguistic_frame or {},
            "engineering_rag_grounding": rag_grounding,
            "current_instance_directory": instance_directory or [],
            "default_hard_constraint": "no_overlap",
        }
        return self._chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose="rag_grounded_layout_intent_understanding",
        )

    def revise_layout_intent(
        self,
        user_message,
        current_semantic_intent,
        validator_feedback,
        rag_grounding,
        *,
        linguistic_frame=None,
        active_layout_id=None,
        instance_directory=None,
        engineering_context=None,
    ):
        system_prompt = (
            "Revise a facility-layout semantic intent after Validator feedback. "
            "Return the complete corrected frame as strict JSON. Apply supported "
            "repairs using the current user request, stable engineering context, "
            "and conversation-derived linguistic "
            "frame, Engineering RAG candidates, active layout and instance directory. "
            "Only the current user request defines the requested operation. The stable "
            "context supplies durable requirements and does not replay completed edits. "
            "Copy exact source spans, preserve partial edit operations, merge co-referent "
            "asset declarations and numbered process references, and never "
            "invent missing facts. If a fact remains unknown, retain the partial slot "
            "and ask one precise clarification question. Do not emit geometry, solver "
            "settings, or hidden reasoning. "
            + self.CANONICAL_CONSTRAINT_GUIDANCE
        )
        payload = {
            "user_message": user_message,
            "stable_engineering_context": str(engineering_context or ""),
            "active_layout_id": active_layout_id,
            "linguistic_frame": linguistic_frame or {},
            "current_semantic_intent": current_semantic_intent,
            "validator_feedback": validator_feedback,
            "engineering_rag_grounding": rag_grounding,
            "current_instance_directory": instance_directory or [],
        }
        return self._chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose="validator_guided_intent_revision",
        )

    def extract_layout_intent(self, user_message, *, active_layout_id=None):
        system_prompt = (
            "You are the Intent Understanding agent for a facility layout rough-design system. "
            "Extract the user's semantic intent as strict JSON only. Do not invent site dimensions, "
            "asset IDs, active layout IDs, engineering facts, or solver results. "
            "Use null or empty lists for missing information. Return this schema: "
            "{intent_type, confidence, site, asset_mentions, domain_intent, constraints, objectives, process_relations, "
            "edit_operations, controlled_violation, repair_scope, allowed_repair_operations, "
            "edit_relaxation_policy, requires_existing_layout, missing_information, "
            "clarification_questions, reasoning_summary}. "
            "intent_type must be one of generate_feasible_layout, optimize_layout_from_scratch, "
            "optimize_existing_layout, edit_layout, constraint_repair, validate_layout, "
            "evaluate_layout, unknown_or_unsupported. A user request to resolve a named "
            "constraint violation in an existing layout maps to constraint_repair, never "
            "optimize_existing_layout. "
            "site is null or {width_m, depth_m, source_text}. "
            "asset_mentions items are {surface_text, model_asset_id, quantity}. "
            "domain_intent is null or {domain, confidence, source_text}. "
            "constraints items are {constraint_type, severity, source_text}. "
            "objectives items are {objective_type, source_text}. "
            "process_relations is empty when no process dependency is stated; otherwise each item is "
            "{from_model_asset_id, to_model_asset_id, from_quantity_index, to_quantity_index, "
            "flow_type, flow_weight, sequence_order, required_connection, source_text, transport_mode, carrier_model_asset_id, carrier_quantity_index, carrier_surface_text, carrier_source_text}. Preserve the user-assigned carrier on every transfer. Carrier model identities must come from the retrieved equipment; operators have no model identity. Keep carrier resources separate from processing endpoints. carrier_source_text quotes the exact source sentence assigning the role from the current request or stable engineering context. For a declared floor buffer or receiving/dispatch door, replace the endpoint model identity with from_site_feature_name or to_site_feature_name using the literal area name. Retain these facility edges; they are not catalog equipment. Quantity "
            "indices are 1-based and may be null only for a unique model instance. Never infer a "
            "process edge from asset list order. "
            "For edit_layout, edit_operations is a non-empty ordered list. Each item is "
            "{op_type, instance_id, target_model_asset_id, new_model_asset_id, quantity, "
            "move_mode, x_m, y_m, z_m, dx_m, dy_m, dz_m, rotation_mode, target_yaw_deg, "
            "delta_yaw_deg, site_width_m, site_depth_m, placement_policy}. "
            "op_type must be move, rotate, resize_site, add, remove, or replace. Use instance_id "
            "when the user or selected scene identifies an instance; otherwise use target_model_asset_id. "
            "placement_policy must be explicit_pose, near_instance, near_site_feature, inside_zone, or auto_place. "
            "For add and replace, put the asset being introduced in new_model_asset_id, not target_model_asset_id. "
            "A replacement is one replace operation, not a remove followed by replace. "
            "For constraint_repair, controlled_violation identifies the requested violation; "
            "repair_scope contains movable_instance_ids and preserve_instance_ids; and "
            "allowed_repair_operations lists only operations authorized by the user or task. "
            "edit_relaxation_policy is null unless the user explicitly allows nearest_feasible. "
            "Do not convert relative moves to absolute coordinates and do not invent missing edit values. "
            "reasoning_summary must be a short explanation, not hidden chain-of-thought. "
            + self.CANONICAL_CONSTRAINT_GUIDANCE
        )
        payload = {
            "user_message": user_message,
            "active_layout_id": active_layout_id,
            "default_hard_constraint": "no_overlap",
        }
        return self._chat_json(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            purpose="layout_intent_understanding",
        )

    def compose_assistant_reply(
        self,
        user_message,
        result_summary,
        *,
        response_language=None,
        structured_markdown=False,
    ):
        language_instruction = (
            f"Reply only in {response_language}."
            if response_language
            else "Reply concisely in the same language as the user."
        )
        format_instruction = (
            "Use short Markdown sections named Result, Validation, and Evidence, "
            "with compact bullet lists."
            if structured_markdown
            else ""
        )
        system_prompt = (
            "You are the assistant in Agentic LayOpt. Reply to the user based only on the "
            "provided layout pipeline result summary. Do not invent solver results, assets, "
            "or validation findings. If clarification is needed, ask the listed questions. "
            "If Evaluator-assisted recovery changed an exact user directive, explicitly say "
            "that the exact directive was not satisfied, why it conflicted, which instances "
            "changed, and the requested-versus-applied displacement. Never present a disclosed "
            "nearest-feasible alternative as an exact Edit success. If a layout was generated, "
            "briefly state the final status, key solver/evaluation facts, and any remaining "
            f"issues. {language_instruction} {format_instruction}"
        )
        user_payload = {
            "user_message": user_message,
            "pipeline_result_summary": result_summary,
        }
        return self._chat_text(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            purpose="assistant_reply",
        )

    def _chat_json(self, messages, *, purpose):
        try:
            import httpx
        except ImportError as exc:
            raise RAGBlockedError(
                "missing_dependency",
                "httpx is required to call the intent LLM.",
                details={"dependency": "httpx", "purpose": purpose},
                suggestions=["Install dependencies with pip before running the rough-design API."],
            ) from exc

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        if self.max_tokens is not None:
            payload["max_tokens"] = int(self.max_tokens)
        body = None
        started = time.perf_counter()
        payload["messages"] = [dict(message) for message in messages]
        if payload["messages"] and payload["messages"][0].get("role") == "system":
            payload["messages"][0]["content"] += PLANNING_SLOT_INSTRUCTIONS
        audit = {"purpose": purpose, "request": payload}
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
            response.raise_for_status()
            body = response.json()
            audit["response"] = body
            content = body["choices"][0]["message"]["content"]
            if body["choices"][0].get("finish_reason") == "length":
                raise ValueError("Intent JSON was truncated by the output-token limit.")
            return json.loads(content)
        except Exception as exc:
            audit["error"] = {"type": type(exc).__name__, "message": str(exc)}
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            raise RAGBlockedError(
                "intent_llm_api_error",
                "Intent LLM call failed or returned invalid JSON.",
                details={"purpose": purpose, "http_status": status_code, "exception_type": type(exc).__name__,
                         "finish_reason": (body or {}).get("choices", [{}])[0].get("finish_reason"),
                         "output_characters": len(str(((body or {}).get("choices", [{}])[0].get("message") or {}).get("content") or ""))},
                suggestions=["Verify API key, network access, base URL, and model name."],
            ) from exc

        finally:
            if self.audit_directory:
                from pathlib import Path
                directory = Path(self.audit_directory)
                directory.mkdir(parents=True, exist_ok=True)
                audit["elapsed_s"] = time.perf_counter()-started
                (directory / (purpose+"-"+str(time.time_ns())+".json")).write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")

    def _chat_text(self, messages, *, purpose):
        try:
            import httpx
        except ImportError as exc:
            raise RAGBlockedError(
                "missing_dependency",
                "httpx is required to call the intent LLM.",
                details={"dependency": "httpx", "purpose": purpose},
                suggestions=["Install dependencies with pip before running the rough-design API."],
            ) from exc

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        if self.thinking is not None:
            payload["thinking"] = self.thinking
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
            response.raise_for_status()
            body = response.json()
            return str(body["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            raise RAGBlockedError(
                "intent_llm_api_error",
                "Intent LLM call failed while composing the assistant reply.",
                details={"purpose": purpose, "http_status": status_code},
                suggestions=["Verify API key, network access, base URL, and model name."],
            ) from exc


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _bounded_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _evidence_refs(value):
    if not isinstance(value, list):
        return []
    refs = []
    for item in value:
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "")
        field_path = str(item.get("field_path") or "")
        if source_id and field_path:
            refs.append({"source_id": source_id, "field_path": field_path})
    return refs


def _http_status_category(status_code):
    if status_code == 429:
        return "rate_limited"
    if status_code in {408, 500, 502, 503, 504}:
        return "transient_http_error"
    if status_code in {401, 403}:
        return "authentication_error"
    if status_code == 404:
        return "model_or_endpoint_not_found"
    if status_code is not None and 400 <= status_code < 500:
        return "deterministic_request_error"
    return "http_error"
