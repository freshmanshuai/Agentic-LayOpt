import copy
import json
import math
import re
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from tool.rag.engineering_rag import EngineeringRAGService
from tool.clients.llm_clients import RAGBlockedError
from tool.context.instance_naming import declaration_role
from tool.logistics.facility_logistics import resolve_facility_ref, fixed_flow_endpoints
from tool.layout.natural_site_requirements import compile_natural_site_requirements
from tool.logistics.relocation_effort import (RELOCATION_EFFORT_OBJECTIVE, relocation_requested, compile_relocation_parameters, validate_relocation_parameters)
from tool.logistics.agv_transport_cost import AGV_TRANSPORT_OBJECTIVE, agv_cost_requested, compile_agv_parameters, validate_agv_parameters
from tool.logistics.transfer_workcell import compile_transfer_cell, compile_functional_subregions


SCHEMA_VERSION = "0.1.0"
DEFAULT_PROJECT_ID = "PROJECT_001"
SUPPORTED_FLP_INTENTS = {
    "generate_feasible_layout",
    "optimize_layout_from_scratch",
    "optimize_existing_layout",
}
SUPPORTED_CONSTRAINTS = {
    "inside_boundary",
    "no_overlap",
    "minimum_clearance",
    "safety_clearance",
    "forbidden_zone_avoidance",
    "fixed_pose",
    "region_membership",
    "spatial_order",
    "near",
    "far_from",
    "adjacency",
    "forbidden_adjacency",
    "non_adjacency",
    "alignment",
    "port_compatibility",
    "corridor_preservation",
    "entrance_exit_unblocked",
}
SUPPORTED_OBJECTIVES = {
    AGV_TRANSPORT_OBJECTIVE,
    "minimize_material_handling_distance",
    "minimize_weighted_flow_distance",
    "minimize_high_flow_distance_excess",
    RELOCATION_EFFORT_OBJECTIVE,
    "maximize_space_utilization",
    "maximize_adjacency_satisfaction",
    "minimize_layout_change",
    "minimize_cross_flow",
    "minimize_total_movement",
    "maximize_process_order_consistency",
    "minimize_distance_to_exit",
    "minimize_distance_to_entrance",
}
CONSTRAINT_SYNONYMS = {
    "within_site_boundary": "inside_boundary",
    "avoid_forbidden_zone": "forbidden_zone_avoidance",
    "adjacent_to": "adjacency",
    # Natural-language extraction uses these descriptive labels before the
    # site compiler turns the requirement into concrete walls, corridors,
    # placement regions, and fixed buffer geometry.  Keep the FLP-IR on the
    # canonical constraint vocabulary consumed by the compiler and validator.
    "boundary_clearance": "forbidden_zone_avoidance",
    "wall_clearance": "forbidden_zone_avoidance",
    "aisle_reserved": "corridor_preservation",
    "aisle_clearance": "corridor_preservation",
    "zone_assignment": "region_membership",
    "buffer_reserved": "forbidden_zone_avoidance",
    "buffer_reservation": "forbidden_zone_avoidance",
    "buffer_unobstructed": "forbidden_zone_avoidance",
    "buffer_clearance": "forbidden_zone_avoidance",
}
OBJECTIVE_SYNONYMS = {
    "minimize_logistics_distance": "minimize_material_handling_distance",
    "minimize_weighted_distance": "minimize_weighted_flow_distance",
    "minimize_total_movement": "minimize_layout_change",
}
FIELD_REQUEST_PATTERNS = [
    (
        "planning_fields.ports",
        re.compile(r"\b(port|ports|connection point|connection points|inlet|outlet)\b", re.I),
    ),
    (
        "planning_fields.maintenance_clearance_m",
        re.compile(
            r"\b(maintenance|maintenance room|service access|service clearance|service space|"
            r"maintenance and safety space|service and safety room|technicians? need|"
            r"room should technicians? have)\b",
            re.I,
        ),
    ),
    (
        "planning_fields.safety_clearance_m",
        re.compile(
            r"\b(safety clearance|exclusion distance|safety distance|safety space|safety gap|"
            r"safety and maintenance space|service and safety room)\b|"
            r"close to the next machine",
            re.I,
        ),
    ),
    (
        "planning_fields.process_capabilities",
        re.compile(
            r"\b(process capabilities|production operations|machining operations|operations can|"
            r"jobs can|what jobs|which operations|work can)\b|can (this|the) machine handle",
            re.I,
        ),
    ),
]


class AssetReferenceAdapter:
    def __init__(self, dataset_dir, rag_service=None):
        self.dataset_dir = Path(dataset_dir)
        assets_path = self.dataset_dir / "model_assets.json"
        self.assets = json.loads(assets_path.read_text(encoding="utf-8"))
        self.assets_by_id = {asset["model_asset_id"]: asset for asset in self.assets}
        viewport_assets_path = self.dataset_dir / "three_viewport_assets.json"
        self.viewport_assets = (
            json.loads(viewport_assets_path.read_text(encoding="utf-8")).get("assets", [])
            if viewport_assets_path.exists()
            else []
        )
        self.rag_service = rag_service if rag_service is not None else EngineeringRAGService(self.dataset_dir)

    def get_asset(self, model_asset_id):
        try:
            return self.assets_by_id[model_asset_id]
        except KeyError as exc:
            raise KeyError(model_asset_id) from exc

    def find_candidates(self, mention):
        if hasattr(self.rag_service, "find_asset_candidates"):
            return self.rag_service.find_asset_candidates(mention)
        return _flatten_asset_matches(self.find_matches(mention))

    def find_matches(self, mention):
        if not hasattr(self.rag_service, "query"):
            return _candidates_to_match(mention, [])
        result = self.rag_service.query(mention, top_k=8)
        asset_matches = result.get("asset_matches", [])
        if asset_matches:
            return asset_matches
        domain_matches = result.get("domain_asset_matches") or []
        if not domain_matches:
            return []
        domain_intent = result.get("domain_intent") or {}
        return [
            {
                "surface_text": domain_intent.get("source_text") or mention,
                "candidate_model_asset_ids": domain_matches,
                "selected_model_asset_id": item["model_asset_id"],
                "selected_library_id": item.get("library_id"),
                "selected_site_id": item.get("site_id"),
                "ambiguity_status": "resolved",
                "match_type": "domain_intent",
            }
            for item in domain_matches[:5]
        ]

    def legacy_find_candidates(self, mention):
        mention_norm = _normalize_text(mention)
        candidates = []
        for asset in self.assets:
            score, match_type = self._match_score(asset, mention_norm)
            if score > 0:
                candidates.append(_candidate(asset, score, match_type))
        if not candidates and "检测" in mention:
            candidates.extend(self._inspection_candidates())
        candidates.sort(key=lambda item: (-item["score"], item["model_asset_id"]))
        return candidates

    def asset_fact(self, model_asset_id):
        asset = self.get_asset(model_asset_id)
        return {
            "model_asset_id": asset["model_asset_id"],
            "asset_scope": asset["asset_scope"],
            "library_id": asset.get("library_id"),
            "site_id": asset.get("site_id"),
            "canonical_name": asset.get("asset_name"),
            "equipment_type": asset.get("equipment_type"),
            "category": asset.get("category"),
            "geometry": copy.deepcopy(asset["geometry"]),
            "planning_fields": copy.deepcopy(asset.get("planning_fields", {})),
            "provenance": copy.deepcopy(asset.get("provenance", [])),
        }

    def audit_references(self):
        library_to_assets = {}
        for asset in self.assets:
            library_to_assets.setdefault(asset.get("library_id"), []).append(asset["model_asset_id"])
        warnings = []
        for library_id, asset_ids in sorted(library_to_assets.items(), key=lambda item: str(item[0])):
            if library_id and len(asset_ids) > 1:
                warnings.append(
                    {
                        "library_id": library_id,
                        "warning_type": "library_id_not_unique",
                        "model_asset_ids": asset_ids,
                    }
                )
        return {
            "direct_referenceable_model_asset_ids": sorted(self.assets_by_id),
            "reference_warnings": warnings,
        }

    def viewport_asset_ids(self, domain=None):
        return [
            item["model_asset_id"]
            for item in self.viewport_assets
            if item.get("model_asset_id") in self.assets_by_id
            and (not domain or domain in item.get("domains", []))
        ]

    def _match_score(self, asset, mention_norm):
        if not mention_norm:
            return 0.0, "semantic"
        if mention_norm == _normalize_text(asset["model_asset_id"]):
            return 1.0, "id_exact"
        names = [asset.get("asset_name", ""), asset.get("equipment_type", "")]
        names.extend(asset.get("aliases", []))
        for name in names:
            name_norm = _normalize_text(name)
            if not name_norm:
                continue
            if mention_norm == name_norm:
                return 0.98, "alias_exact"
            if mention_norm in name_norm or name_norm in mention_norm:
                return 0.9, "alias_exact"
        return 0.0, "semantic"

    def _inspection_candidates(self):
        candidates = []
        for asset in self.assets:
            text = "".join(
                [
                    asset.get("asset_name", ""),
                    asset.get("equipment_type", ""),
                    " ".join(asset.get("aliases", [])),
                ]
            )
            if any(token in text for token in ["测试", "检测", "控制台", "test"]):
                candidates.append(_candidate(asset, 0.62, "semantic"))
        return candidates[:3]


def route_intent(user_message, project_id=DEFAULT_PROJECT_ID, active_layout_id=None, site_id=None):
    message = user_message.strip()
    message_lower = message.lower()
    intent_type = "unknown_or_unsupported"
    detected_operations = []

    operation_keywords = {
        "move": ["移动", "向右", "向左", "向前", "向后", "挪", "move"],
        "rotate": ["旋转", "转90", "rotate"],
        "replace": ["更换", "替换", "replace"],
        "remove": ["删除", "移除", "remove", "delete"],
        "add": ["添加", "增加", "add"],
        "lock": ["锁定", "lock"],
        "unlock": ["解锁", "unlock"],
    }
    for operation, keywords in operation_keywords.items():
        if operation == "move" and any(phrase in message for phrase in ["不要移动", "不移动", "不能移动"]):
            continue
        if any(keyword in message for keyword in keywords):
            detected_operations.append(operation)
    constraint_repair_request = (
        any(
            keyword in message_lower
            for keyword in ("repair", "fix", "resolve", "消除", "修复")
        )
        and any(
            keyword in message_lower
            for keyword in (
                "violation",
                "overlap",
                "out of bounds",
                "forbidden zone",
                "corridor",
                "clearance",
                "违规",
                "重叠",
                "越界",
                "禁区",
                "通道",
                "间距",
            )
        )
    )
    if constraint_repair_request:
        intent_type = "constraint_repair"
    elif detected_operations:
        intent_type = "edit_layout"
    elif any(keyword in message for keyword in ["验证布局", "校验布局", "validate"]):
        intent_type = "validate_layout"
    elif any(keyword in message for keyword in ["评价布局", "评估布局", "evaluate", "assess layout"]):
        intent_type = "evaluate_layout"
    elif any(keyword in message for keyword in ["解释布局", "说明布局", "explain"]):
        intent_type = "explain_layout"
    elif any(keyword in message for keyword in ["比较", "对比", "compare"]):
        intent_type = "compare_layouts"
    elif any(keyword in message for keyword in ["导出", "export"]):
        intent_type = "export_layout"
    elif any(keyword in message for keyword in ["当前布局", "已有布局", "现有布局", "基础上优化"]):
        intent_type = "optimize_existing_layout"
    elif any(keyword in message for keyword in ["优化", "最小化", "最小", "缩短", "尽量"]):
        intent_type = "optimize_layout_from_scratch"
    elif any(keyword in message for keyword in ["生成", "布置", "布局", "安排"]):
        intent_type = "generate_feasible_layout"
    if intent_type == "unknown_or_unsupported":
        intent_type = _route_english_intent(message_lower)

    unresolved_items = []
    requires_existing_layout = intent_type in {
        "optimize_existing_layout",
        "edit_layout",
        "constraint_repair",
        "validate_layout",
        "evaluate_layout",
    }
    if requires_existing_layout and not active_layout_id:
        unresolved_items.append(
            _unresolved_item(
                "intent_missing_active_layout",
                "missing_field",
                "active_layout_id",
                "This intent requires an active layout reference.",
                "blocking",
                [],
                "ask_user",
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "intent_id": "INTENT_001",
        "project_id": project_id,
        "active_layout_id": active_layout_id,
        "site_id": site_id,
        "user_message": user_message,
        "intent_type": intent_type,
        "confidence": 0.9 if intent_type != "unknown_or_unsupported" else 0.25,
        "requires_existing_layout": requires_existing_layout,
        "requires_asset_lookup": intent_type in SUPPORTED_FLP_INTENTS
        or intent_type in {"edit_layout", "constraint_repair"},
        "requires_site": intent_type in SUPPORTED_FLP_INTENTS
        or intent_type == "constraint_repair",
        "requires_confirmation": intent_type in {"unknown_or_unsupported", "edit_layout"},
        "target_refs": [],
        "detected_operations": detected_operations,
        "unresolved_items": unresolved_items,
        "provenance": [
            _provenance("intent.intent_type", "user_input", "INTENT_001", "user_message", 0.9)
        ],
    }


def build_evidence_package(intent, adapter, semantic_frame=None, context=None):
    context = context or {}
    message = intent["user_message"]
    parameter_text = "\n".join(
        value
        for value in (str(context.get("engineering_context") or ""), str(message or ""))
        if value.strip()
    )
    if context.get("instance_directory"):
        matched_assets = [
            {
                "surface_text": item.get("model_asset_id"),
                "candidate_model_asset_ids": [item.get("model_asset_id")],
                "selected_model_asset_id": item.get("model_asset_id"),
                "selected_library_id": None,
                "selected_site_id": None,
                "ambiguity_status": "resolved",
                "match_type": "authoritative_instance_directory",
            }
            for item in context["instance_directory"]
        ]
        asset_facts = _authoritative_asset_facts(
            context["instance_directory"],
            adapter,
        )
        ambiguities = []
    elif _validated_rag_asset_mentions(semantic_frame, context):
        # Consume the already retrieved and human-confirmed identities. A second
        # independent lookup must not replace the user's selected table or machine.
        matched_assets = []
        facts_by_model = {}
        for mention in semantic_frame["asset_mentions"]:
            mid = mention["model_asset_id"]
            quantity = int(mention["quantity"])
            fact = facts_by_model.setdefault(mid, dict(adapter.asset_fact(mid), quantity=0, instance_roles={}))
            start = fact["quantity"]
            for index in range(1, quantity+1):
                fact["instance_roles"][str(start+index)] = declaration_role(mention["surface_text"],quantity,index)
            fact["quantity"] += quantity
            retrieved = next((match for match in context["engineering_rag_grounding"].get("asset_matches") or []
                              if str(match.get("surface_text") or "").casefold() == str(mention["surface_text"]).casefold()), {})
            matched_assets.append(dict(surface_text=mention["surface_text"],selected_model_asset_id=mid,
                candidate_model_asset_ids=copy.deepcopy(retrieved.get("candidate_model_asset_ids") or [mid]),selected_library_id=fact.get("library_id"),selected_site_id=fact.get("site_id"),
                ambiguity_status="resolved",match_type="validated_rag_selection",quantity=quantity))
        # Reuse the same literal station names that the user used in the routes.
        named = set()
        for relation in semantic_frame.get("process_relations") or []:
            for prefix in ("from", "to"):
                mid = relation.get(f"{prefix}_model_asset_id")
                index = relation.get(f"{prefix}_quantity_index")
                name = relation.get(f"{prefix}_surface_text")
                key = (mid,index)
                if mid in facts_by_model and index and name and key not in named:
                    facts_by_model[mid]["instance_roles"][str(index)] = name
                    named.add(key)
        asset_facts = list(facts_by_model.values())
        ambiguities = []
    else:
        matched_assets = (
            _explicit_asset_matches_from_message(message, adapter)
            or _match_assets_from_message(message, adapter)
        )
        asset_facts = []
        ambiguities = []
        for match in matched_assets:
            if (
                match["ambiguity_status"] == "resolved"
                and match["selected_model_asset_id"]
            ):
                fact = adapter.asset_fact(match["selected_model_asset_id"])
                fact["quantity"] = _quantity_from_match(message, match)
                asset_facts.append(fact)
            else:
                ambiguities.append(
                    _unresolved_item(
                        f"ambiguous_asset_{len(ambiguities) + 1:03d}",
                        "ambiguous_reference",
                        "matched_assets",
                        f"Asset mention '{match['surface_text']}' could not be resolved uniquely.",
                        "blocking",
                        [
                            candidate["model_asset_id"]
                            for candidate in match["candidate_model_asset_ids"]
                        ],
                        "ask_user",
                    )
                )
        _align_generic_asset_facts_with_viewport(
            asset_facts, adapter, semantic_frame
        )
        _apply_semantic_quantity_to_asset_facts(asset_facts, semantic_frame)
    _apply_semantic_planning_overrides(asset_facts, semantic_frame)
    field_gaps = _requested_missing_fields(message, asset_facts)
    if context.get("instance_directory"):
        field_gaps = [
            item
            for item in field_gaps
            if not any(
                token in " ".join(
                    str(item.get(key) or "")
                    for key in ("item_id", "field_path", "description")
                ).casefold()
                for token in ("asset", "equipment", "quantity")
            )
        ]

    if not matched_assets and intent["intent_type"] in SUPPORTED_FLP_INTENTS and (
        _mentions_equipment(message) or _mentions_english_equipment(message)
    ):
        ambiguities.append(
            _unresolved_item(
                "unresolved_asset_001",
                "ambiguous_reference",
                "matched_assets",
                "The user mentioned equipment, but no asset matched the current dataset.",
                "blocking",
                [],
                "ask_user",
            )
        )

    site_facts = _site_facts_from_evidence(message, semantic_frame, context)
    matched_constraints = []
    if intent["intent_type"] in SUPPORTED_FLP_INTENTS:
        matched_constraints.append(
            {
                "constraint_type": "no_overlap",
                "default_severity": "hard",
                "parameters_schema": {},
            }
        )
    if site_facts:
        matched_constraints.append(
            {
                "constraint_type": "inside_boundary",
                "default_severity": "hard",
                "parameters_schema": {},
            }
        )

    matched_objectives = []
    weighted_flow_source = _weighted_flow_objective_source(message)
    high_flow_parameters = _high_flow_distance_excess_parameters(message)
    if agv_cost_requested(message):
        try:
            from tool.context.conversation_deltas import inherited_objective_parameters
            parameters = inherited_objective_parameters(context, AGV_TRANSPORT_OBJECTIVE)
            if parameters is None or re.search(r"\b(?:loaded|empty)\s+travel(?:\s+costs?)?\s+\d|\bbatches?\s+per\s+trip|\bbatch\s+per\s+trip", message, re.I):
                parameters = compile_agv_parameters(
                    message if parameters is not None else parameter_text
                )
            validate_agv_parameters(parameters)
            matched_objectives.append(dict(objective_type=AGV_TRANSPORT_OBJECTIVE, sense="minimize", parameters_schema={}, parameters=parameters))
        except ValueError as error:
            ambiguities.append(_unresolved_item("agv_cost_parameters", "missing_field", "objectives.parameters", str(error), "blocking", [], "ask_user"))
    elif weighted_flow_source:
        matched_objectives.append(
            {
                "objective_type": "minimize_weighted_flow_distance",
                "sense": "minimize",
                "parameters_schema": {},
            }
        )
        if high_flow_parameters is not None:
            matched_objectives.append(
                {
                    "objective_type": "minimize_high_flow_distance_excess",
                    "sense": "minimize",
                    "parameters_schema": {
                        "flow_threshold_batches_per_day": "number",
                        "distance_threshold_m": "number",
                    },
                    "parameters": high_flow_parameters,
                }
            )
        elif _high_flow_distance_excess_clause_present(message):
            ambiguities.append(
                _unresolved_item(
                    "high_flow_distance_excess_parameters",
                    "missing_field",
                    "objectives.parameters",
                    "The high-flow distance objective needs positive flow and distance thresholds with meter units.",
                    "blocking",
                    [],
                    "ask_user",
                )
            )
    elif _has_logistics_distance_objective(message) or _has_english_logistics_distance_objective(message):
        matched_objectives.append(
            {
                "objective_type": "minimize_material_handling_distance",
                "sense": "minimize",
                "parameters_schema": {},
            }
        )
    elif "最漂亮" in message or "美观" in message:
        matched_objectives.append(
            {
                "objective_type": "unsupported_aesthetic_layout",
                "sense": "maximize",
                "parameters_schema": {},
            }
        )

    if relocation_requested(message):
        try:
            from tool.context.conversation_deltas import inherited_objective_parameters
            relocation_parameters = inherited_objective_parameters(context, RELOCATION_EFFORT_OBJECTIVE)
            if relocation_parameters is None or re.search(r"\bfixed effort\s+\d|\bdistance effort\s+\d", message, re.I):
                relocation_parameters = compile_relocation_parameters(
                    message if relocation_parameters is not None else parameter_text,
                    context,
                    asset_facts,
                )
            matched_objectives.append(dict(objective_type=RELOCATION_EFFORT_OBJECTIVE, sense="minimize", parameters_schema={}, parameters=relocation_parameters))
        except ValueError as error:
            ambiguities.append(_unresolved_item("relocation_effort_parameters", "missing_field", "objectives.parameters", str(error), "blocking", [], "ask_user"))

    process_relations = copy.deepcopy(
        (semantic_frame or {}).get("process_relations") or []
    )
    if "candidate engineering facts pending human review" in message.casefold():
        for relation in process_relations:
            relation["flow_type"] = "material"
            relation["flow_weight"] = 1.0
            relation["sequence_order"] = None
            relation["required_connection"] = False
            relation["evidence_source"] = "controlled_engineering_candidate"
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_package_id": "EP_001",
        "intent_id": intent["intent_id"],
        "project_id": intent["project_id"],
        "matched_assets": matched_assets,
        "asset_facts": asset_facts,
        "matched_rules": [],
        "matched_constraints": matched_constraints,
        "semantic_constraints": copy.deepcopy(
            (semantic_frame or {}).get("constraints") or []
        ),
        "matched_objectives": matched_objectives,
        "process_relations": process_relations,
        "site_facts_from_user": site_facts,
        "ambiguities": ambiguities,
        "unresolved_items": field_gaps,
        "retrieval_trace": [
            {
                "retrieval_step": "asset_lookup",
                "input_text": message,
                "matched_source_ids": [fact["model_asset_id"] for fact in asset_facts],
                "confidence": 0.9 if asset_facts else None,
            }
        ],
    }


def _apply_semantic_planning_overrides(asset_facts, frame):
    facts_by_id = {fact["model_asset_id"]: fact for fact in asset_facts}
    cursors = {}
    for mention in (frame or {}).get("asset_mentions") or []:
        model_id = mention.get("model_asset_id")
        fact = facts_by_id.get(model_id)
        if fact is None:
            continue
        overrides = mention.get("planning_overrides") or {}
        planning = {}
        for key in ("maintenance_clearance_m", "safety_clearance_m"):
            if key not in overrides:
                continue
            sides = overrides[key]
            if not isinstance(sides, dict):
                raise ValueError(f"{key} must contain numeric side clearances")
            values = {}
            for side in ("front_m", "rear_m", "left_m", "right_m"):
                if side not in sides:
                    continue
                value = sides[side]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{key}.{side} must be a finite nonnegative number")
                values[side] = float(value)
            if values:
                planning[key] = dict(values, source="user_input", confidence=1.0)
        if "allowed_orientations_deg" in overrides:
            angles = overrides["allowed_orientations_deg"]
            if not isinstance(angles, list) or not angles or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in angles
            ):
                raise ValueError("allowed_orientations_deg must contain finite numeric angles")
            planning["allowed_orientations_deg"] = [float(v) % 360 for v in angles]
        start = cursors.get(model_id, 0)
        quantity = max(1, int(mention.get("quantity") or 1))
        labels = mention.get("instance_labels") or []
        for offset in range(quantity):
            index = str(start + offset + 1)
            if planning:
                fact.setdefault("instance_planning_overrides", {})[index] = copy.deepcopy(planning)
            if offset < len(labels):
                fact.setdefault("instance_roles", {})[index] = labels[offset]
        cursors[model_id] = start + quantity


def _validated_rag_asset_mentions(frame, context):
    mentions = (frame or {}).get("asset_mentions") or []
    selected = {m.get("selected_model_asset_id") for m in (context.get("engineering_rag_grounding") or {}).get("asset_matches") or [] if m.get("selected_model_asset_id")}
    return bool(mentions and selected and all(m.get("model_asset_id") in selected and isinstance(m.get("quantity"), int) and m["quantity"]>0 for m in mentions))


def _authoritative_asset_facts(instance_directory, adapter):
    counts = {}
    for item in instance_directory or []:
        model_asset_id = str(item.get("model_asset_id") or "").strip()
        if model_asset_id:
            counts[model_asset_id] = counts.get(model_asset_id, 0) + 1
    facts = []
    for model_asset_id, quantity in sorted(counts.items()):
        fact = adapter.asset_fact(model_asset_id)
        fact["quantity"] = quantity
        fact["quantity_provenance"] = {
            "source_type": "computed",
            "source_id": "authoritative_context",
            "source_field": "instance_directory",
            "confidence": 1.0,
        }
        fact["selection_provenance"] = copy.deepcopy(
            fact["quantity_provenance"]
        )
        facts.append(fact)
    return facts


def _apply_semantic_quantity_to_asset_facts(asset_facts, semantic_frame):
    if not asset_facts or not semantic_frame:
        return

    generic_mentions = [
        item
        for item in semantic_frame.get("asset_mentions", [])
        if not item.get("model_asset_id") and int(item.get("quantity") or 0) > 0
    ]
    if not generic_mentions:
        return

    requested_count = sum(int(item.get("quantity") or 0) for item in generic_mentions)
    if requested_count <= 0:
        return

    selected_count = min(len(asset_facts), requested_count)
    del asset_facts[selected_count:]
    base_quantity, remainder = divmod(requested_count, selected_count)
    source_text = ", ".join(item.get("surface_text") or "functional units" for item in generic_mentions)
    for index, fact in enumerate(asset_facts):
        fact["quantity"] = base_quantity + (1 if index < remainder else 0)
        fact["quantity_provenance"] = {
            "source_type": "user_input",
            "source_id": "INTENT_001",
            "source_field": source_text,
            "confidence": semantic_frame.get("confidence"),
        }


def _align_generic_asset_facts_with_viewport(asset_facts, adapter, semantic_frame):
    if not asset_facts or not semantic_frame:
        return
    if not any(
        not item.get("model_asset_id") and int(item.get("quantity") or 0) > 0
        for item in semantic_frame.get("asset_mentions", [])
    ):
        return

    domain = (semantic_frame.get("domain_intent") or {}).get("domain")
    viewport_asset_ids = adapter.viewport_asset_ids(domain=domain)
    if not viewport_asset_ids:
        return

    facts_by_id = {fact["model_asset_id"]: fact for fact in asset_facts}
    selected = []
    for model_asset_id in viewport_asset_ids:
        fact = facts_by_id.get(model_asset_id)
        if fact is None:
            fact = adapter.asset_fact(model_asset_id)
            fact["selection_provenance"] = {
                "source_type": "computed",
                "source_id": "three_viewport_assets",
                "source_field": domain or "generic",
                "confidence": 0.9,
            }
        selected.append(fact)
    asset_facts[:] = selected


def compile_flp_ir(user_message, intent, evidence_package, context=None):
    context = context or {}
    if intent["intent_type"] not in SUPPORTED_FLP_INTENTS:
        return None

    unresolved_items = copy.deepcopy(evidence_package.get("ambiguities", []))
    unresolved_items.extend(copy.deepcopy(evidence_package.get("unresolved_items", [])))
    provenance = [
        _provenance("source_request.raw_user_message", "user_input", intent["intent_id"], "user_message", 1.0)
    ]
    site_ref = _compile_site_ref(
        evidence_package,
        intent,
        unresolved_items,
        provenance,
        context=context,
    )
    if intent["intent_type"] == "optimize_existing_layout":
        equipment_instances = _compile_existing_layout_equipment_instances(
            context,
            provenance,
        )
        if equipment_instances:
            _cross_check_existing_asset_declarations(
                evidence_package,
                equipment_instances,
                unresolved_items,
            )
        else:
            equipment_instances = _compile_equipment_instances(
                evidence_package,
                provenance,
                context=context,
            )
    else:
        equipment_instances = _compile_equipment_instances(
            evidence_package,
            provenance,
            context=context,
        )
    current_input = str(context.get("current_user_input") or user_message or "")
    engineering_context = str(context.get("engineering_context") or "")
    task_type = intent.get("intent_type")
    site_requirement_text = current_input
    if task_type in {"generate_feasible_layout", "optimize_layout_from_scratch"} and engineering_context:
        site_requirement_text = "\n".join(
            value for value in (engineering_context, current_input) if value.strip()
        )
    organization_text = "\n".join(
        value for value in (engineering_context, current_input) if value.strip()
    )
    natural_site = compile_natural_site_requirements(
        site_requirement_text,
        (site_ref or {}).get("site_snapshot") or {},
        semantic_intent=context.get("current_semantic_intent") or {},
        equipment_instances=equipment_instances,
        organization_text=organization_text,
        intent_id=intent["intent_id"],
    )
    if natural_site.get("site_snapshot"):
        site_ref["site_snapshot"] = natural_site["site_snapshot"]
    unresolved_items.extend(natural_site.get("unresolved_items") or [])
    provenance.extend(natural_site.get("provenance") or [])
    assignments = natural_site.get("instance_region_assignments") or {}
    for index, instance in enumerate(equipment_instances):
        assignment = assignments.get(instance.get("instance_id"))
        if not assignment:
            continue
        instance["allowed_regions"] = [assignment["region_id"]]
        assignment_provenance = _provenance(
            f"equipment_instances[{index}].allowed_regions",
            "user_input",
            intent["intent_id"],
            assignment.get("source_text") or "user_message",
            1.0,
        )
        assignment_provenance["note"] = (
            "Accepted equipment instance is constrained to its source-assigned workstation bay."
            if assignment.get("bay_id")
            else "Placement region derived from the resolved equipment mention and "
            "the explicit cardinal group statement."
        )
        provenance.append(assignment_provenance)
    hard_constraints, soft_constraints = _compile_constraints(evidence_package, equipment_instances, provenance)
    _append_natural_site_constraints(
        hard_constraints,
        natural_site.get("constraint_requirements") or [],
        provenance,
        intent["intent_id"],
    )
    objectives = _compile_objectives(evidence_package, provenance)
    flow_relations = _compile_flow_relations(
        equipment_instances,
        objectives,
        provenance,
        process_relations=evidence_package.get("process_relations"),
        site_features=((site_ref or {}).get("site_snapshot") or {}).get("site_features") or [],
        unresolved_items=unresolved_items,
        allow_instance_id_endpoints=context.get("experiment") != "E3V2",
    )

    if intent["intent_type"] == "optimize_existing_layout":
        base_layout_ref = intent.get("active_layout_id") or context.get("active_layout_id")
        if not base_layout_ref:
            unresolved_items.append(
                _unresolved_item(
                    "missing_active_layout",
                    "missing_field",
                    "optimization_scope.base_layout_id",
                    "optimize_existing_layout requires an active layout reference.",
                    "blocking",
                    [],
                    "ask_user",
                )
            )
    else:
        base_layout_ref = None

    requires_optimization = intent["intent_type"] != "generate_feasible_layout" or bool(objectives)
    flp_ir = {
        "schema_version": SCHEMA_VERSION,
        "task_meta": {
            "task_id": "TASK_001",
            "task_type": intent["intent_type"],
            "project_id": intent["project_id"],
            "domain": "facility_layout",
            "created_from_intent_id": intent["intent_id"],
            "equipment_library_version": None,
            "rule_library_version": None,
        },
        "unit_system": _unit_system(),
        "site_ref": site_ref,
        "equipment_instances": equipment_instances,
        "flow_relations": flow_relations,
        "constraints": {"hard": hard_constraints, "soft": soft_constraints},
        "objectives": objectives,
        "optimization_scope": {
            "base_layout_id": base_layout_ref,
            "locked_instances": [],
            "movable_instances": [item["instance_id"] for item in equipment_instances],
            "editable_zones": [],
        },
        "delta_policy": {
            "use_current_layout_as_reference": intent["intent_type"] == "optimize_existing_layout",
            "penalize_large_movements": intent["intent_type"] == "optimize_existing_layout",
            "max_movement_distance_m": None,
            "movement_cost_weight": None,
        },
        "planning_preferences": {
            "generation_policy": "process_flow_based" if requires_optimization else "asset_based",
            "max_runtime_s": None,
            "return_top_k": None,
        },
        "provenance": provenance,
        "unresolved_items": unresolved_items,
    }
    if intent["intent_type"] == "optimize_existing_layout":
        _merge_existing_design_context(
            flp_ir,
            context,
            current_site_is_missing=_site_evidence_is_missing(evidence_package),
            explicit_flow_update=bool(evidence_package.get("process_relations") and re.search(r"(?:->|→)", str(intent.get("user_message") or "").rsplit("Current user request:\n",1)[-1])),
        )
    # Persisted context supplies stable engineering requirements on a fresh
    # Generate. Later stages inherit the compiled layout and only parse their
    # current request for explicit updates.
    layout_requirement_text = (
        site_requirement_text
        if task_type in {"generate_feasible_layout", "optimize_layout_from_scratch"}
        else current_input
    )
    compile_transfer_cell(flp_ir, layout_requirement_text)
    compile_functional_subregions(flp_ir, layout_requirement_text)
    if any(f.get("transport_mode")=="agv" for f in flp_ir["flow_relations"]) and (re.search(r"continuous access",context.get("current_user_input") or user_message,re.I) or any(o.get("objective_type")==AGV_TRANSPORT_OBJECTIVE for o in objectives)):
        flp_ir["constraints"]["hard"].append(dict(constraint_id=f"constraint_{len(flp_ir['constraints']['hard'])+1:03d}",
            constraint_type="corridor_preservation",target_refs=[dict(target_type="all_equipment",target_id="all")],
            parameters=dict(planning_kind="agv_external_access",basis="Declared continuous AGV access, conservative calibrated body envelope"),
            source="user_input",evidence_refs=["source_request.raw_user_message"]))
    return flp_ir


def _merge_existing_design_context(flp_ir, context, *, current_site_is_missing, explicit_flow_update=False):
    context = context or {}
    base = context.get("last_flp_ir") or context.get("base_flp_ir")
    if not isinstance(base, dict):
        return

    base_site = base.get("site_ref")
    current_site = flp_ir.get("site_ref") or {}
    if (
        current_site_is_missing
        and not current_site.get("site_snapshot")
        and isinstance(base_site, dict)
    ):
        flp_ir["site_ref"] = copy.deepcopy(base_site)

    base_constraints = base.get("constraints") or {}
    current_constraints = flp_ir.get("constraints") or {}
    flp_ir["constraints"] = {
        "hard": _merge_requirement_items(
            base_constraints.get("hard"),
            current_constraints.get("hard"),
            identity_fn=_constraint_identity,
        ),
        "soft": _merge_requirement_items(
            base_constraints.get("soft"),
            current_constraints.get("soft"),
            identity_fn=_constraint_identity,
        ),
    }
    flp_ir["objectives"] = _merge_requirement_items(
        base.get("objectives"),
        flp_ir.get("objectives"),
        identity_fn=lambda item: item.get("objective_type"),
    )

    current_instance_id_order = [
        item.get("instance_id")
        for item in flp_ir.get("equipment_instances") or []
        if item.get("instance_id")
    ]
    current_instance_ids = set(current_instance_id_order)
    current_instance_ids.update(f["site_feature_id"] for f in ((flp_ir.get("site_ref") or {}).get("site_snapshot") or {}).get("site_features",[]) if f.get("site_feature_id"))
    base_flows = [
        copy.deepcopy(item)
        for item in base.get("flow_relations") or []
        if item.get("from_instance_id") in current_instance_ids
        and item.get("to_instance_id") in current_instance_ids
    ]
    if base_flows and not explicit_flow_update:
        # Existing reviewed process state is authoritative.  A fresh LLM
        # paraphrase may not change edge semantics implicitly.
        flp_ir["flow_relations"] = base_flows
        flp_ir["unresolved_items"] = [
            item
            for item in flp_ir.get("unresolved_items") or []
            if item.get("item_id")
            != "missing_process_relations_for_flow_objective"
        ]
    elif not flp_ir.get("flow_relations"):
        inherited_flows = [
            copy.deepcopy(item)
            for item in base.get("flow_relations") or []
            if item.get("from_instance_id") in current_instance_ids
            and item.get("to_instance_id") in current_instance_ids
        ]
        if inherited_flows:
            flp_ir["flow_relations"] = inherited_flows
            flp_ir["unresolved_items"] = [
                item
                for item in flp_ir.get("unresolved_items") or []
                if item.get("item_id")
                != "missing_process_relations_for_flow_objective"
            ]

    base_scope = base.get("optimization_scope") or {}
    scope = flp_ir.get("optimization_scope") or {}
    locked = [
        value
        for value in base_scope.get("locked_instances") or []
        if value in current_instance_ids
    ]
    scope["locked_instances"] = locked
    scope["movable_instances"] = [
        value
        for value in current_instance_id_order
        if value not in set(locked)
    ]
    scope["editable_zones"] = copy.deepcopy(
        base_scope.get("editable_zones") or scope.get("editable_zones") or []
    )
    flp_ir["optimization_scope"] = scope

    base_delta = base.get("delta_policy") or {}
    delta = flp_ir.get("delta_policy") or {}
    for field in ("max_movement_distance_m", "movement_cost_weight"):
        if delta.get(field) is None and base_delta.get(field) is not None:
            delta[field] = copy.deepcopy(base_delta[field])
    flp_ir["delta_policy"] = delta
    if flp_ir.get("flow_relations"):
        flp_ir["planning_preferences"]["generation_policy"] = "process_flow_based"
    flp_ir.setdefault("provenance", []).append(
        _provenance(
            "optimization_scope.base_layout_id",
            "computed",
            scope.get("base_layout_id") or "previous_layout",
            "existing_design_context",
            1.0,
        )
    )


def _merge_requirement_items(base_items, current_items, *, identity_fn):
    merged = []
    positions = {}
    for item in [*(base_items or []), *(current_items or [])]:
        if not isinstance(item, dict):
            continue
        identity = identity_fn(item)
        if identity in positions:
            merged[positions[identity]] = copy.deepcopy(item)
        else:
            positions[identity] = len(merged)
            merged.append(copy.deepcopy(item))
    return merged


def _constraint_identity(item):
    targets = json.dumps(
        item.get("target_refs") or [],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return item.get("constraint_type"), targets


def normalize_flp_ir(flp_ir, *, preserve_instance_ids=None):
    normalized = copy.deepcopy(flp_ir)
    normalized["unit_system"] = _unit_system()
    if preserve_instance_ids is None:
        preserve_instance_ids = (
            (normalized.get("task_meta") or {}).get("task_type")
            == "optimize_existing_layout"
        )
    if preserve_instance_ids:
        id_map = {
            str(item.get("instance_id")): str(item.get("instance_id"))
            for item in normalized.get("equipment_instances", [])
            if item.get("instance_id")
        }
    else:
        for index, instance in enumerate(
            normalized.get("equipment_instances", []),
            start=1,
        ):
            instance["instance_id"] = f"equipment_{index:03d}"
        id_map = {
            old.get("instance_id", f"equipment_{index:03d}"): f"equipment_{index:03d}"
            for index, old in enumerate(flp_ir.get("equipment_instances", []), start=1)
        }
    for scope_key in ["movable_instances", "locked_instances"]:
        normalized["optimization_scope"][scope_key] = [
            id_map.get(instance_id, instance_id)
            for instance_id in normalized["optimization_scope"].get(scope_key, [])
        ]
    for index, constraint in enumerate(normalized["constraints"].get("hard", []), start=1):
        constraint["constraint_id"] = f"constraint_{index:03d}"
        constraint["constraint_type"] = CONSTRAINT_SYNONYMS.get(
            constraint["constraint_type"], constraint["constraint_type"]
        )
    for index, constraint in enumerate(normalized["constraints"].get("soft", []), start=1):
        constraint["constraint_id"] = f"soft_constraint_{index:03d}"
        constraint["constraint_type"] = CONSTRAINT_SYNONYMS.get(
            constraint["constraint_type"], constraint["constraint_type"]
        )
    for index, objective in enumerate(normalized.get("objectives", []), start=1):
        objective["objective_id"] = f"objective_{index:03d}"
        objective["objective_type"] = OBJECTIVE_SYNONYMS.get(
            objective["objective_type"], objective["objective_type"]
        )
    for index, flow in enumerate(normalized.get("flow_relations", []), start=1):
        flow["flow_id"] = f"flow_{index:03d}"
        flow["from_instance_id"] = id_map.get(flow["from_instance_id"], flow["from_instance_id"])
        flow["to_instance_id"] = id_map.get(flow["to_instance_id"], flow["to_instance_id"])
    normalized["equipment_instances"].sort(key=lambda item: item["instance_id"])
    normalized["flow_relations"].sort(key=lambda item: item["flow_id"])
    normalized["constraints"]["hard"].sort(key=lambda item: item["constraint_id"])
    normalized["constraints"]["soft"].sort(key=lambda item: item["constraint_id"])
    normalized["objectives"].sort(key=lambda item: item["objective_id"])
    return normalized


def validate_flp_ir(flp_ir, adapter, schemas_dir):
    errors = []
    warnings = []
    _validate_schema(flp_ir, Path(schemas_dir), errors)
    if flp_ir is None:
        errors.append(_error("invalid_task", "FLP-IR is not available for this intent.", None))
        return _validation_report(errors, warnings)

    if not flp_ir.get("equipment_instances") and not flp_ir.get("unresolved_items"):
        errors.append(
            _error(
                "missing_equipment_instances",
                "At least one equipment instance is required before layout solving.",
                "equipment_instances",
            )
        )

    instance_ids = {item.get("instance_id") for item in flp_ir.get("equipment_instances", [])}
    for instance in flp_ir.get("equipment_instances", []):
        _validate_asset_reference(instance, adapter, errors)

    _validate_site(flp_ir, errors)
    _validate_constraints(flp_ir, instance_ids, errors)
    _validate_objectives(flp_ir, errors)
    _validate_flows(flp_ir, instance_ids, errors)
    _validate_provenance(flp_ir, errors)
    _validate_unresolved_items(flp_ir, errors)
    return _validation_report(errors, warnings)


def repair_or_clarify(flp_ir, validation_report):
    questions = []
    unsupported = []
    repair_actions = []
    issues = []
    auto_actions = {
        "overlap": "move_swap_or_resolve",
        "outside_boundary": "project_to_placement_region",
        "blocked_area_overlap": "move_away_from_blocked_area",
        "safety_clearance_violation": "increase_equipment_spacing",
        "invalid_orientation": "rotate_to_allowed_orientation",
        "missing_layout_instance": "global_resolve",
        "required_adjacency_violation": "move_related_equipment_closer",
        "forbidden_adjacency_violation": "separate_related_equipment",
    }
    clarification_types = {
        "missing_site",
        "ambiguous_asset",
        "missing_active_layout_ref",
        "missing_equipment_instances",
        "missing_asset_reference",
        "missing_required_field",
        "missing_engineering_field",
        "conflicting_constraints",
    }
    fatal_types = {
        "schema_error",
        "hallucinated_geometry",
        "invalid_flow_reference",
        "missing_provenance",
        "solver_no_layout",
        "asset_library_mismatch",
    }
    for error in validation_report.get("errors", []):
        error_type = error["error_type"]
        if error_type in auto_actions:
            classification = "auto_repair"
            repair_actions.append(
                {
                    "action_id": f"repair_action_{len(repair_actions) + 1:03d}",
                    "action_type": auto_actions[error_type],
                    "error_type": error_type,
                    "target_instance_ids": list(error.get("instance_ids") or []),
                    "zone_id": error.get("zone_id"),
                    "preserve_user_constraints": True,
                }
            )
        elif error_type in clarification_types:
            classification = "clarification_required"
        elif error_type in fatal_types or error_type.startswith("unsupported"):
            classification = "fatal"
        else:
            classification = error.get("repairability") or "fatal"
        issues.append(
            {
                "error_type": error_type,
                "classification": classification,
                "field_path": error.get("field_path"),
                "instance_ids": list(error.get("instance_ids") or []),
                "message": error.get("message"),
            }
        )
        if error_type == "missing_site":
            questions.append(
                "请提供矩形场地的长和宽，例如 20m x 12m；没有场地尺寸时不会生成默认 site。"
            )
        elif error_type == "ambiguous_asset":
            candidates = error.get("candidates", [])
            if candidates:
                questions.append(
                    "设备名称存在歧义，请从这些候选资产中选择一个：" + ", ".join(candidates)
                )
            else:
                questions.append("请提供设备在当前资产库中的准确名称、别名或 model_asset_id。")
        elif error_type == "missing_active_layout_ref":
            questions.append("该任务需要基于已有布局优化，请提供 active_layout_id 或 base_layout_ref。")
        elif error_type == "missing_equipment_instances":
            questions.append("请提供至少一个可布局设备，或先生成/选择一个已有布局作为优化基础。")
        elif error_type == "missing_asset_reference":
            questions.append("请提供可在资产库中解析的 model_asset_id，或明确允许系统重新检索候选资产。")
        elif error_type in {"missing_required_field", "missing_engineering_field"}:
            questions.append(error.get("message") or "请补充缺失的布局约束参数。")
        elif error_type == "conflicting_constraints":
            questions.append("当前硬约束互相冲突，请明确应保留或放宽哪一项约束。")
        elif error_type.startswith("unsupported") or error_type == "edit_operation_in_flp_ir":
            unsupported.append(error_type)

    classifications = {item["classification"] for item in issues}
    if not issues:
        disposition = "none"
    elif "fatal" in classifications:
        disposition = "fatal"
    elif "clarification_required" in classifications:
        disposition = "clarification_required"
    else:
        disposition = "auto_repair"
    return {
        "repaired_flp_ir": flp_ir,
        "repair_actions": repair_actions,
        "actions": repair_actions,
        "issues": issues,
        "disposition": disposition,
        "can_auto_repair": disposition == "auto_repair",
        "requires_user_clarification": disposition == "clarification_required",
        "fatal": disposition == "fatal",
        "source": "validator",
        "clarification_questions": questions,
        "unsupported_reason": ", ".join(sorted(set(unsupported))) if unsupported else None,
    }


def run_flp_ir_pipeline(user_message, dataset_dir, active_layout_id=None, rag_service=None):
    intent = route_intent(user_message, active_layout_id=active_layout_id)
    if intent["intent_type"] not in SUPPORTED_FLP_INTENTS:
        error_type = "edit_operation_in_flp_ir" if intent["intent_type"] == "edit_layout" else "unsupported_task"
        report = {
            "status": "UNSUPPORTED_TASK",
            "errors": [_error(error_type, f"{intent['intent_type']} is not compiled as FLP-IR.", None)],
            "warnings": [],
        }
        return {
            "intent": intent,
            "evidence_package": None,
            "flp_ir": None,
            "validation_report": report,
            "repair": repair_or_clarify(None, report),
            "handoff": "layout_edit_ir" if intent["intent_type"] == "edit_layout" else None,
        }

    try:
        adapter = AssetReferenceAdapter(dataset_dir, rag_service=rag_service)
        evidence = build_evidence_package(intent, adapter)
        flp_ir = normalize_flp_ir(compile_flp_ir(user_message, intent, evidence))
        report = validate_flp_ir(flp_ir, adapter, Path(dataset_dir) / "schemas")
        repair = repair_or_clarify(flp_ir, report)
        return {
            "intent": intent,
            "evidence_package": evidence,
            "flp_ir": flp_ir,
            "validation_report": report,
            "repair": repair,
            "handoff": None,
        }
    except RAGBlockedError as exc:
        return {
            "status": "BLOCKED",
            "blocker": exc.to_report(),
            "intent": intent,
            "evidence_package": None,
            "flp_ir": None,
            "validation_report": {
                "status": "BLOCKED",
                "errors": [{"error_type": "rag_blocked", "message": exc.message, "field_path": None}],
                "warnings": [],
            },
            "repair": {
                "repaired_flp_ir": None,
                "repair_actions": [],
                "clarification_questions": [],
                "unsupported_reason": None,
            },
            "handoff": None,
        }


def _match_assets_from_message(message, adapter):
    return adapter.find_matches(message)


def _explicit_asset_matches_from_message(message, adapter):
    matches = []
    seen = set()
    # Model IDs are namespaced by their source format.  Keep this matcher
    # format-agnostic so an explicit GLB or STL selection follows the same
    # grounded path as an explicit FBX selection.
    for match in re.finditer(
        r"\basset/[A-Za-z0-9_\-]+/[A-Za-z0-9_\-]+\b",
        str(message or ""),
    ):
        model_asset_id = match.group(0)
        if model_asset_id in seen or model_asset_id not in adapter.assets_by_id:
            continue
        seen.add(model_asset_id)
        asset = adapter.get_asset(model_asset_id)
        matches.append(
            {
                "surface_text": model_asset_id,
                "candidate_model_asset_ids": [_candidate(asset, 1.0, "id_exact")],
                "selected_model_asset_id": model_asset_id,
                "selected_library_id": asset.get("library_id"),
                "selected_site_id": asset.get("site_id"),
                "ambiguity_status": "resolved",
                "match_type": "id_exact",
            }
        )
    return matches


def _quantity_from_match(message, match):
    explicit = match.get("quantity")
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            return 1

    surfaces = [
        match.get("surface_text"),
        match.get("selected_model_asset_id"),
    ]
    text = str(message or "")
    for surface in [item for item in surfaces if item]:
        escaped = re.escape(str(surface))
        before = re.search(rf"(\d+)\s*(?:台|个|套|units?|pcs?)?\s*(?:of\s+)?{escaped}", text, re.IGNORECASE)
        after = re.search(rf"{escaped}\s*(\d+)\s*(?:台|个|套|units?|pcs?)?", text, re.IGNORECASE)
        match_obj = before or after
        if match_obj:
            return max(1, int(match_obj.group(1)))
    return 1


def _flatten_asset_matches(matches):
    candidates = []
    for match in matches:
        candidates.extend(match.get("candidate_model_asset_ids", []))
    candidates.sort(key=lambda item: (-item["score"], item["model_asset_id"]))
    return candidates


def _candidates_to_match(surface_text, candidates):
    if not candidates:
        return [
            {
                "surface_text": surface_text,
                "candidate_model_asset_ids": [],
                "selected_model_asset_id": None,
                "selected_library_id": None,
                "selected_site_id": None,
                "ambiguity_status": "unresolved",
            }
        ]
    candidates = sorted(candidates, key=lambda item: (-item["score"], item["model_asset_id"]))
    top = candidates[0]
    second_score = candidates[1]["score"] if len(candidates) > 1 else -1.0
    if top["score"] - second_score < 0.05:
        return [
            {
                "surface_text": surface_text,
                "candidate_model_asset_ids": candidates,
                "selected_model_asset_id": None,
                "selected_library_id": None,
                "selected_site_id": None,
                "ambiguity_status": "ambiguous",
            }
        ]
    return [
        {
            "surface_text": surface_text,
            "candidate_model_asset_ids": candidates,
            "selected_model_asset_id": top["model_asset_id"],
            "selected_library_id": top.get("library_id"),
            "selected_site_id": top.get("site_id"),
            "ambiguity_status": "resolved",
        }
    ]


def _route_english_intent(message_lower):
    if (
        any(
            keyword in message_lower
            for keyword in ("repair", "fix", "resolve")
        )
        and any(
            keyword in message_lower
            for keyword in (
                "violation",
                "overlap",
                "out of bounds",
                "forbidden zone",
                "corridor",
                "clearance",
            )
        )
    ):
        return "constraint_repair"
    if any(keyword in message_lower for keyword in ["validate", "check layout", "verify layout"]):
        return "validate_layout"
    if any(keyword in message_lower for keyword in ["evaluate", "assess layout", "score layout"]):
        return "evaluate_layout"
    if any(keyword in message_lower for keyword in ["current layout", "existing layout", "active layout"]):
        if any(keyword in message_lower for keyword in ["optimize", "improve", "minimize", "reduce"]):
            return "optimize_existing_layout"
    if any(keyword in message_lower for keyword in ["optimize", "minimize", "shorten", "reduce travel", "best layout"]):
        return "optimize_layout_from_scratch"
    if any(keyword in message_lower for keyword in ["generate", "layout", "lay out", "arrange", "place", "rough design", "feasible layout"]):
        return "generate_feasible_layout"
    return "unknown_or_unsupported"


def _compile_site_ref(
    evidence_package,
    intent,
    unresolved_items,
    provenance,
    *,
    context=None,
):
    site_snapshot = _site_snapshot_from_context(context)
    snapshot_dimensions = _site_snapshot_dimensions(site_snapshot)
    if site_snapshot and snapshot_dimensions:
        provenance.append(
            _provenance(
                "site_ref",
                "manual_override",
                site_snapshot.get("site_id") or intent["intent_id"],
                "scene_context.site_snapshot",
                1.0,
            )
        )
        bound_site_asset = site_snapshot.get("bound_site_asset") or {}
        return {
            "site_id": site_snapshot.get("site_id") or "SITE_CONTEXT_001",
            "site_source": site_snapshot.get("site_definition_mode") or "user_defined",
            "bound_model_asset_id": bound_site_asset.get("model_asset_id"),
            "default_dimensions_m": snapshot_dimensions,
            "site_snapshot": site_snapshot,
        }
    site_facts = evidence_package.get("site_facts_from_user", [])
    if not site_facts:
        raise RAGBlockedError(
            "missing_authoritative_site",
            "Layout compilation requires an authoritative site snapshot or explicit site dimensions.",
            details={"field": "site_snapshot"},
        )

    fact = site_facts[0]
    provenance.append(_provenance("site_ref", "user_input", intent["intent_id"], "user_message", 0.95))
    dimensions = copy.deepcopy(fact["value"])
    inline_snapshot = {
        "schema_version": "inline-site-v1",
        "site_id": "SITE_INLINE_001",
        "site_definition_mode": "user_defined",
        "boundary": {
            "boundary_id": "SITE_INLINE_001/boundary",
            "dimensions_m": dimensions,
            "pose": {"x_m": 0.0, "y_m": 0.0, "z_m": 0.0},
        },
        "site_features": [],
        "zones": [],
        "embedded_equipment_instances": [],
    }
    return {
        "site_id": "SITE_INLINE_001",
        "site_source": "user_defined",
        "bound_model_asset_id": None,
        "default_dimensions_m": dimensions,
        "site_snapshot": inline_snapshot,
    }


def _site_snapshot_from_context(context):
    context = context or {}
    overrides = context.get("semantic_overrides") or {}
    snapshot = (
        context.get("site_snapshot")
        or overrides.get("site_snapshot")
        or context.get("site_template")
    )
    return copy.deepcopy(snapshot) if isinstance(snapshot, dict) else None


def _site_snapshot_dimensions(site_snapshot):
    if not isinstance(site_snapshot, dict):
        return None
    boundary = site_snapshot.get("boundary") or {}
    dimensions = boundary.get("dimensions_m") or site_snapshot.get("dimensions_m") or {}
    try:
        width = float(dimensions.get("width_m"))
        depth = float(dimensions.get("depth_m"))
    except (TypeError, ValueError):
        return None
    if width <= 0 or depth <= 0:
        return None
    return {"width_m": width, "depth_m": depth}


def _site_evidence_is_missing(evidence_package):
    facts = evidence_package.get("site_facts_from_user") or []
    return not facts


def _compile_equipment_instances(evidence_package, provenance, *, context=None):
    context = context or {}
    directory = context.get("instance_directory") or []
    directory_by_model_index = {}
    directory_models = set()
    for item in directory:
        if not isinstance(item, dict):
            raise ValueError("instance_directory entries must be objects")
        instance_id = str(item.get("instance_id") or "").strip()
        model_asset_id = str(item.get("model_asset_id") or "").strip()
        quantity_index = item.get("quantity_index")
        if (
            not instance_id
            or not model_asset_id
            or isinstance(quantity_index, bool)
            or not isinstance(quantity_index, int)
            or quantity_index < 1
        ):
            raise ValueError("instance_directory contains an invalid instance identity")
        key = (model_asset_id, quantity_index)
        if key in directory_by_model_index:
            raise ValueError("instance_directory repeats a model quantity index")
        directory_by_model_index[key] = instance_id
        directory_models.add(model_asset_id)
    instances = []
    used_instance_ids = set()
    for fact in evidence_package.get("asset_facts", []):
        quantity = max(1, int(fact.get("quantity") or 1))
        base_planning = fact.get("planning_fields", {})
        indices=sorted(index for model,index in directory_by_model_index if model==fact["model_asset_id"]) if fact["model_asset_id"] in directory_models else list(range(1,quantity+1))
        if len(indices)!=quantity:
            raise ValueError("instance_directory does not cover the declared model quantity")
        for quantity_index in indices:
            planning = copy.deepcopy(base_planning)
            for key, value in (fact.get("instance_planning_overrides", {}).get(str(quantity_index)) or {}).items():
                if isinstance(value, dict):
                    planning[key] = dict(planning.get(key) or {}, **value)
                else:
                    planning[key] = copy.deepcopy(value)
            instance_number = len(instances) + 1
            directory_key = (fact["model_asset_id"], quantity_index)
            instance_id = directory_by_model_index.get(directory_key)
            if instance_id is None:
                if fact["model_asset_id"] in directory_models:
                    raise ValueError(
                        "instance_directory does not cover the declared model quantity range"
                    )
                instance_id = f"equipment_{instance_number:03d}"
            if instance_id in used_instance_ids:
                raise ValueError("instance_directory repeats an instance_id")
            used_instance_ids.add(instance_id)
            instances.append(
                {
                    "instance_id": instance_id,
                    "model_asset_id": fact["model_asset_id"],
                    "library_id": fact["library_id"],
                    "quantity_index": quantity_index,
                    "role": (fact.get("instance_roles") or {}).get(str(quantity_index)),
                    "locked": False,
                    "initial_pose": None,
                    "fixed_pose": None,
                    "allowed_regions": [],
                    "required_ports": [],
                    "geometry": copy.deepcopy(fact["geometry"]),
                    "maintenance_clearance_m": copy.deepcopy(planning.get("maintenance_clearance_m")),
                    "safety_clearance_m": copy.deepcopy(planning.get("safety_clearance_m")),
                    "allowed_orientations_deg": copy.deepcopy(planning.get("allowed_orientations_deg", [])),
                }
            )
            provenance.append(
                _provenance(
                    f"equipment_instances[{instance_number - 1}].geometry",
                    "model_extraction",
                    fact["model_asset_id"],
                    "geometry",
                    fact["geometry"].get("confidence"),
                )
            )
            quantity_provenance = fact.get("quantity_provenance")
            if quantity_provenance:
                provenance.append(
                    _provenance(
                        f"equipment_instances[{instance_number - 1}].quantity_index",
                        quantity_provenance["source_type"],
                        quantity_provenance["source_id"],
                        quantity_provenance["source_field"],
                        quantity_provenance.get("confidence"),
                    )
                )
            selection_provenance = fact.get("selection_provenance")
            if selection_provenance:
                provenance.append(
                    _provenance(
                        f"equipment_instances[{instance_number - 1}].model_asset_id",
                        selection_provenance["source_type"],
                        selection_provenance["source_id"],
                        selection_provenance["source_field"],
                        selection_provenance.get("confidence"),
                    )
                )
    return instances


def _compile_existing_layout_equipment_instances(context, provenance):
    source_flp_ir = context.get("last_flp_ir") or context.get("base_flp_ir")
    if not source_flp_ir or not source_flp_ir.get("equipment_instances"):
        return []

    instances = copy.deepcopy(source_flp_ir["equipment_instances"])
    model_counts = {}
    for instance in instances:
        model_asset_id = str(instance.get("model_asset_id") or "").strip()
        model_counts[model_asset_id] = model_counts.get(model_asset_id, 0) + 1
        quantity_index = instance.get("quantity_index")
        if quantity_index is None:
            instance["quantity_index"] = model_counts[model_asset_id]
        elif (
            isinstance(quantity_index, bool)
            or not isinstance(quantity_index, int)
            or quantity_index < 1
        ):
            raise ValueError(
                "existing layout equipment has an invalid quantity_index"
            )

    directory = context.get("instance_directory")
    if directory:
        expected = {
            (
                str(item.get("instance_id") or ""),
                str(item.get("model_asset_id") or ""),
                item.get("quantity_index"),
            )
            for item in directory
            if isinstance(item, dict)
        }
        actual = {
            (
                str(item.get("instance_id") or ""),
                str(item.get("model_asset_id") or ""),
                item.get("quantity_index"),
            )
            for item in instances
        }
        if expected != actual or len(expected) != len(instances):
            raise ValueError(
                "instance_directory disagrees with authoritative existing layout equipment"
            )
    source_layout = context.get("last_layout") or context.get("seed_layout") or {}
    placements_by_id = {
        placement.get("instance_id"): placement
        for placement in source_layout.get("placements", [])
        if placement.get("instance_id")
    }
    for instance in instances:
        placement = placements_by_id.get(instance.get("instance_id"))
        if placement:
            instance["initial_pose"] = _pose_from_layout_placement(placement)

    source_id = source_layout.get("layout_id") or source_flp_ir.get("task_meta", {}).get("task_id") or "previous_layout"
    provenance.append(
        _provenance(
            "equipment_instances",
            "computed",
            source_id,
            "existing_layout_equipment_instances",
            0.85,
        )
    )
    for index, instance in enumerate(instances):
        provenance.append(
            _provenance(
                f"equipment_instances[{index}].geometry",
                "computed",
                source_id,
                "existing_layout_equipment_geometry",
                instance.get("geometry", {}).get("confidence"),
            )
        )
    return instances


def _cross_check_existing_asset_declarations(
    evidence_package,
    equipment_instances,
    unresolved_items,
):
    declared = {}
    for fact in evidence_package.get("asset_facts") or []:
        model_asset_id = str(fact.get("model_asset_id") or "").strip()
        if not model_asset_id:
            continue
        quantity = fact.get("quantity")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            quantity = 1
        declared[model_asset_id] = declared.get(model_asset_id, 0) + quantity
    if not declared:
        return

    existing = {}
    for item in equipment_instances:
        model_asset_id = str(item.get("model_asset_id") or "").strip()
        existing[model_asset_id] = existing.get(model_asset_id, 0) + 1
    if declared == existing:
        return
    unresolved_items.append(
        _unresolved_item(
            "existing_layout_asset_declaration_mismatch",
            "conflicting_value",
            "equipment_instances",
            "Asset mentions disagree with the authoritative existing-layout instance set.",
            "blocking",
            [],
            "ask_user",
        )
    )


def _pose_from_layout_placement(placement):
    position = placement.get("position") or {}
    rotation = placement.get("rotation") or {}
    return {
        "x_m": _float_or_default(position.get("x_m"), 0.0),
        "y_m": _float_or_default(position.get("y_m"), 0.0),
        "z_m": _float_or_default(position.get("z_m"), 0.0),
        "roll_deg": _float_or_default(rotation.get("roll_deg"), 0.0),
        "pitch_deg": _float_or_default(rotation.get("pitch_deg"), 0.0),
        "yaw_deg": _float_or_default(placement.get("yaw_deg", rotation.get("yaw_deg")), 0.0),
        "coordinate_frame_id": placement.get("coordinate_frame_id") or "site",
    }


def _float_or_default(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _compile_constraints(evidence_package, equipment_instances, provenance):
    hard = []
    soft = []
    for item in evidence_package.get("matched_constraints", []):
        constraint = {
            "constraint_id": f"constraint_{len(hard) + len(soft) + 1:03d}",
            "constraint_type": CONSTRAINT_SYNONYMS.get(item["constraint_type"], item["constraint_type"]),
            "target_refs": [{"target_type": "all_equipment", "target_id": "all"}],
            "parameters": {},
            "source": "user_input",
            "evidence_refs": ["source_request.raw_user_message"],
        }
        if item["default_severity"] == "hard":
            hard.append(constraint)
            field_path = f"constraints.hard[{len(hard) - 1}]"
        else:
            constraint["weight"] = None
            soft.append(constraint)
            field_path = f"constraints.soft[{len(soft) - 1}]"
        provenance.append(_provenance(field_path, "user_input", "INTENT_001", "user_message", 0.9))
    existing_by_type = {
        item.get("constraint_type"): item
        for item in [*hard, *soft]
        if isinstance(item, dict) and item.get("constraint_type")
    }
    for item in evidence_package.get("semantic_constraints", []):
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get("constraint_type") or "").strip()
        if not raw_type:
            continue
        constraint_type = CONSTRAINT_SYNONYMS.get(raw_type, raw_type)
        source_text = str(item.get("source_text") or "").strip()
        severity = str(item.get("severity") or "hard").casefold()
        if constraint_type in existing_by_type:
            existing_by_type[constraint_type]["parameters"].update(copy.deepcopy(item.get("parameters") or {}))
            provenance.append(
                _provenance(
                    "constraints",
                    "user_input" if source_text and source_text != "default" else "computed",
                    "INTENT_001",
                    source_text or "semantic_intent",
                    1.0 if source_text and source_text != "default" else 0.8,
                )
            )
            continue
        constraint = {
            "constraint_id": f"constraint_{len(hard) + len(soft) + 1:03d}",
            "constraint_type": constraint_type,
            "target_refs": [{"target_type": "all_equipment", "target_id": "all"}],
            "parameters": copy.deepcopy(item.get("parameters") or {}),
            "source": "user_input" if source_text and source_text != "default" else "computed",
            "evidence_refs": ["source_request.raw_user_message"],
        }
        if severity == "hard":
            hard.append(constraint)
            field_path = f"constraints.hard[{len(hard) - 1}]"
        else:
            constraint["weight"] = None
            soft.append(constraint)
            field_path = f"constraints.soft[{len(soft) - 1}]"
        provenance.append(
            _provenance(
                field_path,
                "user_input" if source_text and source_text != "default" else "computed",
                "INTENT_001",
                source_text or "semantic_intent",
                1.0 if source_text and source_text != "default" else 0.8,
            )
        )
        existing_by_type[constraint_type] = constraint
    clearance_by_instance = {
        instance["instance_id"]: clearance
        for instance in equipment_instances
        for clearance in [_instance_clearance_radius(instance)]
        if clearance > 0
    }
    if clearance_by_instance:
        hard.append(
            {
                "constraint_id": f"constraint_{len(hard) + len(soft) + 1:03d}",
                "constraint_type": "minimum_clearance",
                "target_refs": [{"target_type": "all_equipment", "target_id": "all"}],
                "parameters": {"clearance_by_instance": clearance_by_instance},
                "source": "computed",
                "evidence_refs": ["equipment_instances.clearance"],
            }
        )
        provenance.append(
            _provenance(
                f"constraints.hard[{len(hard) - 1}]",
                "computed",
                "INTENT_001",
                "equipment_clearance",
                0.8,
            )
        )
    return hard, soft


def _append_natural_site_constraints(hard_constraints, requirements, provenance, intent_id):
    existing = {
        item.get("constraint_type")
        for item in hard_constraints
        if isinstance(item, dict)
    }
    for requirement in requirements:
        constraint_type = requirement.get("constraint_type")
        if constraint_type in existing:
            continue
        hard_constraints.append(
            {
                "constraint_id": f"constraint_{len(hard_constraints) + 1:03d}",
                "constraint_type": constraint_type,
                "target_refs": [{"target_type": "all_equipment", "target_id": "all"}],
                "parameters": {},
                "source": "user_input",
                "evidence_refs": ["source_request.raw_user_message"],
            }
        )
        provenance.append(
            _provenance(
                f"constraints.hard[{len(hard_constraints) - 1}]",
                "user_input",
                intent_id,
                requirement.get("source_text") or "user_message",
                1.0,
            )
        )
        existing.add(constraint_type)


def _instance_clearance_radius(instance):
    values = []
    for clearance_key in ["maintenance_clearance_m", "safety_clearance_m"]:
        clearance = instance.get(clearance_key) or {}
        for field in ["front_m", "rear_m", "left_m", "right_m"]:
            value = clearance.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
    return max(values or [0.0])


def _compile_objectives(evidence_package, provenance):
    objectives = []
    for item in evidence_package.get("matched_objectives", []):
        objectives.append(
            {
                "objective_id": f"objective_{len(objectives) + 1:03d}",
                "objective_type": OBJECTIVE_SYNONYMS.get(item["objective_type"], item["objective_type"]),
                "sense": item["sense"],
                "weight": 1.0,
                "parameters": copy.deepcopy(item.get("parameters") or {}),
                "source": "user_input",
                "evidence_refs": ["source_request.raw_user_message"],
            }
        )
        provenance.append(
            _provenance(
                f"objectives[{len(objectives) - 1}]",
                "user_input",
                "INTENT_001",
                "user_message",
                0.9,
            )
        )
    return objectives


def canonical_objective_type_from_literal(user_message, source_text):
    """Map only exact recognized objective wording to the normal vocabulary."""

    source = _literal_key(source_text)
    if not source or source not in _literal_key(user_message):
        return None
    if relocation_requested(source_text) and agv_cost_requested(source_text):
        # A shared quote may legitimately support two distinct objectives.
        # It cannot identify which one an individual objective node denotes.
        return None
    if relocation_requested(source_text):
        return RELOCATION_EFFORT_OBJECTIVE
    if re.search(r"\b(?:make|keep)\b[^.]{0,90}\bmoves?\s+as\s+small\s+as\s+possible\b|\bminimi[sz]e\s+(?:the\s+)?(?:required\s+|total\s+)?(?:layout\s+)?(?:movement|displacement)\b",str(source_text),re.I):
        return "minimize_layout_change"
    if agv_cost_requested(source_text):
        return AGV_TRANSPORT_OBJECTIVE
    if (
        _high_flow_distance_excess_clause_present(source_text)
        and _high_flow_distance_excess_parameters(source_text) is None
    ):
        return None
    if _high_flow_distance_excess_parameters(source_text) is not None:
        return "minimize_high_flow_distance_excess"
    if _weighted_flow_objective_source(source_text):
        return "minimize_weighted_flow_distance"
    if re.search(r"\b(?:material handling|logistics)\b", source, re.IGNORECASE) and re.search(
        r"\b(?:distance|travel)\b", source, re.IGNORECASE
    ):
        return "minimize_material_handling_distance"
    return None


def _weighted_flow_objective_source(text):
    source = str(text or "")
    lowered = source.casefold()
    return bool(
        re.search(r"\b(?:travel|distance)\b", lowered)
        and "weighted" in lowered
        and re.search(r"\bbatch(?:es)?\b", lowered)
    )


def _high_flow_distance_excess_parameters(text):
    source = str(text or "")
    match = re.search(
        r"\b(?:connection|transfer|flow)\b[^.!?]{0,160}?"
        r"\b(?:at\s+least|>=|greater\s+than\s+or\s+equal\s+to)\s*"
        r"(?P<flow>\d+(?:\.\d+)?)\s*batches?\s+per\s+day\b"
        r"[^.!?]{0,200}?\b(?:travel|distance)\b[^.!?]{0,100}?"
        r"\bbeyond\s*(?P<distance>\d+(?:\.\d+)?)\s*m\b",
        source,
        re.IGNORECASE,
    )
    if not match:
        return None
    flow_threshold = float(match.group("flow"))
    distance_threshold = float(match.group("distance"))
    if (
        not math.isfinite(flow_threshold)
        or not math.isfinite(distance_threshold)
        or flow_threshold <= 0.0
        or distance_threshold <= 0.0
    ):
        return None
    return {
        "flow_threshold_batches_per_day": flow_threshold,
        "distance_threshold_m": distance_threshold,
    }


def _high_flow_distance_excess_clause_present(text):
    return bool(
        re.search(
            r"\b(?:connection|transfer|flow)\b[^.!?]{0,160}?"
            r"\b(?:at\s+least|>=|greater\s+than\s+or\s+equal\s+to)\s*"
            r"\d+(?:\.\d+)?\s*batches?\s+per\s+day\b"
            r"[^.!?]{0,240}?\bbeyond\b",
            str(text or ""),
            re.IGNORECASE,
        )
    )


def _literal_key(value):
    return " ".join(str(value or "").casefold().split())


def _compile_flow_relations(
    equipment_instances,
    objectives,
    provenance,
    *,
    process_relations=None,
    site_features=None,
    unresolved_items=None,
    allow_instance_id_endpoints=True,
):
    unresolved_items = unresolved_items if unresolved_items is not None else []
    objective_types = {objective.get("objective_type") for objective in objectives}
    needs_flow = any(
        objective_type
        in {
            "minimize_material_handling_distance",
            "minimize_weighted_flow_distance",
            "minimize_high_flow_distance_excess",
            "minimize_cross_flow",
            "maximize_process_order_consistency",
        }
        for objective_type in objective_types
    )
    process_relations = [
        item for item in process_relations or [] if isinstance(item, dict)
    ]
    if not process_relations:
        if needs_flow:
            unresolved_items.append(
                _unresolved_item(
                    "missing_process_relations_for_flow_objective",
                    "missing_field",
                    "flow_relations",
                    "A flow-distance objective requires reviewed process or material-flow relations.",
                    "blocking",
                    [],
                    "ask_user",
                )
            )
        return []

    instance_ids = {
        str(item.get("instance_id"))
        for item in equipment_instances
        if item.get("instance_id")
    }
    flows = []
    flow_index_by_identity = {}
    seen_evidence = set()
    for relation_index, relation in enumerate(process_relations, start=1):
        from_ids = _resolve_process_endpoint(
            relation,
            "from",
            equipment_instances,
            instance_ids,
            allow_instance_id=allow_instance_id_endpoints,
            site_features=site_features,
        )
        to_ids = _resolve_process_endpoint(
            relation,
            "to",
            equipment_instances,
            instance_ids,
            allow_instance_id=allow_instance_id_endpoints,
            site_features=site_features,
        )
        if not from_ids or not to_ids:
            unresolved_items.append(
                _unresolved_item(
                    f"unresolved_process_relation_{relation_index:03d}",
                    "ambiguous_reference",
                    f"process_relations[{relation_index - 1}]",
                    "A reviewed process relation could not be mapped to equipment instances.",
                    "blocking",
                    [],
                    "manual_review",
                )
            )
            continue
        flow_type = str(relation.get("flow_type") or "material")
        raw_weight = relation.get("flow_weight", 1.0)
        if isinstance(raw_weight, bool):
            flow_weight = float("nan")
        else:
            try:
                flow_weight = float(raw_weight)
            except (TypeError, ValueError):
                flow_weight = float("nan")
        if not math.isfinite(flow_weight):
            unresolved_items.append(
                _unresolved_item(
                    f"invalid_process_relation_weight_{relation_index:03d}",
                    "missing_field",
                    f"process_relations[{relation_index - 1}].flow_weight",
                    "A reviewed process relation requires a finite numeric flow_weight.",
                    "blocking",
                    [],
                    "ask_user",
                )
            )
            continue
        carrier_requested=relation.get("carrier_model_asset_id") or relation.get("carrier_instance_id")
        carrier_ids = _resolve_process_endpoint(relation,"carrier",equipment_instances,instance_ids) if carrier_requested else []
        if carrier_requested and len(carrier_ids)!=1:
            unresolved_items.append(_unresolved_item(f"unresolved_transport_resource_{relation_index}","ambiguous_reference",f"process_relations[{relation_index-1}]","Specify which named transport resource performs this transfer.","blocking",[],"ask_user"))
            continue
        carrier_id = carrier_ids[0] if carrier_ids else None
        for from_instance_id, to_instance_id in _process_endpoint_pairs(
            from_ids,
            to_ids,
        ):
            if from_instance_id == to_instance_id:
                continue
            identity = (from_instance_id, to_instance_id, flow_type, relation.get("transport_mode"), carrier_id)
            evidence_identity = (
                identity,
                _literal_key(relation.get("evidence_ref") or relation.get("source_text")),
                flow_weight,
                relation.get("sequence_order"),
                bool(relation.get("required_connection", False)),
            )
            if evidence_identity in seen_evidence:
                continue
            seen_evidence.add(evidence_identity)
            flow_index = flow_index_by_identity.get(identity)
            if flow_index is None:
                flow_index = len(flows)
                flow_index_by_identity[identity] = flow_index
                flows.append(
                    {
                        "flow_id": f"flow_{len(flows) + 1:03d}",
                        "from_instance_id": from_instance_id,
                        "to_instance_id": to_instance_id,
                        "flow_type": flow_type,
                        "flow_weight": flow_weight,
                        "sequence_order": relation.get("sequence_order"),
                        "required_connection": bool(
                            relation.get("required_connection", False)
                        ),
                    }
                )
            else:
                flows[flow_index]["flow_weight"] += flow_weight
                flows[flow_index]["required_connection"] = bool(
                    flows[flow_index].get("required_connection")
                    or relation.get("required_connection", False)
                )
            if relation.get("transport_mode"):
                flows[flow_index].update(transport_mode=relation["transport_mode"],carrier_instance_id=carrier_id,carrier_source_text=relation.get("carrier_source_text"))
            provenance.append(
                _process_relation_provenance(
                    relation,
                    flow_index=flow_index,
                    relation_index=relation_index,
                )
            )
    return flows


def _process_relation_provenance(relation, *, flow_index, relation_index):
    evidence_source = str(relation.get("evidence_source") or "user_input").casefold()
    source_type = {
        "user_input": "user_input",
        "engineering_rag": "rule_library",
        "human_engineering_review": "manual_override",
        "human_review": "manual_override",
        "manual_override": "manual_override",
    }.get(evidence_source, "user_input")
    source_id = str(
        relation.get("evidence_ref")
        or relation.get("source_text")
        or f"process_relation_{relation_index:03d}"
    )
    return _provenance(
        f"flow_relations[{flow_index}]",
        source_type,
        source_id,
        "reviewed_process_relations",
        1.0 if source_type in {"user_input", "manual_override"} else 0.95,
    )


def _resolve_process_endpoint(
    relation,
    prefix,
    equipment_instances,
    instance_ids,
    *,
    allow_instance_id=True,
    site_features=None,
):
    site_name = relation.get(f"{prefix}_site_feature_name")
    if site_name:
        feature_id = resolve_facility_ref(site_features, site_name)
        return [feature_id] if feature_id else []
    instance_id = relation.get(f"{prefix}_instance_id")
    if instance_id and allow_instance_id:
        value = str(instance_id)
        facility_ids={f.get("site_feature_id") for f in site_features or []}
        return [value] if value in instance_ids or value in facility_ids else []
    model_asset_id = relation.get(f"{prefix}_model_asset_id")
    if not model_asset_id:
        return []
    matches = [
        item
        for item in equipment_instances
        if item.get("model_asset_id") == model_asset_id
    ]
    quantity_index = relation.get(f"{prefix}_quantity_index")
    if quantity_index is None:
        return [str(matches[0]["instance_id"])] if len(matches) == 1 else []
    if (
        isinstance(quantity_index, bool)
        or not isinstance(quantity_index, int)
        or quantity_index < 1
    ):
        return []
    matches = [
        item for item in matches if item.get("quantity_index") == quantity_index
    ]
    return [str(item["instance_id"]) for item in matches if item.get("instance_id")]


def _process_endpoint_pairs(from_ids, to_ids):
    if len(from_ids) != 1 or len(to_ids) != 1:
        return []
    return [(from_ids[0], to_ids[0])]


def _requested_missing_fields(message, asset_facts):
    requested_paths = [
        field_path
        for field_path, pattern in FIELD_REQUEST_PATTERNS
        if pattern.search(message)
    ]
    unresolved = []
    for fact in asset_facts:
        for field_path in requested_paths:
            if not _is_missing_engineering_value(_nested_value(fact, field_path)):
                continue
            asset_id = fact["model_asset_id"]
            unresolved.append(
                _unresolved_item(
                    f"missing_engineering_field_{len(unresolved) + 1:03d}",
                    "missing_field",
                    f"asset_facts[{asset_id}].{field_path}",
                    f"{field_path} is not verified for {asset_id}.",
                    "blocking",
                    [],
                    "ask_user",
                )
            )
    return unresolved


def _nested_value(value, field_path):
    current = value
    for part in field_path.split("."):
        current = current.get(part) if isinstance(current, dict) else None
    return current


def _is_missing_engineering_value(value):
    if value in (None, [], {}):
        return True
    if isinstance(value, dict):
        return not any(
            item is not None
            for key, item in value.items()
            if key not in {"source", "confidence"}
        )
    return False


def _validate_schema(flp_ir, schemas_dir, errors):
    if flp_ir is None:
        return
    schema_path = schemas_dir / "task" / "flp_ir.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, registry=_schema_registry(schemas_dir))
    for error in sorted(validator.iter_errors(flp_ir), key=lambda item: list(item.path)):
        errors.append(_error("schema_error", error.message, ".".join(str(part) for part in error.path)))


def _validate_asset_reference(instance, adapter, errors):
    model_asset_id = instance.get("model_asset_id")
    if not model_asset_id:
        errors.append(_error("missing_asset_reference", "Missing model_asset_id.", "equipment_instances"))
        return
    try:
        asset = adapter.get_asset(model_asset_id)
    except KeyError:
        errors.append(
            _error("missing_asset_reference", f"Unknown model_asset_id: {model_asset_id}", "equipment_instances")
        )
        return
    if instance.get("library_id") != asset.get("library_id"):
        errors.append(
            _error("asset_library_mismatch", "Instance library_id does not match asset library_id.", "library_id")
        )
    if instance.get("geometry") != asset.get("geometry"):
        errors.append(
            _error(
                "hallucinated_geometry",
                "Instance geometry differs from the asset library geometry.",
                "equipment_instances.geometry",
            )
        )


def _validate_site(flp_ir, errors):
    site_ref = flp_ir.get("site_ref") or {}
    if not site_ref.get("site_id") or site_ref.get("site_id") == "SITE_UNRESOLVED":
        errors.append(_error("missing_site", "Missing site reference.", "site_ref"))
        return


def _validate_constraints(flp_ir, instance_ids, errors):
    for severity in ["hard", "soft"]:
        for constraint in flp_ir.get("constraints", {}).get(severity, []):
            if constraint.get("constraint_type") not in SUPPORTED_CONSTRAINTS:
                errors.append(
                    _error(
                        "unsupported_constraint",
                        f"Unsupported constraint type: {constraint.get('constraint_type')}",
                        "constraints",
                    )
                )
            for target in constraint.get("target_refs", []):
                if target.get("target_type") == "equipment_instance" and target.get("target_id") not in instance_ids:
                    errors.append(
                        _error(
                            "invalid_constraint_target",
                            f"Constraint target does not exist: {target.get('target_id')}",
                            "constraints.target_refs",
                        )
                    )


def _validate_objectives(flp_ir, errors):
    for objective in flp_ir.get("objectives", []):
        if objective.get("objective_type") not in SUPPORTED_OBJECTIVES:
            errors.append(
                _error(
                    "unsupported_objective",
                    f"Unsupported objective type: {objective.get('objective_type')}",
                    "objectives.objective_type",
                )
            )
            continue
        if objective.get("objective_type") == AGV_TRANSPORT_OBJECTIVE:
            try:
                validate_agv_parameters(objective.get("parameters") or {})
            except (ValueError, KeyError, TypeError) as error:
                errors.append(_error("invalid_agv_cost_parameters", str(error), "objectives.parameters"))
        if objective.get("objective_type") == RELOCATION_EFFORT_OBJECTIVE:
            try:
                validate_relocation_parameters(objective.get("parameters") or {}, [p["instance_id"] for p in flp_ir.get("equipment_instances") or []])
            except (ValueError, KeyError, TypeError) as error:
                errors.append(_error("invalid_relocation_effort_parameters", str(error), "objectives.parameters"))
        if objective.get("objective_type") == "minimize_high_flow_distance_excess":
            parameters = objective.get("parameters") or {}
            try:
                flow_threshold = float(
                    parameters.get("flow_threshold_batches_per_day")
                )
                distance_threshold = float(parameters.get("distance_threshold_m"))
            except (TypeError, ValueError):
                flow_threshold = distance_threshold = float("nan")
            if (
                not math.isfinite(flow_threshold)
                or not math.isfinite(distance_threshold)
                or flow_threshold <= 0.0
                or distance_threshold <= 0.0
            ):
                errors.append(
                    _error(
                        "invalid_high_flow_distance_excess_parameters",
                        "minimize_high_flow_distance_excess requires positive finite flow_threshold_batches_per_day and distance_threshold_m values.",
                        "objectives.parameters",
                    )
                )


def _validate_flows(flp_ir, instance_ids, errors):
    site_ids = set(fixed_flow_endpoints({}, flp_ir))
    instance_ids = set(instance_ids) | site_ids
    for flow in flp_ir.get("flow_relations", []):
        if flow.get("carrier_instance_id") and flow["carrier_instance_id"] not in instance_ids-site_ids:
            errors.append(_error("invalid_transport_resource","A carrier must refer to a placed equipment resource, not a floor feature.","flow_relations"))
        if flow.get("from_instance_id") not in instance_ids or flow.get("to_instance_id") not in instance_ids:
            errors.append(
                _error(
                    "invalid_flow_reference",
                    "Flow relation references a missing equipment instance.",
                    "flow_relations",
                )
            )


def _validate_provenance(flp_ir, errors):
    provenance = flp_ir.get("provenance", [])
    if not provenance:
        errors.append(_error("missing_provenance", "FLP-IR has no provenance entries.", "provenance"))
        return
    field_paths = [item.get("field_path", "") for item in provenance]
    if flp_ir.get("site_ref", {}).get("site_id") and not any(path == "site_ref" for path in field_paths):
        errors.append(_error("missing_provenance", "Site reference lacks provenance.", "site_ref"))
    for index, _instance in enumerate(flp_ir.get("equipment_instances", [])):
        expected = f"equipment_instances[{index}].geometry"
        if expected not in field_paths:
            errors.append(_error("missing_provenance", "Equipment geometry lacks provenance.", expected))


def _validate_unresolved_items(flp_ir, errors):
    for item in flp_ir.get("unresolved_items", []):
        field_path = item.get("field_path")
        description = item.get("description", "")
        if field_path == "site.boundary":
            errors.append(_error("missing_site", description, field_path))
        elif "active_layout" in item.get("item_id", "") or "base_layout" in str(field_path):
            errors.append(_error("missing_active_layout_ref", description, field_path))
        elif item.get("item_type") == "ambiguous_reference":
            error = _error("ambiguous_asset", description, field_path)
            error["candidates"] = item.get("candidates", [])
            errors.append(error)
        elif item.get("item_type") == "unsupported_requirement":
            errors.append(_error("unsupported_task", description, field_path))
        elif item.get("item_type") == "missing_field" and item.get("severity") == "blocking":
            errors.append(_error("missing_engineering_field", description, field_path))


def _validation_report(errors, warnings):
    error_types = {error["error_type"] for error in errors}
    if not errors:
        status = "VALID_WITH_WARNINGS" if warnings else "VALID_SOLVER_READY"
    elif any(error_type.startswith("unsupported") or error_type == "edit_operation_in_flp_ir" for error_type in error_types):
        status = "UNSUPPORTED_TASK"
    elif {
        "missing_site",
        "ambiguous_asset",
        "missing_active_layout_ref",
        "missing_asset_reference",
        "missing_equipment_instances",
        "missing_engineering_field",
    } & error_types:
        status = "NEED_CLARIFICATION"
    elif "schema_error" in error_types:
        status = "REPAIR_REQUIRED"
    else:
        status = "INVALID_TASK"
    return {"status": status, "errors": errors, "warnings": warnings}


def _site_facts_from_evidence(message, semantic_frame=None, context=None):
    """Choose a site rectangle from explicit current text or accepted site evidence.

    Equipment clarification text often includes body footprints such as
    "7.52 m by 1.04 m". Those dimensions are not a site declaration. Prefer a
    current explicit site change, then the source-backed semantic site, then
    the persisted facility brief.
    """
    context = context or {}
    current_input = str(context.get("current_user_input") or "")
    engineering_context = str(context.get("engineering_context") or "")

    current_facts = _extract_site_facts(current_input, allow_unqualified=False)
    if current_facts:
        return current_facts

    site = (semantic_frame or {}).get("site") or {}
    if isinstance(site, dict):
        source_text = str(site.get("source_text") or "").strip()
        supported_sources = (current_input, engineering_context, str(message or ""))
        try:
            width = float(site.get("width_m"))
            depth = float(site.get("depth_m"))
        except (TypeError, ValueError):
            width = depth = 0.0
        if (
            source_text
            and any(source_text.casefold() in value.casefold() for value in supported_sources if value)
            and width > 0.0
            and depth > 0.0
        ):
            return _site_fact(width, depth, source_text)

    context_facts = _extract_site_facts(engineering_context, allow_unqualified=False)
    if context_facts:
        return context_facts

    message_facts = _extract_site_facts(message, allow_unqualified=False)
    if message_facts:
        return message_facts

    has_asset_evidence = bool((semantic_frame or {}).get("asset_mentions")) or bool(
        (context.get("engineering_rag_grounding") or {}).get("asset_matches")
    )
    fallback_text = current_input or str(message or "")
    if has_asset_evidence or _mentions_english_equipment(fallback_text):
        return []
    # Preserve support for compact site-only prompts, including the existing
    # non-English x-by-y form, when there is no equipment evidence to confuse
    # with a facility dimension.
    return _extract_site_facts(fallback_text, allow_unqualified=True)


def _extract_site_facts(message, *, allow_unqualified=True):
    directional_patterns = [
        re.compile(
            r"\b(?:the\s+)?(?:floor|site|workshop|building|facility)\b"
            r"[^.!?\n]{0,96}?"
            r"(?P<width>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)?\s+"
            r"east\s*(?:to|-)\s*west\s*(?:and|by|,)\s*"
            r"(?P<depth>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)?\s+"
            r"north\s*(?:to|-)\s*south\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:the\s+)?(?:floor|site|workshop|building|facility)\b"
            r"[^.!?\n]{0,96}?"
            r"(?P<depth>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)?\s+"
            r"north\s*(?:to|-)\s*south\s*(?:and|by|,)\s*"
            r"(?P<width>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)?\s+"
            r"east\s*(?:to|-)\s*west\b",
            re.IGNORECASE,
        ),
    ]
    for pattern in directional_patterns:
        directional_match = pattern.search(message)
        if directional_match:
            return _site_fact(
                directional_match.group("width"),
                directional_match.group("depth"),
                directional_match.group(0),
            )
    english_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*m?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*m?",
        re.IGNORECASE,
    )
    english_matches = list(english_pattern.finditer(message))
    english_match = next(
        (
            match
            for match in english_matches
            if re.search(
                r"\b(?:floor|site|workshop|building|facility)\b"
                r"(?:[^.!?\n]{0,36}(?:\b(?:is|are|measures?|measuring|has|dimensions?|size|to)\b|:))?\s*$",
                message[max(0, match.start() - 48) : match.start()],
                re.IGNORECASE,
            )
            or re.match(
                r"\s*(?:m\s*)?(?:[-,:]\s*)?\b(?:floor|site|workshop|building|facility)\b",
                message[match.end() : match.end() + 36],
                re.IGNORECASE,
            )
        ),
        None,
    )
    if english_match is None and allow_unqualified:
        english_match = english_matches[0] if english_matches else None
    if english_match:
        return _site_fact(
            english_match.group(1), english_match.group(2), english_match.group(0)
        )
    if not allow_unqualified:
        return []
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m|米)?\s*[xX×*]\s*(\d+(?:\.\d+)?)\s*(?:m|米)?")
    match = pattern.search(message)
    if not match:
        return []
    return _site_fact(match.group(1), match.group(2), match.group(0))


def _site_fact(width_value, depth_value, source_text):
    width_m = float(width_value)
    depth_m = float(depth_value)
    return [
        {
            "fact_id": "SITE_FACT_001",
            "fact_type": "rectangular_site",
            "value": {"width_m": width_m, "depth_m": depth_m},
            "source_text": source_text,
            "confidence": 0.95,
        }
    ]


def _has_logistics_distance_objective(message):
    return ("物流" in message or "距离" in message) and any(
        keyword in message for keyword in ["最小", "缩短", "尽量", "minimize"]
    )


def _mentions_equipment(message):
    return any(keyword in message for keyword in ["设备", "机组", "柜", "机", "泵", "罐", "炉"])


def _has_english_logistics_distance_objective(message):
    message_lower = message.lower()
    mentions_distance = any(
        keyword in message_lower
        for keyword in ["material handling", "logistics", "flow distance", "travel distance", "distance"]
    )
    asks_to_reduce = any(keyword in message_lower for keyword in ["minimize", "shorten", "reduce", "least"])
    return mentions_distance and asks_to_reduce


def _mentions_english_equipment(message):
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in ["equipment", "machine", "asset", "model_asset_id"])


def _candidate(asset, score, match_type):
    return {
        "model_asset_id": asset["model_asset_id"],
        "library_id": asset.get("library_id"),
        "site_id": asset.get("site_id"),
        "score": score,
        "match_type": match_type,
    }


def _unit_system():
    return {
        "length_unit": "m",
        "angle_unit": "deg",
        "mass_unit": "unknown",
        "time_unit": "unknown",
        "coordinate_system": "right_handed",
        "canonical_storage": "meter_degree",
    }


def _unresolved_item(item_id, item_type, field_path, description, severity, candidates, resolution_policy):
    return {
        "item_id": item_id,
        "item_type": item_type,
        "field_path": field_path,
        "description": description,
        "severity": severity,
        "candidates": candidates,
        "resolution_policy": resolution_policy,
    }


def _provenance(field_path, source_type, source_id, source_field, confidence):
    return {
        "field_path": field_path,
        "source_type": source_type,
        "source_id": source_id,
        "source_field": source_field,
        "confidence": confidence,
        "note": None,
    }


def _error(error_type, message, field_path):
    return {"error_type": error_type, "message": message, "field_path": field_path}


def _normalize_text(value):
    return re.sub(r"\s+", "", str(value).lower())


def _contains_cjk(value):
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _schema_registry(schemas_dir):
    resources = []
    for schema_path in Path(schemas_dir).rglob("*.json"):
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        relative_id = schema_path.relative_to(schemas_dir).as_posix()
        resource = Resource.from_contents(schema)
        resources.append((relative_id, resource))
        if "$id" in schema:
            resources.append((schema["$id"], resource))
    return Registry().with_resources(resources)
