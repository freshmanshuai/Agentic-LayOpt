"""Authoritative context resolution for layout-agent turns.

The resolver is deliberately independent from the intent LLM.  It merges all
state that is already authoritative before semantic completeness is evaluated,
so a model omission cannot turn known state into a clarification request.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter


SCHEMA_VERSION = "authoritative-context-v1"


class AuthoritativeContextResolver:
    """Merge durable session and current-turn layout context."""

    _TASK_TYPES = {
        "generate": "Generate",
        "generation": "Generate",
        "optimize": "Optimize",
        "optimization": "Optimize",
        "edit": "DirectiveEdit",
        "directiveedit": "DirectiveEdit",
        "directive_edit": "DirectiveEdit",
        "repair": "ConstraintRepair",
        "constraintrepair": "ConstraintRepair",
        "constraint_repair": "ConstraintRepair",
    }

    def resolve(self, user_input, scene_context=None):
        scene_context = scene_context or {}
        initial_layout = _first_mapping(
            scene_context.get("initial_layout"),
            scene_context.get("seed_layout"),
        )
        last_layout = _first_mapping(
            scene_context.get("last_layout"),
            initial_layout,
        )
        initial_flp_ir = _first_mapping(
            scene_context.get("initial_flp_ir"),
            scene_context.get("base_flp_ir"),
        )
        last_flp_ir = _first_mapping(
            scene_context.get("last_flp_ir"),
            initial_flp_ir,
        )
        history = _merge_history(
            scene_context.get("conversation_history"),
            scene_context.get("assistant_history"),
        )
        instance_directory = _instance_directory(
            scene_context.get("instance_directory"),
            last_layout,
            last_flp_ir,
        )
        selected_instance_ids = _unique_strings(
            scene_context.get("selected_instance_ids")
            or (scene_context.get("selection_resolution") or {}).get(
                "selected_instance_ids"
            )
            or []
        )
        task_type = self._canonical_task_type(scene_context.get("task_type"))
        site_snapshot = _first_mapping(
            scene_context.get("site_snapshot"),
            (last_flp_ir.get("site_ref") or {}).get("site_snapshot"),
            _site_snapshot_from_layout(last_layout),
        )
        resolved = {
            "schema_version": SCHEMA_VERSION,
            "current_user_input": str(user_input or ""),
            "task_type": task_type,
            "site_snapshot": copy.deepcopy(site_snapshot),
            "initial_layout": copy.deepcopy(initial_layout) or None,
            "last_layout": copy.deepcopy(last_layout) or None,
            "authoritative_layout": copy.deepcopy(last_layout) or None,
            "initial_flp_ir": copy.deepcopy(initial_flp_ir) or None,
            "last_flp_ir": copy.deepcopy(last_flp_ir) or None,
            "authoritative_flp_ir": copy.deepcopy(last_flp_ir) or None,
            "selected_instance_ids": selected_instance_ids,
            "instance_directory": instance_directory,
            "conversation_history": history,
        }
        resolved["known_fields"] = self._known_fields(resolved)
        resolved["source_map"] = {
            "site_snapshot": (
                "scene_context.site_snapshot"
                if scene_context.get("site_snapshot")
                else (
                    "authoritative_flp_ir.site_ref.site_snapshot"
                    if (last_flp_ir.get("site_ref") or {}).get("site_snapshot")
                    else "authoritative_layout.site"
                )
            ),
            "layout": (
                "scene_context.last_layout"
                if scene_context.get("last_layout")
                else "scene_context.initial_layout_or_seed_layout"
            ),
            "flp_ir": (
                "scene_context.last_flp_ir"
                if scene_context.get("last_flp_ir")
                else "scene_context.initial_or_base_flp_ir"
            ),
            "instance_directory": (
                "scene_context.instance_directory"
                if scene_context.get("instance_directory")
                else "derived_from_authoritative_layout_or_flp_ir"
            ),
            "selected_instance_ids": "scene_context.selected_instance_ids",
            "conversation_history": "conversation_history+assistant_history",
            "current_user_input": "current_user_input",
        }
        resolved["context_sha256"] = _stable_sha256(resolved)
        return resolved

    def apply(self, user_input, scene_context=None):
        """Resolve context and store it on a shallow copy of ``scene_context``."""

        result = dict(scene_context or {})
        resolved = self.resolve(user_input, result)
        result["authoritative_context"] = resolved
        if resolved["site_snapshot"]:
            result["site_snapshot"] = copy.deepcopy(resolved["site_snapshot"])
        if resolved["initial_layout"]:
            result.setdefault(
                "initial_layout", copy.deepcopy(resolved["initial_layout"])
            )
        if resolved["last_layout"]:
            result["last_layout"] = copy.deepcopy(resolved["last_layout"])
            result.setdefault("seed_layout", copy.deepcopy(resolved["last_layout"]))
        if resolved["initial_flp_ir"]:
            result.setdefault(
                "initial_flp_ir", copy.deepcopy(resolved["initial_flp_ir"])
            )
        if resolved["last_flp_ir"]:
            result["last_flp_ir"] = copy.deepcopy(resolved["last_flp_ir"])
            result.setdefault("base_flp_ir", copy.deepcopy(resolved["last_flp_ir"]))
        if resolved["instance_directory"]:
            result["instance_directory"] = copy.deepcopy(
                resolved["instance_directory"]
            )
        if resolved["selected_instance_ids"]:
            result["selected_instance_ids"] = list(
                resolved["selected_instance_ids"]
            )
        if resolved["task_type"]:
            result["task_type"] = resolved["task_type"]
        return result

    def enrich_semantic_frame(self, semantic_frame, resolved_context):
        """Overlay only facts that are already authoritative."""

        frame = copy.deepcopy(semantic_frame or {})
        context = resolved_context or {}
        task_type = context.get("task_type")
        if not task_type:
            task_type={"generate_feasible_layout":"Generate","optimize_layout_from_scratch":"Generate",
                       "optimize_existing_layout":"Optimize","edit_layout":"DirectiveEdit","constraint_repair":"ConstraintRepair"}.get(frame.get("intent_type"))
        intent_by_task = {
            "Generate": "optimize_layout_from_scratch",
            "Optimize": "optimize_existing_layout",
            "DirectiveEdit": "edit_layout",
            "ConstraintRepair": "constraint_repair",
        }
        if task_type in intent_by_task:
            if context.get("task_type"):
                frame["intent_type"] = intent_by_task[task_type]
            frame["task_contract_type"] = task_type

        boundary = (context.get("site_snapshot") or {}).get("boundary") or {}
        dimensions = boundary.get("dimensions_m") or {}
        if dimensions.get("width_m") and dimensions.get("depth_m"):
            frame["site"] = {
                "width_m": float(dimensions["width_m"]),
                "depth_m": float(dimensions["depth_m"]),
                "source_text": "authoritative_context.site_snapshot.boundary",
                "value_source": "authoritative_context",
            }

        directory = context.get("instance_directory") or []
        if directory:
            counts = Counter(
                str(item.get("model_asset_id"))
                for item in directory
                if item.get("model_asset_id")
            )
            frame["asset_mentions"] = [
                {
                    "surface_text": model_asset_id,
                    "model_asset_id": model_asset_id,
                    "quantity": quantity,
                    "value_source": "authoritative_context",
                }
                for model_asset_id, quantity in sorted(counts.items())
            ]
            frame["authoritative_instance_directory"] = copy.deepcopy(directory)

        authoritative_flp_ir = context.get("authoritative_flp_ir") or {}
        flows = authoritative_flp_ir.get("flow_relations")
        if (
            task_type in {"DirectiveEdit", "ConstraintRepair", "Optimize"}
            and authoritative_flp_ir.get("objectives")
            and not frame.get("objectives")
        ):
            frame["objectives"] = copy.deepcopy(
                authoritative_flp_ir["objectives"]
            )
            if task_type == "DirectiveEdit" and any(op.get("op_type") in {"add","remove","replace"} for op in frame.get("edit_operations") or []):
                # Relocation-only coefficients do not price new/disposed or
                # replacement equipment. Keep the inherited transport objective.
                frame["objectives"]=[o for o in frame["objectives"] if o.get("objective_type")!="minimize_relocation_effort"]
        current_request = str(context.get("current_user_input") or "").rsplit("Current user request:\n",1)[-1]
        inherit_routes = task_type in {"Optimize","ConstraintRepair"} and not re.search(r"(?:->|→)", current_request)
        if flows and (not frame.get("process_relations") or inherit_routes):
            frame["process_relations"] = [
                _semantic_relation_from_flp(item) for item in flows
            ]
            frame["process_relations_source"] = "authoritative_flp_ir"
        elif flows and task_type == "DirectiveEdit" and frame.get("process_relations"):
            current=str(context.get("current_user_input") or "").rsplit("Current user request:\n",1)[-1]
            chains=[line for line in current.splitlines() if "->" in line or "→" in line]
            if chains and all(re.search(r"\btool\s+service\b",line,re.I) for line in chains):
                # A complete service-plan update leaves reviewed product routes
                # in place; it must not erase receiving-to-dispatch logistics.
                frame["process_relations"]=[r for r in frame["process_relations"] if r.get("flow_type")=="tool_service"]+[
                    _semantic_relation_from_flp(item) for item in flows if item.get("flow_type")!="tool_service"]
                frame["process_relations_source"]="current_tool_service_plan+authoritative_material_routes"
        elif not frame.get("process_relations"):
            explicit_relations = _explicit_candidate_relations(context)
            if explicit_relations:
                frame["process_relations"] = explicit_relations
                frame["process_relations_source"] = (
                    "authoritative_context.current_user_input"
                )

        if context.get("selected_instance_ids"):
            frame["selected_instance_ids"] = list(
                context["selected_instance_ids"]
            )
        frame["authoritative_context_sha256"] = context.get("context_sha256")
        frame["missing_information"] = self._filter_known_missing_information(
            frame.get("missing_information") or [],
            context.get("known_fields") or {},
        )
        return frame

    def filter_intent_contract(
        self,
        intent_contract,
        resolved_context,
        *,
        semantic_frame=None,
    ):
        """Remove model-contract issues for fields supplied by authoritative state."""

        contract = copy.deepcopy(intent_contract or {})
        known = (resolved_context or {}).get("known_fields") or {}
        kept = []
        removed = []
        for issue in contract.get("issues") or []:
            if _issue_is_satisfied(
                issue,
                known,
                semantic_frame=semantic_frame,
            ):
                removed.append(copy.deepcopy(issue))
            else:
                kept.append(copy.deepcopy(issue))
        contract["issues"] = kept
        contract["status"] = (
            "VALID" if not kept else contract.get("status", "INVALID")
        )
        contract["authoritatively_resolved_issues"] = removed
        return contract

    @classmethod
    def _canonical_task_type(cls, value):
        text = str(value or "").strip()
        if text in cls._TASK_TYPES.values():
            return text
        legacy = {
            "Edit": "DirectiveEdit",
            "Repair": "ConstraintRepair",
        }
        if text in legacy:
            return legacy[text]
        return cls._TASK_TYPES.get(text.casefold())

    @staticmethod
    def _known_fields(context):
        layout = context.get("authoritative_layout") or {}
        flp_ir = context.get("authoritative_flp_ir") or {}
        site_snapshot = context.get("site_snapshot") or {}
        site_constraint_types = _site_hard_constraint_types(site_snapshot)
        return {
            "site": bool(site_snapshot),
            "site_dimensions": bool(
                (site_snapshot.get("boundary") or {}).get(
                    "dimensions_m"
                )
            ),
            "equipment": bool(context.get("instance_directory")),
            "equipment_quantity": bool(context.get("instance_directory")),
            "active_layout": bool(layout),
            "flp_ir": bool(flp_ir),
            "selected_instances": bool(context.get("selected_instance_ids")),
            "process_relations": bool(flp_ir.get("flow_relations")),
            "objectives": bool(flp_ir.get("objectives")),
            "hard_constraints": bool(
                (flp_ir.get("constraints") or {}).get("hard")
            ),
            "hard_constraint_types": sorted(
                {
                    str(item.get("constraint_type"))
                    for item in (
                        (flp_ir.get("constraints") or {}).get("hard") or []
                    )
                    if item.get("constraint_type")
                }
                | site_constraint_types
            ),
        }

    @staticmethod
    def _filter_known_missing_information(items, known):
        kept = []
        for item in items:
            text = str(item or "").casefold()
            if known.get("site") and any(
                token in text for token in ("site", "boundary", "dimension")
            ):
                continue
            if known.get("equipment") and any(
                token in text for token in ("equipment", "asset", "quantity")
            ):
                continue
            if known.get("active_layout") and any(
                token in text
                for token in ("active layout", "existing layout", "current layout")
            ):
                continue
            kept.append(item)
        return kept


def _first_mapping(*values):
    for value in values:
        if isinstance(value, dict) and value:
            return value
    return {}


def _site_snapshot_from_layout(layout):
    site = (layout or {}).get("site") or {}
    dimensions = site.get("dimensions_m") or {}
    width = dimensions.get("width_m") or site.get("width_m")
    depth = dimensions.get("depth_m") or site.get("depth_m")
    try:
        width = float(width)
        depth = float(depth)
    except (TypeError, ValueError):
        return {}
    if width <= 0.0 or depth <= 0.0:
        return {}
    center_x = site.get("center_x_m")
    center_z = site.get("center_z_m")
    try:
        center_x = float(center_x)
    except (TypeError, ValueError):
        center_x = (
            (float(site["min_x_m"]) + float(site["max_x_m"])) / 2.0
            if "min_x_m" in site and "max_x_m" in site
            else 0.0
        )
    try:
        center_z = float(center_z)
    except (TypeError, ValueError):
        center_z = (
            (float(site["min_z_m"]) + float(site["max_z_m"])) / 2.0
            if "min_z_m" in site and "max_z_m" in site
            else 0.0
        )
    zones = copy.deepcopy(site.get("zones") or [])
    for item in site.get("non_placeable_zones") or []:
        zone = copy.deepcopy(item)
        zone.setdefault("zone_type", "forbidden_zone")
        zones.append(zone)
    for item in site.get("placement_regions") or []:
        zone = copy.deepcopy(item)
        zone.setdefault("zone_type", "placement_region")
        zones.append(zone)
    site_id = site.get("site_id") or "SITE_FROM_LAYOUT"
    return {
        "schema_version": "layout-site-snapshot-v1",
        "site_id": site_id,
        "site_definition_mode": "authoritative_layout",
        "boundary": {
            "boundary_id": f"{site_id}/boundary",
            "dimensions_m": {"width_m": width, "depth_m": depth},
            "pose": {"x_m": center_x, "y_m": 0.0, "z_m": center_z},
        },
        "site_features": copy.deepcopy(site.get("site_features") or []),
        "zones": zones,
        "embedded_equipment_instances": copy.deepcopy(
            site.get("fixed_equipment_instances") or []
        ),
    }


def _merge_history(*histories):
    result = []
    seen = set()
    for history in histories:
        for item in history or []:
            if not isinstance(item, dict):
                continue
            key = _stable_sha256(item)
            if key in seen:
                continue
            seen.add(key)
            result.append(copy.deepcopy(item))
    return result


def _instance_directory(value, layout, flp_ir):
    if isinstance(value, list) and value:
        return copy.deepcopy(value)
    rows = []
    source = (flp_ir or {}).get("equipment_instances") or (
        layout or {}
    ).get("placements") or []
    counts = Counter()
    role_counts = Counter(str(item.get("role")) for item in source if item.get("role"))
    for item in source:
        instance_id = item.get("instance_id")
        model_asset_id = item.get("model_asset_id")
        if not instance_id or not model_asset_id:
            continue
        counts[model_asset_id] += 1
        quantity_index = int(item.get("quantity_index") or counts[model_asset_id])
        role_alias = item.get("role") if role_counts.get(str(item.get("role"))) == 1 else None
        stable_alias = item.get("stable_alias") or role_alias or (
            f"{model_asset_id}#{quantity_index}"
        )
        rows.append(
            {
                "instance_id": str(instance_id),
                "model_asset_id": str(model_asset_id),
                "quantity_index": quantity_index,
                "stable_alias": copy.deepcopy(stable_alias),
            }
        )
    return rows


def _unique_strings(values):
    result = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _semantic_relation_from_flp(item):
    return {
        "from_instance_id": item.get("from_instance_id"),
        "to_instance_id": item.get("to_instance_id"),
        "flow_type": item.get("flow_type")
        or item.get("relation_type")
        or "material",
        "flow_weight": float(item.get("flow_weight") or 1.0),
        "sequence_order": item.get("sequence_order"),
        "required_connection": bool(item.get("required_connection", False)),
        "transport_mode": item.get("transport_mode"),
        "carrier_instance_id": item.get("carrier_instance_id"),
        "carrier_source_text": item.get("carrier_source_text"),
        "evidence_source": item.get("evidence_source")
        or "authoritative_flp_ir",
    }


def _explicit_candidate_relations(context):
    """Bind all explicitly listed review-candidate links to stable instances."""

    text = str((context or {}).get("current_user_input") or "")
    if "candidate engineering facts" not in text.casefold():
        return []
    aliases = []
    for item in (context or {}).get("instance_directory") or []:
        instance_id = item.get("instance_id")
        model_asset_id = item.get("model_asset_id")
        if not instance_id or not model_asset_id:
            continue
        stable = item.get("stable_alias") or {}
        for alias in stable.values():
            alias = str(alias or "").strip()
            if alias:
                aliases.append(
                    {
                        "alias": alias,
                        "instance_id": str(instance_id),
                        "model_asset_id": str(model_asset_id),
                        "quantity_index": int(item.get("quantity_index") or 1),
                    }
                )
    matches = []
    for left in aliases:
        for right in aliases:
            if left["instance_id"] == right["instance_id"]:
                continue
            pattern = re.compile(
                re.escape(left["alias"])
                + r"\s*(?:to|->|→|到|至)\s*"
                + re.escape(right["alias"])
                + r"\s*[:：]",
                re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                matches.append((match.start(), match.group(0), left, right))
    result = []
    seen = set()
    for _start, source_text, left, right in sorted(
        matches,
        key=lambda item: (item[0], item[2]["instance_id"], item[3]["instance_id"]),
    ):
        key = (left["instance_id"], right["instance_id"])
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "from_instance_id": left["instance_id"],
                "to_instance_id": right["instance_id"],
                "from_model_asset_id": left["model_asset_id"],
                "to_model_asset_id": right["model_asset_id"],
                "from_quantity_index": left["quantity_index"],
                "to_quantity_index": right["quantity_index"],
                "flow_type": "material",
                "flow_weight": 1.0,
                "sequence_order": None,
                "required_connection": False,
                "source_text": source_text,
                "evidence_source": "current_user_input",
            }
        )
    return result


def _issue_is_satisfied(issue, known, *, semantic_frame=None):
    text = " ".join(
        str(issue.get(key) or "")
        for key in ("issue_id", "issue_type", "field_path", "message")
    ).casefold()
    if known.get("site") and any(
        token in text for token in ("site", "boundary", "dimension")
    ):
        return True
    if known.get("equipment") and any(
        token in text for token in ("equipment", "asset", "quantity")
    ):
        return True
    if known.get("active_layout") and any(
        token in text
        for token in ("active_layout", "active layout", "existing layout")
    ):
        return True
    if known.get("flp_ir") and str(issue.get("field_path") or "").startswith(
        "objectives["
    ):
        return True
    if (
        "edit_operations" in text
        and (semantic_frame or {}).get("edit_operations")
    ):
        return True
    constraint_match = re.match(
        r"constraints\[(\d+)\]",
        str(issue.get("field_path") or ""),
    )
    if constraint_match and known.get("hard_constraint_types"):
        constraints = (semantic_frame or {}).get("constraints") or []
        index = int(constraint_match.group(1))
        if index < len(constraints) and str(
            constraints[index].get("constraint_type") or ""
        ) in set(known["hard_constraint_types"]):
            return True
    return False


def _site_hard_constraint_types(site_snapshot):
    """Infer only constraints that are explicit in the authoritative snapshot."""

    if not isinstance(site_snapshot, dict) or not site_snapshot:
        return set()
    result = {"inside_boundary"} if site_snapshot.get("boundary") else set()
    feature_types = {
        str(item.get("feature_type") or "").casefold()
        for item in site_snapshot.get("site_features") or []
        if isinstance(item, dict) and item.get("must_remain_clear") is not False
    }
    zone_types = {
        str(item.get("zone_type") or "").casefold()
        for item in site_snapshot.get("zones") or []
        if isinstance(item, dict)
    }
    if feature_types & {"wall", "column", "pillar", "obstacle"}:
        result.add("structural_obstacles_clear")
    if feature_types & {"corridor", "aisle", "passage"}:
        result.add("corridor_preservation")
    if feature_types & {"entrance", "exit", "door", "egress"}:
        result.add("entrance_exit_unblocked")
    if feature_types & {"forbidden_zone", "no_placement", "restricted_area"} or zone_types & {
        "forbidden_zone",
        "no_placement",
        "restricted_area",
    }:
        result.add("forbidden_zone_avoidance")
    if site_snapshot.get("embedded_equipment_instances"):
        result.add("fixed_equipment_clear")
    return result


def _stable_sha256(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
