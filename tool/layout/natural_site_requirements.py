"""Compile a narrow set of explicit English site requirements into site geometry.

This module deliberately handles only rectangular sites with explicit wall
margins, centered orthogonal aisles, rectangular fixed buffer areas, and
cardinal placement groups. Door alignment determines the aisle axis where
explicitly stated. Near-door floor buffers have a recorded initial planning
placement; this placement is distinguished from a user-specified center.
"""

from __future__ import annotations

import copy
import math
import re

from tool.logistics.facility_logistics import derive_aligned_doors


_PREFIX = "natural_site/"
_CARDINAL_DIRECTIONS = {"north", "south", "east", "west"}
_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_LENGTH = r"(?P<value>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)"
_EDGE_OFFSET = re.compile(
    _LENGTH + r"\s+from\s+(?:the\s+)?(?P<edge>west|east|south|north)\s+edge\b",
    re.IGNORECASE,
)
_BUFFER_SIZE = re.compile(
    r"(?P<width>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)\s*"
    r"(?:by|x|×)\s*"
    r"(?P<depth>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)\s+"
    r"(?:fixed\s+)?(?:(?:inbound|outbound)\s+)?buffers?(?:\s+areas?)?\b",
    re.IGNORECASE,
)
_BUFFER_HEADING = re.compile(
    r"\b(?:center|centre)\s+(?:the\s+)?(?P<name>[a-z][a-z0-9\s-]*?)\s+buffer\b",
    re.IGNORECASE,
)
_DIRECTION_HEADING = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?P<direction>south|north|west|east)\s*:\s*"
    r"(?P<content>[^\n]+)",
)


def compile_natural_site_requirements(
    user_message,
    site_snapshot,
    *,
    semantic_intent=None,
    equipment_instances=None,
    organization_text=None,
    intent_id="INTENT_001",
):
    """Return an enriched snapshot and source-backed instance-region assignments.

    The input snapshot supplies the authoritative rectangular site dimensions.
    The raw request supplies only explicit features that can be located without
    a case-specific template. ``unresolved_items`` records any feature wording
    that this narrow compiler recognizes but cannot locate unambiguously.
    """
    text = str(user_message or "")
    snapshot = copy.deepcopy(site_snapshot or {})
    result = {
        "site_snapshot": snapshot,
        "instance_region_assignments": {},
        "unresolved_items": [],
        "provenance": [],
        "constraint_requirements": [],
        "literal_spans": natural_site_literal_spans(text),
    }
    bounds = _site_bounds(snapshot)
    if bounds is None:
        if _mentions_site_feature(text):
            _add_unresolved(
                result,
                "missing_rectangular_site",
                "site_ref.site_snapshot.boundary",
                "A rectangular site boundary is required before spatial features can be located.",
                _first_spatial_sentence(text),
            )
        return result

    _ensure_snapshot_fields(snapshot, text, intent_id)
    has_existing_wall_margin = _has_natural_wall_margin(snapshot)
    has_existing_aisle = _has_natural_central_aisle(snapshot) or _has_existing_corridor(snapshot)
    has_existing_buffers = _has_natural_buffers(snapshot)
    margin, margin_literal = _wall_margin(text)
    if _mentions_wall_margin(text) and margin is None:
        if has_existing_wall_margin and _feature_change_requested(text, "wall"):
            _add_unresolved(
                result,
                "wall_margin_update",
                "site_ref.site_snapshot.site_features",
                "Changing an inherited wall margin requires an explicit site-update operation.",
                _first_sentence_matching(text, r"\b(?:wall|outer|perimeter)\b"),
            )
        elif not has_existing_wall_margin:
            _add_unresolved(
                result,
                "wall_margin",
                "site_ref.site_snapshot.site_features",
                "The wall-clearance requirement needs one explicit margin value.",
                _first_sentence_matching(text, r"\b(?:wall|outer|perimeter)\b"),
            )
    if margin is not None:
        existing_margin = _existing_wall_margin(snapshot, bounds)
        if has_existing_wall_margin and not math.isclose(
            margin, existing_margin, abs_tol=1e-9
        ):
            _add_unresolved(
                result,
                "wall_margin_update",
                "site_ref.site_snapshot.site_features",
                "Changing an inherited wall margin requires an explicit site-update operation.",
                margin_literal,
            )
            margin = existing_margin
        elif not _valid_margin(margin, bounds):
            _add_unresolved(
                result,
                "wall_margin_out_of_range",
                "site_ref.site_snapshot.site_features",
                "The wall margin leaves no usable rectangular placement area.",
                margin_literal,
            )
            margin = None
        else:
            _apply_wall_margin(snapshot, bounds, margin, margin_literal, result, intent_id)
    if margin is None:
        margin = _existing_wall_margin(snapshot, bounds)

    aisle = _central_aisle(text)
    if _mentions_aisle(text) and aisle is None:
        if has_existing_aisle and _feature_change_requested(text, "aisle"):
            _add_unresolved(
                result,
                "central_aisle_update",
                "site_ref.site_snapshot.site_features",
                "Changing an inherited central aisle requires the dedicated aisle-update operation.",
                _first_sentence_matching(text, r"\b(?:aisle|corridor)\b"),
            )
        elif not has_existing_aisle:
            _add_unresolved(
                result,
                "central_aisle",
                "site_ref.site_snapshot.site_features",
                "A centered aisle requires an explicit width and east-west or north-south direction.",
                _first_sentence_matching(text, r"\b(?:aisle|corridor)\b"),
            )
    if aisle is not None and has_existing_aisle and _feature_change_requested(text, "aisle"):
        _add_unresolved(
            result,
            "central_aisle_update",
            "site_ref.site_snapshot.site_features",
            "Changing an inherited central aisle requires the dedicated aisle-update operation.",
            aisle["literal"],
        )
        aisle = None
    if aisle is not None:
        if not _aisle_fits(aisle, bounds, margin):
            _add_unresolved(
                result,
                "central_aisle_out_of_range",
                "site_ref.site_snapshot.site_features",
                "The stated aisle width does not fit inside the usable site boundary.",
                aisle["literal"],
            )
            aisle = None
        else:
            _apply_central_aisle(snapshot, bounds, margin, aisle, result, intent_id)

    derive_aligned_doors(snapshot, text)
    buffers = _buffer_requirements(
        text,
        bounds,
        margin,
        result,
        has_existing_buffers=has_existing_buffers,
    )
    for buffer in buffers:
        _upsert_feature(snapshot, _buffer_feature(buffer, _frame_id(snapshot)))
        _record_provenance(
            result,
            snapshot,
            f"site_features[{buffer['feature_id']}]",
            buffer["literal"],
            intent_id,
            buffer.get("planning_note") or "Rectangular buffer center was computed from the stated edge offsets.",
        )
    if buffers:
        _add_constraint_requirement(
            result,
            "forbidden_zone_avoidance",
            "; ".join(buffer["literal"] for buffer in buffers),
        )

    region_statement = _group_region_statement(text)
    headings = _group_headings(text)
    group_literal = (
        result["literal_spans"].get("group_regions")
        or (region_statement or {}).get("literal")
        or _first_direction_heading_literal_from_items(headings)
    )
    if (region_statement or headings) and aisle is None:
        _add_unresolved(
            result,
            "group_regions_without_central_aisle",
            "site_ref.site_snapshot.zones",
            "Equipment groups relative to an aisle require a resolved centered aisle.",
            (region_statement or {}).get("literal") or _first_direction_heading_literal(text),
        )
    elif aisle is not None:
        region_ids = _apply_group_regions(
            snapshot,
            bounds,
            margin,
            aisle,
            region_statement,
            headings,
            group_literal,
            result,
            intent_id,
        )
        if region_ids:
            _add_constraint_requirement(
                result,
                "region_membership",
                group_literal,
            )
            _assign_instances_to_regions(
                text,
                semantic_intent or {},
                equipment_instances or [],
                headings,
                region_statement,
                region_ids,
                result,
            )
    elif _mentions_group_direction(text):
        _add_unresolved(
            result,
            "group_region_statement",
            "site_ref.site_snapshot.zones",
            "The equipment-group placement statement could not be mapped to a central aisle.",
            _first_sentence_matching(text, r"\b(?:south|north|west|east)\s+of\s+(?:the\s+)?(?:aisle|corridor)\b"),
        )

    if aisle is not None:
        _add_constraint_requirement(
            result,
            "corridor_preservation",
            aisle["literal"],
        )
    if margin is not None:
        _add_constraint_requirement(
            result,
            "forbidden_zone_avoidance",
            margin_literal,
        )
    from tool.layout.organized_workstations import compile_organized_workstations

    organization = compile_organized_workstations(
        text if organization_text is None else organization_text,
        snapshot,
        semantic_intent=semantic_intent,
        equipment_instances=equipment_instances,
        intent_id=intent_id,
    )
    result["site_snapshot"] = organization.get("site_snapshot") or snapshot
    result["instance_region_assignments"].update(
        organization.get("instance_region_assignments") or {}
    )
    result["workstation_bays"] = copy.deepcopy(
        organization.get("workstation_bays") or []
    )
    for item in organization.get("unresolved_items") or []:
        if not any(
            existing.get("item_id") == item.get("item_id")
            for existing in result["unresolved_items"]
        ):
            result["unresolved_items"].append(copy.deepcopy(item))
    result["provenance"].extend(
        copy.deepcopy(organization.get("provenance") or [])
    )
    for requirement in organization.get("constraint_requirements") or []:
        _add_constraint_requirement(
            result,
            requirement.get("constraint_type"),
            requirement.get("source_text"),
        )
    result["literal_spans"]["organization"] = (
        organization.get("literal_spans") or {}
    ).get("organization", "")
    result["literal_spans"]["maintenance"] = copy.deepcopy(
        (organization.get("literal_spans") or {}).get("maintenance") or []
    )
    return result


def apply_central_aisle_width_update(
    user_message,
    site_snapshot,
    *,
    organization_text=None,
    semantic_intent=None,
    equipment_instances=None,
    intent_id="INTENT_001",
):
    """Update one inherited central aisle while retaining all other site geometry.

    The function accepts no layout or affected-instance list. Existing layout
    poses are therefore checked by the normal repair path against the updated
    canonical site rather than being pre-classified here.
    """
    text = str(user_message or "")
    snapshot = copy.deepcopy(site_snapshot or {})
    result = {
        "site_snapshot": snapshot,
        "unresolved_items": [],
        "provenance": [],
        "changed_zone_ids": [],
    }
    if not _mentions_aisle(text):
        return result
    update = _aisle_width_update(text)
    # A preserved centerline or direction does not preserve the aisle width.
    # Only leave the inherited corridor untouched when no new width is stated.
    if update is None and _preserves_existing_aisle(text):
        return result
    if update is None:
        _add_unresolved(
            result,
            "central_aisle_width_update",
            "site_ref.site_snapshot.site_features",
            "The aisle update needs one explicit target width.",
            _first_sentence_matching(text, r"\b(?:aisle|corridor)\b"),
        )
        return result
    features = snapshot.get("site_features") or []
    corridors = [
        item
        for item in features
        if isinstance(item, dict)
        and item.get("site_feature_id", "").startswith(_PREFIX + "central_")
        and item.get("site_feature_id", "").endswith("_aisle")
        and str(item.get("feature_type") or "").casefold() == "corridor"
    ]
    if len(corridors) != 1:
        _add_unresolved(
            result,
            "central_aisle_identity",
            "site_ref.site_snapshot.site_features",
            "The aisle update requires exactly one inherited central corridor.",
            update["literal"],
        )
        return result
    corridor = corridors[0]
    axis = _axis_from_feature(corridor)
    if axis is None or (update["axis"] and update["axis"] != axis):
        _add_unresolved(
            result,
            "central_aisle_direction",
            "site_ref.site_snapshot.site_features",
            "The requested aisle direction does not match the inherited central corridor.",
            update["literal"],
        )
        return result
    bounds = _site_bounds(snapshot)
    if bounds is None:
        _add_unresolved(
            result,
            "missing_rectangular_site",
            "site_ref.site_snapshot.boundary",
            "A rectangular site boundary is required before an aisle can be widened.",
            update["literal"],
        )
        return result
    margin = _existing_wall_margin(snapshot, bounds)
    if not _aisle_fits({"axis": axis, "width_m": update["width_m"]}, bounds, margin):
        _add_unresolved(
            result,
            "central_aisle_out_of_range",
            "site_ref.site_snapshot.site_features",
            "The requested aisle width does not fit inside the inherited usable site boundary.",
            update["literal"],
        )
        return result

    changed_snapshot = copy.deepcopy(snapshot)
    changed_corridor = next(
        item
        for item in changed_snapshot.get("site_features") or []
        if item.get("site_feature_id") == corridor.get("site_feature_id")
    )
    dimensions = changed_corridor.get("dimensions_m") or {}
    if axis == "east_west":
        dimensions["depth_m"] = update["width_m"]
    else:
        dimensions["width_m"] = update["width_m"]
    changed_corridor["dimensions_m"] = dimensions

    changed_features = []
    door_target = re.search(r"\b(?:door|gate)\s+openings?\s+to\s+(\d+(?:\.\d+)?)\s*m\b", text, re.I)
    if door_target:
        for feature in changed_snapshot.get("site_features") or []:
            if feature.get("feature_type") in {"entrance", "exit"}:
                key = "depth_m" if axis == "east_west" else "width_m"
                feature["dimensions_m"][key] = float(door_target.group(1))
                changed_features.append(feature["site_feature_id"])
    allow_buffer_shift = re.search(r"\b(?:shift|move|relocate)\s+(?:(?:a|the|any|only|both|either|affected|obstructing)\s+){0,3}buffers?\b", text, re.I)
    if allow_buffer_shift:
        coordinate = "z_m" if axis == "east_west" else "x_m"
        extent = "depth_m" if axis == "east_west" else "width_m"
        center = changed_corridor["pose"][coordinate]
        half = changed_corridor["dimensions_m"][extent] / 2
        for feature in changed_snapshot.get("site_features") or []:
            if not str(feature.get("site_feature_id", "")).startswith(_PREFIX+"buffer_"):
                continue
            value = feature["pose"][coordinate];radius = feature["dimensions_m"][extent]/2
            if value-radius < center+half-1e-9 and value+radius > center-half+1e-9:
                feature["pose"][coordinate] = center + (half+radius) * (1 if value>=center else -1)
                changed_features.append(feature["site_feature_id"])
                _record_provenance(result,changed_snapshot,f"site_features[{feature['site_feature_id']}].pose",text,intent_id,"Moved only the obstructing floor buffer by the minimum perpendicular translation authorized in the request.")
    result["changed_site_feature_ids"] = list(
        dict.fromkeys([*changed_features, changed_corridor["site_feature_id"]])
    )

    if not _resize_adjacent_regions(changed_snapshot, changed_corridor, axis):
        _add_unresolved(
            result,
            "central_aisle_regions",
            "site_ref.site_snapshot.zones",
            "The inherited central aisle does not have both adjacent placement regions to resize.",
            update["literal"],
        )
        return result
    if axis == "east_west":
        if organization_text and equipment_instances is not None:
            from tool.layout.organized_workstations import compile_organized_workstations

            before_zones = {
                item.get("zone_id"): copy.deepcopy(item)
                for item in changed_snapshot.get("zones") or []
                if isinstance(item, dict)
            }
            organized = compile_organized_workstations(
                organization_text,
                changed_snapshot,
                semantic_intent=semantic_intent,
                equipment_instances=equipment_instances,
                intent_id=intent_id,
            )
            if organized.get("unresolved_items"):
                result["unresolved_items"].extend(organized["unresolved_items"])
                return result
            changed_snapshot = organized["site_snapshot"]
            result["workstation_bays"] = organized.get("workstation_bays") or []
            result["instance_region_assignments"] = organized.get("instance_region_assignments") or {}
            result["provenance"].extend(organized.get("provenance") or [])
            after_zones = {
                item.get("zone_id"): item
                for item in changed_snapshot.get("zones") or []
                if isinstance(item, dict)
            }
            result["changed_zone_ids"].extend(
                zone_id
                for zone_id, zone in after_zones.items()
                if zone_id in before_zones
                and any(
                    zone.get(key) != before_zones[zone_id].get(key)
                    for key in ("pose", "dimensions_m")
                )
                and str(zone_id).startswith(_PREFIX + "workstation_bay_")
            )
        elif not _resize_workstation_bays(
            changed_snapshot,
            result,
            update["literal"],
            intent_id,
        ):
            _add_unresolved(
                result,
                "central_aisle_workstation_bays",
                "site_ref.site_snapshot.zones",
                "The widened aisle leaves insufficient south-side depth for the existing organized workstation rows without the accepted equipment requirements.",
                update["literal"],
            )
            return result
        if not _resize_functional_north_regions(
            changed_snapshot,
            changed_corridor,
            result,
            update["literal"],
            intent_id,
        ):
            _add_unresolved(
                result,
                "central_aisle_functional_regions",
                "site_ref.site_snapshot.zones",
                "The widened aisle leaves insufficient north-side depth for the existing functional regions.",
                update["literal"],
            )
            return result
    result["site_snapshot"] = changed_snapshot
    _record_provenance(
        result,
        changed_snapshot,
        f"site_features[{changed_corridor['site_feature_id']}].dimensions_m",
        update["literal"],
        intent_id,
        "Updated the inherited central aisle width while retaining its centerline.",
    )
    return result


def natural_site_literal_spans(user_message):
    """Return exact raw spans supporting the narrow natural-site grammar."""
    text = str(user_message or "")
    headings = _group_headings(text)
    _margin, margin_literal = _wall_margin(text)
    aisle = _central_aisle(text) or {}
    group_statement = _group_region_statement(text) or {}
    return {
        "wall_margin": _sentence_containing_literal(text, margin_literal),
        "central_aisle": str(aisle.get("literal") or ""),
        "buffer_requirements": " ".join(_buffer_requirement_sentences(text)),
        "group_assignment": str(group_statement.get("literal") or ""),
        "group_regions": _combined_group_heading_literal(text, headings),
        "unobstructed_areas": _unobstructed_areas_literal(text),
    }


def _unobstructed_areas_literal(text):
    """Find an explicit aisle-and-buffer clearance statement verbatim."""

    for sentence in _sentences(text):
        folded = sentence.casefold()
        if (
            "unobstructed" in folded
            and re.search(r"\b(?:aisle|corridor)\b", folded)
            and re.search(r"\bbuffers?\b", folded)
        ):
            return sentence
    return ""


def _is_unobstructed_area_literal(value):
    folded = str(value or "").casefold()
    return bool(
        "unobstructed" in folded
        and re.search(r"\b(?:aisle|corridor)\b", folded)
        and re.search(r"\bbuffers?\b", folded)
    )


def canonical_natural_site_constraint_type(
    user_message,
    source_text,
    *,
    source_messages=None,
):
    """Map an exact recognized site primitive to its canonical FLP type.

    This deliberately uses the literal source rather than the LLM's free-form
    constraint label.  A label remains unchanged when its cited text is not
    one of the narrow natural-site primitives understood by this module.
    """

    source_key = _natural_literal_key(source_text)
    if not source_key:
        return None
    for text in [str(user_message or ""), *(source_messages or [])]:
        text = str(text or "")
        if source_key not in _natural_literal_key(text):
            continue
        spans = natural_site_literal_spans(text)

        def matches(value):
            return source_key == _natural_literal_key(value)

        if matches(spans.get("wall_margin")):
            return "forbidden_zone_avoidance"
        if matches(spans.get("central_aisle")):
            return "corridor_preservation"
        if matches(spans.get("group_assignment")) or matches(spans.get("group_regions")):
            return "region_membership"
        if matches(spans.get("unobstructed_areas")) or _is_unobstructed_area_literal(
            source_text
        ):
            return "forbidden_zone_avoidance"

        buffer_literals = _buffer_requirement_sentences(text)
        if matches(spans.get("buffer_requirements")) or any(
            matches(literal) for literal in buffer_literals
        ):
            return "forbidden_zone_avoidance"
    return None


def _buffer_requirement_sentences(text):
    return [
        sentence
        for sentence in _sentences(text)
        if _BUFFER_SIZE.search(sentence)
        or (
            re.search(r"\b(?:center|centre)\b", sentence, re.IGNORECASE)
            and re.search(r"\bbuffer\b", sentence, re.IGNORECASE)
        )
    ]


def _sentence_containing_literal(text, literal):
    literal_key = _natural_literal_key(literal)
    if not literal_key:
        return ""
    return next(
        (
            sentence
            for sentence in _sentences(text)
            if literal_key in _natural_literal_key(sentence)
        ),
        "",
    )


def _natural_literal_key(value):
    return re.sub(
        r"[\s\.,;:!?]+",
        " ",
        str(value or "").casefold(),
    ).strip()


def _site_bounds(snapshot):
    boundary = (snapshot or {}).get("boundary") or {}
    dimensions = boundary.get("dimensions_m") or {}
    pose = boundary.get("pose") or {}
    try:
        width = float(dimensions.get("width_m"))
        depth = float(dimensions.get("depth_m"))
        center_x = float(pose.get("x_m") or 0.0)
        center_z = float(pose.get("z_m") or 0.0)
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or depth <= 0.0:
        return None
    return {
        "width_m": width,
        "depth_m": depth,
        "center_x_m": center_x,
        "center_z_m": center_z,
        "min_x_m": center_x - width / 2.0,
        "max_x_m": center_x + width / 2.0,
        "min_z_m": center_z - depth / 2.0,
        "max_z_m": center_z + depth / 2.0,
    }


def _ensure_snapshot_fields(snapshot, text, intent_id):
    snapshot.setdefault("schema_version", "inline-site-v1")
    snapshot.setdefault("site_id", "SITE_INLINE_001")
    snapshot.setdefault("site_definition_mode", "user_defined")
    snapshot.setdefault("site_features", [])
    snapshot.setdefault("zones", [])
    snapshot.setdefault("embedded_equipment_instances", [])
    if snapshot.get("schema_version") != "inline-site-v1":
        snapshot.setdefault("provenance", [])
        snapshot.setdefault(
            "user_defined_source",
            {
                "source_type": "user_input",
                "source_text": _first_site_sentence(text),
                "source_file_id": None,
                "created_from_intent_id": intent_id,
            },
        )


def _wall_margin(text):
    patterns = [
        re.compile(
            r"\b(?:leave|keep|reserve)\s+(?:the\s+)?(?:outer|perimeter)\s+"
            + _LENGTH
            + r"\s+(?:along|from|inside)\s+(?:the\s+)?walls?\b",
            re.IGNORECASE,
        ),
        re.compile(
            _LENGTH + r"\s+(?:wall\s+margin|wall\s+clearance)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:keep|leave)\s+" + _LENGTH + r"\s+(?:clear|free)\s+from\s+(?:the\s+)?walls?\b",
            re.IGNORECASE,
        ),
    ]
    matches = [match for pattern in patterns for match in pattern.finditer(text)]
    if not matches:
        return None, ""
    values = {round(float(match.group("value")), 9) for match in matches}
    if len(values) != 1:
        return None, matches[0].group(0).strip()
    return values.pop(), matches[0].group(0).strip()


def _central_aisle(text):
    candidates = []
    for sentence in _sentences(text):
        if not re.search(r"\b(?:aisle|corridor)\b", sentence, re.IGNORECASE):
            continue
        width = re.search(
            r"(?P<value>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)\s*(?:-\s*)?wide\b",
            sentence,
            re.IGNORECASE,
        )
        axis = _axis_from_text(sentence)
        if axis is None and re.search(r"\balign\w*\b[^.]*\bdoors?\b[^.]*\baisle\b", text, re.I):
            if re.search(r"\breceiving\s+door\b[^.]*\bwest\s+wall\b", text, re.I) and re.search(r"\bdispatch\s+door\b[^.]*\beast\s+wall\b", text, re.I):
                axis = "east_west"
        central = bool(
            re.search(r"\bcentral\b|\bthrough\s+the\s+cent(?:er|re)\b", sentence, re.IGNORECASE)
        )
        if width and axis and central:
            candidates.append(
                {
                    "axis": axis,
                    "width_m": float(width.group("value")),
                    "literal": sentence.strip(),
                }
            )
    if not candidates:
        return None
    first = candidates[0]
    if any(
        item["axis"] != first["axis"]
        or not math.isclose(item["width_m"], first["width_m"], abs_tol=1e-9)
        for item in candidates[1:]
    ):
        return None
    return first


def _buffer_requirements(text, bounds, margin, result, *, has_existing_buffers):
    if not re.search(r"\bbuffers?\b", text, re.IGNORECASE):
        return []
    dimensions = [
        (float(match.group("width")), float(match.group("depth")), match.group(0).strip())
        for match in _BUFFER_SIZE.finditer(text)
    ]
    buffer_locations_requested = bool(
        re.search(
            r"\b(?:center|centre|reserve|add)\s+[^.!?\n:;]{0,80}\bbuffers?\b",
            text,
            re.IGNORECASE,
        )
    )
    if has_existing_buffers and _feature_change_requested(text, "buffer"):
        _add_unresolved(
            result,
            "buffer_update",
            "site_ref.site_snapshot.site_features",
            "Changing an inherited buffer requires an explicit site-update operation.",
            _first_sentence_matching(text, r"\bbuffers?\b"),
        )
        return []
    if not dimensions and has_existing_buffers and not buffer_locations_requested:
        return []
    if not dimensions:
        _add_unresolved(
            result,
            "buffer_dimensions",
            "site_ref.site_snapshot.site_features",
            "Each fixed buffer area needs explicit rectangular dimensions.",
            _first_sentence_matching(text, r"\bbuffers?\b"),
        )
        return []
    unique_dimensions = {(round(width, 9), round(depth, 9)) for width, depth, _ in dimensions}
    if len(unique_dimensions) != 1:
        _add_unresolved(
            result,
            "buffer_dimensions_ambiguous",
            "site_ref.site_snapshot.site_features",
            "The buffer declaration contains more than one incompatible rectangle size.",
            dimensions[0][2],
        )
        return []
    width, depth, dimension_literal = dimensions[0]
    buffers = []
    seen_names = set()
    for sentence in _sentences(text):
        if not _BUFFER_HEADING.match(sentence.strip().lstrip("-* ")):
            continue
        name_match = _BUFFER_HEADING.search(sentence)
        offsets = [
            (float(match.group("value")), match.group("edge").casefold())
            for match in _EDGE_OFFSET.finditer(sentence)
        ]
        if not name_match or len(offsets) < 2:
            _add_unresolved(
                result,
                "buffer_location",
                "site_ref.site_snapshot.site_features",
                "Each named buffer needs one east-west and one north-south center offset.",
                sentence.strip(),
            )
            continue
        offsets_by_axis = {}
        for value, edge in offsets:
            axis = "x" if edge in {"west", "east"} else "z"
            if axis in offsets_by_axis:
                _add_unresolved(
                    result,
                    "buffer_location_ambiguous",
                    "site_ref.site_snapshot.site_features",
                    "A named buffer has more than one offset on the same axis.",
                    sentence.strip(),
                )
                offsets_by_axis = {}
                break
            offsets_by_axis[axis] = (value, edge)
        if set(offsets_by_axis) != {"x", "z"}:
            continue
        name = _slug(name_match.group("name"))
        if not name or name in seen_names:
            _add_unresolved(
                result,
                "buffer_identity",
                "site_ref.site_snapshot.site_features",
                "Each buffer requires a distinct name.",
                sentence.strip(),
            )
            continue
        seen_names.add(name)
        center_x = _coordinate_from_edge(bounds, *offsets_by_axis["x"])
        center_z = _coordinate_from_edge(bounds, *offsets_by_axis["z"])
        if not _rectangle_inside(center_x, center_z, width, depth, bounds, margin):
            _add_unresolved(
                result,
                "buffer_out_of_bounds",
                "site_ref.site_snapshot.site_features",
                "The stated buffer rectangle overlaps the wall margin or lies outside the site.",
                sentence.strip(),
            )
            continue
        buffers.append(
            {
                "feature_id": _PREFIX + "buffer_" + name,
                "name": name_match.group("name").strip(),
                "width_m": width,
                "depth_m": depth,
                "center_x_m": center_x,
                "center_z_m": center_z,
                "literal": dimension_literal + " | " + sentence.strip(),
            }
        )
    if not buffers and re.search(r"\binbound buffer\s+near\b", text, re.I) and re.search(r"\boutbound buffer\s+near\b", text, re.I):
        corridors = _central_aisle(text)
        if corridors and corridors.get("axis") == "east_west":
            half_aisle = corridors["width_m"] / 2
            for name, x, z in [("inbound", bounds["min_x_m"]+width/2, -half_aisle-depth/2), ("outbound", bounds["max_x_m"]-width/2, half_aisle+depth/2)]:
                buffers.append(dict(feature_id=_PREFIX+"buffer_"+name,name=name,width_m=width,depth_m=depth,center_x_m=x,center_z_m=z,
                    literal=_first_sentence_matching(text, r"\binbound buffer\b")+" "+_first_sentence_matching(text, r"\boutbound buffer\b"),
                    planning_note="Initial design placement adjacent to the named door and outside the aisle; not a measured or user-specified center."))
    declared = _declared_buffer_count(text)
    if declared is not None and len(buffers) != declared:
        _add_unresolved(
            result,
            "buffer_count",
            "site_ref.site_snapshot.site_features",
            "The declared number of fixed buffers does not match the number with complete locations.",
            _first_sentence_matching(text, r"\bbuffers?\b"),
        )
    return buffers


def _group_region_statement(text):
    pattern = re.compile(
        r"\b(?:put|place|keep|locate)\s+(?P<first>.+?)\s+"
        r"(?P<first_direction>south|north|west|east)\s+of\s+(?:the\s+)?"
        r"(?:central\s+)?(?:aisle|corridor)\s+(?:and|,\s*while)\s+"
        r"(?P<second>.+?)\s+(?P<second_direction>south|north|west|east)\s+"
        r"of\s+(?:it|the\s+(?:central\s+)?(?:aisle|corridor))\b",
        re.IGNORECASE,
    )
    for sentence in _sentences(text):
        match = pattern.search(sentence)
        if match:
            return {
                "literal": sentence.strip(),
                "groups": [
                    {
                        "direction": match.group("first_direction").casefold(),
                        "description": match.group("first").strip(),
                    },
                    {
                        "direction": match.group("second_direction").casefold(),
                        "description": match.group("second").strip(),
                    },
                ],
            }
    return None


def _group_headings(text):
    return [
        {
            "direction": match.group("direction").casefold(),
            "content": match.group("content").strip(),
            "literal": match.group(0),
            "start": match.start(),
            "end": match.end(),
        }
        for match in _DIRECTION_HEADING.finditer(text)
    ]


def _combined_group_heading_literal(text, headings):
    if not headings:
        return ""
    if len(headings) == 1:
        return headings[0]["literal"]
    return str(text)[headings[0]["start"] : headings[-1]["end"]]


def _apply_wall_margin(snapshot, bounds, margin, literal, result, intent_id):
    frame = _frame_id(snapshot)
    width = bounds["width_m"]
    depth = bounds["depth_m"]
    values = [
        ("west", bounds["min_x_m"] + margin / 2.0, bounds["center_z_m"], margin, depth),
        ("east", bounds["max_x_m"] - margin / 2.0, bounds["center_z_m"], margin, depth),
        ("south", bounds["center_x_m"], bounds["min_z_m"] + margin / 2.0, width, margin),
        ("north", bounds["center_x_m"], bounds["max_z_m"] - margin / 2.0, width, margin),
    ]
    for direction, x_m, z_m, feature_width, feature_depth in values:
        feature_id = _PREFIX + "wall_margin_" + direction
        _upsert_feature(
            snapshot,
            _rect_feature(
                feature_id,
                "wall",
                direction + " wall margin",
                x_m,
                z_m,
                feature_width,
                feature_depth,
                frame,
            ),
        )
        _record_provenance(
            result,
            snapshot,
            f"site_features[{feature_id}]",
            literal,
            intent_id,
            "Wall-margin rectangle derived from the stated clearance and site boundary.",
        )


def _apply_central_aisle(snapshot, bounds, margin, aisle, result, intent_id):
    frame = _frame_id(snapshot)
    if aisle["axis"] == "east_west":
        width = bounds["width_m"] - 2.0 * margin
        depth = aisle["width_m"]
        feature_id = _PREFIX + "central_east_west_aisle"
        name = "central east-west aisle"
    else:
        width = aisle["width_m"]
        depth = bounds["depth_m"] - 2.0 * margin
        feature_id = _PREFIX + "central_north_south_aisle"
        name = "central north-south aisle"
    _upsert_feature(
        snapshot,
        _rect_feature(
            feature_id,
            "corridor",
            name,
            bounds["center_x_m"],
            bounds["center_z_m"],
            width,
            depth,
            frame,
        ),
    )
    _record_provenance(
        result,
        snapshot,
        f"site_features[{feature_id}]",
        aisle["literal"],
        intent_id,
        "Centered aisle rectangle derived from its stated axis and width.",
    )


def _apply_group_regions(
    snapshot,
    bounds,
    margin,
    aisle,
    statement,
    headings,
    group_literal,
    result,
    intent_id,
):
    directions = set()
    if statement:
        directions.update(item["direction"] for item in statement["groups"])
    directions.update(item["direction"] for item in headings)
    if not directions:
        return {}
    allowed = {"south", "north"} if aisle["axis"] == "east_west" else {"west", "east"}
    if not directions.issubset(allowed):
        _add_unresolved(
            result,
            "group_region_axis",
            "site_ref.site_snapshot.zones",
            "The stated equipment-group directions are incompatible with the central aisle axis.",
            (statement or {}).get("literal") or _first_direction_heading_literal_from_items(headings),
        )
        return {}
    corridor = _central_feature(snapshot, aisle["axis"])
    if corridor is None:
        return {}
    corridor_bounds = _feature_bounds(corridor)
    if corridor_bounds is None:
        return {}
    outer = {
        "min_x_m": bounds["min_x_m"] + margin,
        "max_x_m": bounds["max_x_m"] - margin,
        "min_z_m": bounds["min_z_m"] + margin,
        "max_z_m": bounds["max_z_m"] - margin,
    }
    rectangles = {}
    if aisle["axis"] == "east_west":
        rectangles["south"] = (outer["min_x_m"], outer["max_x_m"], outer["min_z_m"], corridor_bounds["min_z_m"])
        rectangles["north"] = (outer["min_x_m"], outer["max_x_m"], corridor_bounds["max_z_m"], outer["max_z_m"])
    else:
        rectangles["west"] = (outer["min_x_m"], corridor_bounds["min_x_m"], outer["min_z_m"], outer["max_z_m"])
        rectangles["east"] = (corridor_bounds["max_x_m"], outer["max_x_m"], outer["min_z_m"], outer["max_z_m"])
    region_ids = {}
    literal = group_literal or (statement or {}).get("literal")
    for direction in sorted(directions):
        min_x, max_x, min_z, max_z = rectangles[direction]
        if max_x <= min_x or max_z <= min_z:
            _add_unresolved(
                result,
                "group_region_out_of_range",
                "site_ref.site_snapshot.zones",
                "The central aisle and wall margin leave no usable placement area for one equipment group.",
                literal,
            )
            continue
        region_id = _PREFIX + "placement_" + direction + "_of_" + _axis_label(aisle["axis"]) + "_aisle"
        region_ids[direction] = region_id
        _upsert_zone(
            snapshot,
            _rect_zone(
                region_id,
                "placement region " + direction,
                (min_x + max_x) / 2.0,
                (min_z + max_z) / 2.0,
                max_x - min_x,
                max_z - min_z,
                _frame_id(snapshot),
            ),
        )
        _record_provenance(
            result,
            snapshot,
            f"zones[{region_id}]",
            literal,
            intent_id,
            "Placement-region bounds derived from the central aisle and wall margin.",
        )
    return region_ids


def _assign_instances_to_regions(text, semantic_intent, equipment_instances, headings, statement, region_ids, result):
    """Assign regions from literal group declarations and labeled references.

    A catalog model can represent more than one declared unit.  Model-level
    assignment is therefore safe only until the request distinguishes literal
    instances, such as "work table A" and "work table B", in different
    cardinal regions.  This function preserves that distinction by binding
    only the declaration's actual quantity indices, never by interpreting a
    letter as an ordinal on its own.
    """

    mentions = [
        item
        for item in (semantic_intent or {}).get("asset_mentions") or []
        if isinstance(item, dict)
        and str(item.get("model_asset_id") or "").strip()
        and str(item.get("surface_text") or "").strip()
    ]
    if not mentions:
        _add_unresolved(
            result,
            "group_region_bindings",
            "equipment_instances.allowed_regions",
            "The equipment groups cannot be mapped because their resolved source mentions are unavailable.",
            (statement or {}).get("literal") or _first_direction_heading_literal_from_items(headings),
        )
        return
    declarations = _region_declarations(
        text,
        mentions,
        equipment_instances,
    )
    quantity_indices_by_instance = _instance_quantity_indices(
        equipment_instances,
    )

    direction_by_instance = {}
    literal_by_instance = {}
    assignment_origin = {}

    def assign(instance_key, direction, literal, origin):
        prior = direction_by_instance.get(instance_key)
        if prior and prior != direction:
            # A broad prose grouping is supplementary.  It cannot undo a
            # more specific heading or labeled instance assignment.
            if origin == "statement" and assignment_origin.get(instance_key) == "heading":
                return
            _add_unresolved(
                result,
                "group_region_conflict",
                "equipment_instances.allowed_regions",
                "One declared equipment instance was assigned to more than one placement group.",
                literal,
            )
            return
        direction_by_instance[instance_key] = direction
        literal_by_instance[instance_key] = literal
        assignment_origin[instance_key] = origin

    groups = [
        {**group, "origin": "heading"}
        for group in headings
    ]
    if statement:
        groups.extend(
            {
                "direction": item["direction"],
                "content": item["description"],
                "literal": statement["literal"],
                "origin": "statement",
            }
            for item in statement["groups"]
        )
    for group in groups:
        direction = group.get("direction")
        if direction not in region_ids:
            continue
        for declaration in declarations:
            quantity_indices = _declaration_indices_for_group(
                declaration,
                group.get("content"),
            )
            if not quantity_indices:
                continue
            for quantity_index in quantity_indices:
                assign(
                    (declaration["model_asset_id"], quantity_index),
                    direction,
                    group.get("literal"),
                    group.get("origin") or "heading",
                )

    # RAG may resolve a catalog type name such as "Storage rack" while the
    # accepted instance role retains the longer user label "Raw-material
    # storage rack". Resolve those explicit heading aliases from the stable
    # role and quantity index when the catalog mention surface is not a literal
    # substring of the short group list.
    for group in groups:
        direction = group.get("direction")
        if direction not in region_ids or group.get("origin") != "heading":
            continue
        for instance in equipment_instances or []:
            model_asset_id = str(instance.get("model_asset_id") or "")
            instance_id = str(instance.get("instance_id") or "")
            quantity_index = quantity_indices_by_instance.get(instance_id)
            role = str(instance.get("role") or "").strip()
            instance_key = (model_asset_id, quantity_index)
            if (
                not model_asset_id
                or quantity_index is None
                or instance_key in direction_by_instance
                or not _instance_role_listed_in_group(role, group.get("content"))
            ):
                continue
            assign(
                instance_key,
                direction,
                group.get("literal"),
                "heading_role_alias",
            )

    for reference in _directional_labeled_instance_references(text, declarations):
        if reference["direction"] not in region_ids:
            continue
        assign(
            (reference["model_asset_id"], reference["quantity_index"]),
            reference["direction"],
            reference["literal"],
            "instance_reference",
        )

    if not direction_by_instance:
        _add_unresolved(
            result,
            "group_region_bindings",
            "equipment_instances.allowed_regions",
            "No resolved equipment mention could be matched to the stated placement groups.",
            (statement or {}).get("literal") or _first_direction_heading_literal_from_items(headings),
        )
        return
    for instance in equipment_instances or []:
        model_asset_id = str(instance.get("model_asset_id") or "")
        quantity_index = quantity_indices_by_instance.get(
            str(instance.get("instance_id") or "")
        )
        if quantity_index is None:
            continue
        instance_key = (model_asset_id, quantity_index)
        direction = direction_by_instance.get(instance_key)
        if not direction:
            continue
        instance_id = instance.get("instance_id")
        if instance_id:
            result["instance_region_assignments"][instance_id] = {
                "region_id": region_ids[direction],
                "source_text": literal_by_instance[instance_key],
            }


def _region_declarations(text, mentions, equipment_instances):
    """Expand resolved literal declarations to actual per-model quantity slots."""

    instances_by_model = _instances_by_model(equipment_instances)
    quantity_indices_by_instance = _instance_quantity_indices(
        equipment_instances,
    )

    grouped = {}
    for order, mention in enumerate(mentions or []):
        surface_text = str(mention.get("surface_text") or "").strip()
        model_asset_id = str(mention.get("model_asset_id") or "").strip()
        quantity = _positive_int(mention.get("quantity")) or 1
        if not surface_text or not model_asset_id:
            continue
        literal_surface = _literal_quantity_declaration_surface(
            text,
            surface_text,
            quantity,
        )
        if literal_surface is None:
            continue
        key = (model_asset_id, _normalized(literal_surface))
        if key in grouped:
            continue
        position = _literal_position(text, literal_surface)
        grouped[key] = {
            "surface_text": literal_surface,
            "model_asset_id": model_asset_id,
            "quantity": quantity,
            "labels": _declared_labels(literal_surface, quantity),
            "position": position if position is not None else math.inf,
            "order": order,
        }

    declarations = []
    for model_asset_id, instances in instances_by_model.items():
        available_indices = [
            quantity_indices_by_instance.get(
                str(instance.get("instance_id") or "")
            )
            for instance in instances
        ]
        cursor = 0
        records = sorted(
            (
                item
                for item in grouped.values()
                if item["model_asset_id"] == model_asset_id
            ),
            key=lambda item: (item["position"], item["order"]),
        )
        for record in records:
            quantity = record["quantity"]
            indices = tuple(available_indices[cursor : cursor + quantity])
            if len(indices) != quantity:
                continue
            cursor += quantity
            declarations.append({**record, "quantity_indices": indices})
    return declarations


def _declaration_indices_for_group(declaration, group_content):
    surface_key = _normalized(declaration["surface_text"])
    content_key = _normalized(group_content)
    if surface_key and surface_key in content_key:
        return tuple(declaration.get("quantity_indices") or ())
    labels = tuple(declaration.get("labels") or ())
    if labels:
        return _labeled_declaration_indices_in_text(
            declaration,
            group_content,
        )
    if not _mention_matches_text(declaration["surface_text"], group_content):
        return ()
    return tuple(declaration.get("quantity_indices") or ())


def _instance_role_listed_in_group(role, group_content):
    role_tokens = _region_tokens(role)
    if not role_tokens:
        return False
    role_label = _functional_role_label(role)
    for alias in re.split(r"\s*(?:;|,|\band\b)\s*", str(group_content or ""), flags=re.I):
        alias_tokens = _region_tokens(alias)
        if not alias_tokens:
            continue
        alias_label = _functional_role_label(alias)
        if alias_label and (not role_label or alias_label != role_label):
            continue
        if alias_label and not any(token != alias_label for token in alias_tokens):
            # Splitting "family A and B" on conjunctions leaves a bare
            # label. It cannot identify which equipment family owns it.
            continue
        if all(token in role_tokens for token in alias_tokens):
            return True
    return False


def _functional_role_label(value):
    tokens = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    if not tokens:
        return None
    token = tokens[-1]
    return token if token.isdigit() or len(token) == 1 and token.isalpha() else None


def _directional_labeled_instance_references(text, declarations):
    """Find literal per-instance cardinal statements without guessing IDs."""

    results = []
    for sentence in _sentences(text):
        for declaration in declarations:
            labels = tuple(declaration.get("labels") or ())
            for label_position, label in enumerate(labels):
                label_pattern = _declared_label_reference_pattern(
                    declaration,
                    label,
                )
                if label_pattern is None:
                    continue
                pattern = re.compile(
                    label_pattern
                    + r"\s+(?:(?:belongs|is|should\s+be|must\s+be|will\s+be|"
                    r"needs\s+to\s+be|is\s+parked|parked|placed|located)\s+)?"
                    r"(?:in|on|to)\s+(?:the\s+)?"
                    r"(?P<direction>south(?:ern)?|north(?:ern)?|west(?:ern)?|east(?:ern)?)"
                    r"(?:\s+(?:area|region|side|zone))?\b",
                    re.IGNORECASE,
                )
                for match in pattern.finditer(sentence):
                    direction = _canonical_direction(match.group("direction"))
                    indices = tuple(declaration.get("quantity_indices") or ())
                    if direction is None or label_position >= len(indices):
                        continue
                    results.append(
                        {
                            "model_asset_id": declaration["model_asset_id"],
                            "quantity_index": indices[label_position],
                            "direction": direction,
                            "literal": sentence,
                        }
                    )
    return results


def _labeled_declaration_indices_in_text(declaration, text):
    indices = tuple(declaration.get("quantity_indices") or ())
    values = []
    for label_position, label in enumerate(declaration.get("labels") or ()):
        pattern = _declared_label_reference_pattern(declaration, label)
        if (
            pattern is not None
            and re.search(pattern, str(text or ""), flags=re.IGNORECASE)
            and label_position < len(indices)
        ):
            values.append(indices[label_position])
    return tuple(values)


def _declared_label_reference_pattern(declaration, label):
    base_tokens = _region_tokens(
        _declaration_label_base(declaration.get("surface_text"))
    )
    if not base_tokens or not label:
        return None
    words = []
    for token in base_tokens:
        stem = re.escape(token[:-1] if len(token) > 3 and token.endswith("s") else token)
        words.append(r"\b" + stem + r"s?\b")
    return (
        r"(?:" + r"\s+".join(words) + r")"
        r"(?:\s+(?:labeled|labelled))?\s+"
        + re.escape(str(label))
        + r"\b"
    )


def _declared_labels(surface_text, quantity):
    match = re.search(
        r"\b(?:labeled|labelled)\s+(.+)$",
        str(surface_text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return ()
    return _leading_declared_labels(match.group(1), quantity)


def _leading_declared_labels(value, quantity):
    """Read exactly the declared labels before a descriptive suffix."""

    if quantity < 1:
        return ()
    token = r"[A-Za-z0-9][A-Za-z0-9_-]*"
    sequence = token + (rf"\s*(?:,|and|or)\s*{token}" * (quantity - 1))
    match = re.match(
        rf"^\s*(?P<labels>{sequence})(?=\s*(?:$|[;.]|,|\bfor\b|\bused\b|\bwith\b|\bthat\b|\bwhich\b|\boutside\b))",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return ()
    labels = tuple(
        token.casefold()
        for token in re.findall(token, match.group("labels"))
        if token.casefold() not in {"and", "or"}
    )
    return labels if len(labels) == quantity and len(set(labels)) == quantity else ()


def _declaration_label_base(surface_text):
    value = re.split(r"\b(?:labeled|labelled)\b", str(surface_text or ""), maxsplit=1, flags=re.IGNORECASE)[0]
    value = re.sub(
        r"^\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


def _region_tokens(value):
    stop_words = {
        "a", "an", "and", "the", "for", "of", "to", "in", "on",
        "labeled", "labelled", "equipment", "unit", "units", "one", "two",
        "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    }
    tokens = []
    for token in re.findall(r"[a-z0-9]+", str(value or "").casefold()):
        if token in stop_words:
            continue
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.append(token)
    return tuple(tokens)


def _canonical_direction(value):
    lowered = str(value or "").casefold()
    return {
        "southern": "south",
        "northern": "north",
        "western": "west",
        "eastern": "east",
    }.get(lowered, lowered if lowered in _CARDINAL_DIRECTIONS else None)


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _instances_by_model(equipment_instances):
    grouped = {}
    for order, instance in enumerate(equipment_instances or []):
        if not isinstance(instance, dict):
            continue
        model_asset_id = str(instance.get("model_asset_id") or "").strip()
        instance_id = str(instance.get("instance_id") or "").strip()
        if not model_asset_id or not instance_id:
            continue
        grouped.setdefault(model_asset_id, []).append((order, instance))
    result = {}
    for model_asset_id, values in grouped.items():
        result[model_asset_id] = [
            item
            for _order, item in sorted(
                values,
                key=lambda pair: (
                    _positive_int(pair[1].get("quantity_index")) or math.inf,
                    pair[0],
                    str(pair[1].get("instance_id") or ""),
                ),
            )
        ]
    return result


def _instance_quantity_indices(equipment_instances):
    """Return one stable, per-model index for every concrete instance."""

    result = {}
    for _model_asset_id, instances in _instances_by_model(
        equipment_instances
    ).items():
        used = set()
        for instance in instances:
            explicit = _positive_int(instance.get("quantity_index"))
            if explicit is not None and explicit not in used:
                result[str(instance.get("instance_id"))] = explicit
                used.add(explicit)
        next_index = 1
        for instance in instances:
            instance_id = str(instance.get("instance_id"))
            if instance_id in result:
                continue
            while next_index in used:
                next_index += 1
            result[instance_id] = next_index
            used.add(next_index)
    return result


def _is_explicit_quantity_declaration(surface_text):
    return bool(
        re.match(
            r"\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
            str(surface_text or ""),
            flags=re.IGNORECASE,
        )
    )


_SURFACE_QUANTITY_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def quantity_insensitive_surface_equivalent(
    left,
    right,
    *,
    left_quantity=None,
    right_quantity=None,
):
    """Match a literal declaration to an omission of its same count prefix.

    This deliberately does not remove arbitrary leading numerals.  A count can
    be omitted only when the retained typed quantity agrees with that literal
    declaration.  Thus ``two CNC machines`` can match ``CNC machines`` with
    quantity two, while ``three-axis CNC`` cannot match ``axis CNC`` with
    quantity one and explicit one/two conflicts remain distinct.
    """

    left_key = _normalized(left)
    right_key = _normalized(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        left_count = _leading_surface_quantity(left)
        right_count = _leading_surface_quantity(right)
        return not (
            left_count is not None
            and right_count is not None
            and left_count != right_count
        )
    left_count, left_base = _leading_surface_quantity_and_base(left)
    right_count, right_base = _leading_surface_quantity_and_base(right)
    if left_count is not None and right_count is not None:
        return left_count == right_count and left_base == right_base
    if left_count is not None:
        return (
            left_base == right_key
            and _positive_int(right_quantity) == left_count
        )
    if right_count is not None:
        return (
            right_base == left_key
            and _positive_int(left_quantity) == right_count
        )
    return False


def _literal_quantity_declaration_surface(text, surface_text, quantity):
    """Find the literal counted declaration that supports a normalized mention."""

    if _is_explicit_quantity_declaration(surface_text):
        return surface_text if _literal_position(text, surface_text) is not None else None
    target_tokens = _normalized(surface_text).split()
    expected_quantity = _positive_int(quantity)
    if not target_tokens or expected_quantity is None:
        return None
    for match in re.finditer(
        r"(?=(?P<full>\b(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
        r"(?P<body>[^;.\n]+)))",
        str(text or ""),
        flags=re.IGNORECASE,
    ):
        count = _surface_quantity_value(match.group("count"))
        body_tokens = _normalized(match.group("body")).split()
        if (
            count == expected_quantity
            and body_tokens[: len(target_tokens)] == target_tokens
        ):
            return match.group("full").strip()
    return None


def _leading_surface_quantity(value):
    count, _base = _leading_surface_quantity_and_base(value)
    return count


def _leading_surface_quantity_and_base(value):
    text = str(value or "").strip()
    match = re.match(
        r"^(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?P<body>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None, _normalized(text)
    return _surface_quantity_value(match.group("count")), _normalized(match.group("body"))


def _surface_quantity_value(value):
    text = str(value or "").casefold()
    if text.isdigit():
        return _positive_int(text)
    return _SURFACE_QUANTITY_WORDS.get(text)


def _literal_position(text, value):
    position = str(text or "").casefold().find(str(value or "").casefold())
    return position if position >= 0 else None


def _resize_adjacent_regions(snapshot, corridor, axis):
    corridor_bounds = _feature_bounds(corridor)
    if corridor_bounds is None:
        return False
    directions = ("south", "north") if axis == "east_west" else ("west", "east")
    zones_by_id = {
        item.get("zone_id"): item
        for item in snapshot.get("zones") or []
        if isinstance(item, dict)
    }
    ids = {
        direction: _PREFIX + "placement_" + direction + "_of_" + _axis_label(axis) + "_aisle"
        for direction in directions
    }
    if not all(zone_id in zones_by_id for zone_id in ids.values()):
        return False
    rectangles = {direction: _zone_bounds(zones_by_id[zone_id]) for direction, zone_id in ids.items()}
    if any(rectangle is None for rectangle in rectangles.values()):
        return False
    if axis == "east_west":
        south = rectangles["south"]
        north = rectangles["north"]
        updates = {
            "south": (south["min_x_m"], south["max_x_m"], south["min_z_m"], corridor_bounds["min_z_m"]),
            "north": (north["min_x_m"], north["max_x_m"], corridor_bounds["max_z_m"], north["max_z_m"]),
        }
    else:
        west = rectangles["west"]
        east = rectangles["east"]
        updates = {
            "west": (west["min_x_m"], corridor_bounds["min_x_m"], west["min_z_m"], west["max_z_m"]),
            "east": (corridor_bounds["max_x_m"], east["max_x_m"], east["min_z_m"], east["max_z_m"]),
        }
    if any(max_x <= min_x or max_z <= min_z for min_x, max_x, min_z, max_z in updates.values()):
        return False
    for direction, (min_x, max_x, min_z, max_z) in updates.items():
        zone = zones_by_id[ids[direction]]
        _set_rect(zone, min_x, max_x, min_z, max_z)
    return True


def _resize_workstation_bays(snapshot, result, literal, intent_id):
    """Move organized row rectangles with the resized south placement band."""
    row_pattern = re.compile(
        r"^" + re.escape(_PREFIX) + r"workstation_bay_row_(\d+)_column_(\d+)$"
    )
    zones_by_id = {
        item.get("zone_id"): item
        for item in snapshot.get("zones") or []
        if isinstance(item, dict)
    }
    row_zones = {}
    for zone_id, zone in zones_by_id.items():
        match = row_pattern.match(str(zone_id or ""))
        if match:
            row_zones.setdefault(int(match.group(1)), []).append((int(match.group(2)), zone))
    if not row_zones:
        return True

    south_id = _PREFIX + "placement_south_of_east_west_aisle"
    south_region = zones_by_id.get(south_id)
    south_bounds = _zone_bounds(south_region) if south_region else None
    if not south_bounds:
        return False
    row_ids = sorted(
        row_zones,
        key=lambda row_id: (
            sum(float((zone.get("pose") or {}).get("z_m") or 0.0) for _, zone in row_zones[row_id])
            / max(1, len(row_zones[row_id])),
            row_id,
        ),
    )
    row_depths = {}
    for row_id in row_ids:
        depths = []
        for _, zone in row_zones[row_id]:
            try:
                depths.append(float((zone.get("dimensions_m") or {})["depth_m"]))
            except (KeyError, TypeError, ValueError):
                return False
        if not depths or max(depths) - min(depths) > 1e-6:
            return False
        row_depths[row_id] = depths[0]

    band_depth = south_bounds["max_z_m"] - south_bounds["min_z_m"]
    required_depth = sum(row_depths.values())
    if band_depth <= 0.0 or required_depth > band_depth + 1e-9:
        return False
    gap = (band_depth - required_depth) / (len(row_ids) - 1) if len(row_ids) > 1 else 0.0
    row_bounds = {}
    cursor = south_bounds["min_z_m"]
    for position, row_id in enumerate(row_ids):
        depth = row_depths[row_id]
        south = cursor
        north = south + depth
        if position == len(row_ids) - 1:
            north = south_bounds["max_z_m"]
            south = north - depth
        row_bounds[row_id] = (south, north)
        cursor = north + gap

    for row_id, cells in row_zones.items():
        south, north = row_bounds[row_id]
        for column_id, zone in cells:
            bay_id = zone["zone_id"]
            bounds = _zone_bounds(zone)
            if bounds is None:
                return False
            _set_rect(zone, bounds["min_x_m"], bounds["max_x_m"], south, north)
            _record_provenance(
                result,
                snapshot,
                f"zones[{bay_id}].pose",
                literal,
                intent_id,
                "Repositioned the source-ordered workstation bay with its resized south placement band.",
            )
            result.setdefault("changed_zone_ids", []).append(bay_id)
            placement_id = bay_id + "/placement"
            placement = zones_by_id.get(placement_id)
            if placement is not None:
                placement_bounds = _zone_bounds(placement)
                if placement_bounds is None:
                    return False
                _set_rect(
                    placement,
                    placement_bounds["min_x_m"],
                    placement_bounds["max_x_m"],
                    south,
                    north,
                )
                _record_provenance(
                    result,
                    snapshot,
                    f"zones[{placement_id}].pose",
                    literal,
                    intent_id,
                    "Kept the assigned workstation placement region aligned with its resized bay.",
                )
                result.setdefault("changed_zone_ids", []).append(placement_id)
    return True


def _resize_functional_north_regions(snapshot, corridor, result, literal, intent_id):
    """Keep NW/NE functional zones attached to the north edge of the aisle."""
    zone_ids = ("functional/northwest", "functional/northeast")
    zones_by_id = {
        item.get("zone_id"): item
        for item in snapshot.get("zones") or []
        if isinstance(item, dict)
    }
    functional = [zones_by_id[zone_id] for zone_id in zone_ids if zone_id in zones_by_id]
    if not functional:
        return True
    bounds = _site_bounds(snapshot)
    corridor_bounds = _feature_bounds(corridor)
    if bounds is None or corridor_bounds is None:
        return False
    min_z = corridor_bounds["max_z_m"]
    max_z = bounds["max_z_m"]
    if max_z <= min_z:
        return False
    for zone in functional:
        zone_bounds = _zone_bounds(zone)
        if zone_bounds is None or zone_bounds["max_x_m"] <= zone_bounds["min_x_m"]:
            return False
        _set_rect(zone, zone_bounds["min_x_m"], zone_bounds["max_x_m"], min_z, max_z)
        _record_provenance(
            result,
            snapshot,
            f"zones[{zone['zone_id']}].pose",
            literal,
            intent_id,
            "Kept the functional region aligned with the north edge of the updated central aisle.",
        )
        result.setdefault("changed_zone_ids", []).append(zone["zone_id"])
    return True


def _rect_feature(feature_id, feature_type, name, x_m, z_m, width_m, depth_m, frame):
    return {
        "site_feature_id": feature_id,
        "feature_type": feature_type,
        "feature_name": name,
        "pose": _pose(x_m, z_m, frame),
        "dimensions_m": {"width_m": width_m, "depth_m": depth_m, "height_m": 0.1},
        "must_remain_clear": True,
        "source": "user_input",
        "source_model_asset_id": None,
        "source_node_ref": None,
        "confidence": 1.0,
    }


def _buffer_feature(buffer, frame):
    return _rect_feature(
        buffer["feature_id"],
        "forbidden_zone",
        buffer["name"] + " buffer",
        buffer["center_x_m"],
        buffer["center_z_m"],
        buffer["width_m"],
        buffer["depth_m"],
        frame,
    )


def _rect_zone(zone_id, name, x_m, z_m, width_m, depth_m, frame):
    return {
        "zone_id": zone_id,
        "zone_type": "placement_region",
        "zone_name": name,
        "pose": _pose(x_m, z_m, frame),
        "dimensions_m": {"width_m": width_m, "depth_m": depth_m, "height_m": 0.1},
        "allowed_categories": [],
        "forbidden_categories": [],
        "source": "computed",
        "source_model_asset_id": None,
        "source_node_ref": None,
        "confidence": 1.0,
    }


def _pose(x_m, z_m, frame):
    return {
        "x_m": x_m,
        "y_m": 0.0,
        "z_m": z_m,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "coordinate_frame_id": frame,
    }


def _upsert_feature(snapshot, feature):
    features = snapshot.setdefault("site_features", [])
    for index, item in enumerate(features):
        if item.get("site_feature_id") == feature["site_feature_id"]:
            features[index] = feature
            return
    features.append(feature)


def _upsert_zone(snapshot, zone):
    zones = snapshot.setdefault("zones", [])
    for index, item in enumerate(zones):
        if item.get("zone_id") == zone["zone_id"]:
            zones[index] = zone
            return
    zones.append(zone)


def _record_provenance(result, snapshot, field_path, literal, intent_id, note):
    record = {
        "field_path": "site_ref.site_snapshot." + field_path,
        "source_type": "user_input",
        "source_id": intent_id,
        "source_field": str(literal or ""),
        "confidence": 1.0,
        "note": note,
    }
    result["provenance"].append(record)
    if snapshot.get("schema_version") != "inline-site-v1":
        snapshot.setdefault("provenance", []).append(
            {**record, "field_path": field_path}
        )


def _add_constraint_requirement(result, constraint_type, source_text):
    if any(item["constraint_type"] == constraint_type for item in result["constraint_requirements"]):
        return
    result["constraint_requirements"].append(
        {"constraint_type": constraint_type, "source_text": source_text}
    )


def _add_unresolved(result, item_id, field_path, description, literal):
    record = {
        "item_id": "natural_site_" + item_id,
        "item_type": "missing_field",
        "field_path": field_path,
        "description": description,
        "severity": "blocking",
        "candidates": [],
        "resolution_policy": "ask_user",
    }
    if not any(item.get("item_id") == record["item_id"] for item in result["unresolved_items"]):
        result["unresolved_items"].append(record)


def _existing_wall_margin(snapshot, bounds):
    values = []
    for feature in snapshot.get("site_features") or []:
        feature_id = str(feature.get("site_feature_id") or "")
        if not feature_id.startswith(_PREFIX + "wall_margin_"):
            continue
        dimensions = feature.get("dimensions_m") or {}
        if feature_id.endswith(("west", "east")):
            value = dimensions.get("width_m")
        else:
            value = dimensions.get("depth_m")
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return 0.0
    return max(values)


def _has_natural_wall_margin(snapshot):
    return any(
        str(item.get("site_feature_id") or "").startswith(_PREFIX + "wall_margin_")
        for item in (snapshot or {}).get("site_features") or []
        if isinstance(item, dict)
    )


def _has_natural_central_aisle(snapshot):
    return any(
        str(item.get("site_feature_id") or "").startswith(_PREFIX + "central_")
        and str(item.get("site_feature_id") or "").endswith("_aisle")
        and str(item.get("feature_type") or "").casefold() == "corridor"
        for item in (snapshot or {}).get("site_features") or []
        if isinstance(item, dict)
    )


def _has_existing_corridor(snapshot):
    return any(
        str(item.get("feature_type") or "").casefold() in {"corridor", "aisle"}
        for item in (snapshot or {}).get("site_features") or []
        if isinstance(item, dict)
    )


def _has_natural_buffers(snapshot):
    return any(
        str(item.get("site_feature_id") or "").startswith(_PREFIX + "buffer_")
        for item in (snapshot or {}).get("site_features") or []
        if isinstance(item, dict)
    )


def _feature_change_requested(text, feature):
    feature_words = {
        "aisle": r"(?:aisle|corridor)",
        "buffer": r"(?:buffer|buffers)",
        "wall": r"(?:wall|walls|wall\s+margin|wall\s+clearance)",
    }[feature]
    value = str(text or "")
    operations = r"(?:move|resize|change|update|increase|decrease|widen|narrow|relocate|shift)"
    # Remove explicit negated actions before detecting affirmative updates.
    # Keep other clauses intact (e.g. "Do not move racks; move the buffer").
    negated = (
        r"\bno\s+" + feature_words + r"\s+(?:changes?|updates?|moves?|relocations?)\b",
        r"\bno\s+(?:changes?|updates?|moves?)\s+to\s+(?:the\s+)?" + feature_words + r"\b",
        r"\b(?:do\s+)?not\s+" + operations + r"\b",
        r"\b" + feature_words + r"\s+(?:(?:must|should|shall|will|do|does)\s+)?not\s+(?:be\s+)?" + operations + r"(?:d|ed)?\b",
    )
    for pattern in negated:
        value = re.sub(pattern, lambda match: " " * len(match.group(0)), value, flags=re.IGNORECASE)
    return bool(
        re.search(
            r"\b(?:move|resize|change|update|increase|decrease|widen|narrow|relocate|shift)\b"
            r"[^.!?\n;]{0,80}\b"
            + feature_words
            + r"\b|\b"
            + feature_words
            + r"\b[^.!?\n;]{0,80}\b(?:move|resize|change|update|increase|decrease|widen|narrow|relocate|shift)\b",
            value,
            re.IGNORECASE,
        )
    )


def _preserves_existing_aisle(text):
    value = str(text or "")
    if not _mentions_aisle(value):
        return False
    if re.search(r"\b(?:same|unchanged|as[-\s]is)\b[^.!?\n]{0,60}\b(?:aisle|corridor)\b", value, re.IGNORECASE):
        return True
    if re.search(r"\b(?:aisle|corridor)\b[^.!?\n]{0,60}\b(?:same|unchanged|as[-\s]is)\b", value, re.IGNORECASE):
        return True
    if re.search(r"\b(?:keep|preserve|retain|leave)\b[^.!?\n]{0,60}\b(?:aisle|corridor)\b", value, re.IGNORECASE) and not re.search(
        r"\d+(?:\.\d+)?\s*(?:m|met(?:er|re)s?)\s*(?:-\s*)?wide", value, re.IGNORECASE
    ):
        return True
    return False


def _axis_from_text(text):
    lowered = str(text or "").casefold()
    east_west = bool(re.search(r"\beast\s*(?:to|-)\s*west\b|\be-?w\b", lowered))
    north_south = bool(re.search(r"\bnorth\s*(?:to|-)\s*south\b|\bn-?s\b", lowered))
    if east_west == north_south:
        return None
    return "east_west" if east_west else "north_south"


def _axis_from_feature(feature):
    feature_id = str(feature.get("site_feature_id") or "")
    if "east_west" in feature_id:
        return "east_west"
    if "north_south" in feature_id:
        return "north_south"
    return None


def _aisle_width_update(text):
    candidates = []
    for sentence in _sentences(text):
        if not re.search(r"\b(?:aisle|corridor)\b", sentence, re.IGNORECASE):
            continue
        target = re.search(
            r"\b(?:to|at)\s+(?P<value>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)\s*(?:-\s*)?wide\b",
            sentence,
            re.IGNORECASE,
        )
        if target is None:
            target = re.search(
                r"\b(?:widen|expand|increase|set)\s+(?:the\s+)?(?:central\s+)?"
                r"(?:(?:east\s*(?:-\s*|\s+)west|north\s*(?:-\s*|\s+)south|e-?w|n-?s)\s+)?"
                r"(?:logistics\s+)?(?:aisle|corridor)\b"
                r"[^.!?\n]{0,120}?\bto\s+(?P<value>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)\b",
                sentence, re.IGNORECASE,
            )
        if target is None:
            target = re.search(
                r"(?P<value>\d+(?:\.\d+)?)\s*(?:m|met(?:er|re)s?)\s*(?:-\s*)?wide\s+(?:main\s+)?(?:logistics\s+)?(?:aisle|corridor)\b",
                sentence,
                re.IGNORECASE,
            )
        if target is None:
            target = re.search(
                r"\b(?:aisle|corridor)\b[^.!?\n]{0,100}?\b"
                r"(?:must\s+now\s+be|should\s+be|needs?\s+to\s+be|is|be)\s+"
                r"(?P<value>\d+(?:\.\d+)?)\s*"
                r"(?:m|met(?:er|re)s?)\s*(?:-\s*)?wide\b",
                sentence,
                re.IGNORECASE,
            )
        if target:
            candidates.append(
                {
                    "width_m": float(target.group("value")),
                    "axis": _axis_from_text(sentence),
                    "literal": sentence.strip(),
                }
            )
    if not candidates:
        return None
    first = candidates[0]
    if any(not math.isclose(item["width_m"], first["width_m"], abs_tol=1e-9) for item in candidates[1:]):
        return None
    return first


def _feature_bounds(feature):
    pose = feature.get("pose") or {}
    dimensions = feature.get("dimensions_m") or {}
    try:
        x_m = float(pose.get("x_m"))
        z_m = float(pose.get("z_m"))
        width = float(dimensions.get("width_m"))
        depth = float(dimensions.get("depth_m"))
    except (TypeError, ValueError):
        return None
    return {
        "min_x_m": x_m - width / 2.0,
        "max_x_m": x_m + width / 2.0,
        "min_z_m": z_m - depth / 2.0,
        "max_z_m": z_m + depth / 2.0,
    }


def _zone_bounds(zone):
    return _feature_bounds(zone)


def _set_rect(item, min_x, max_x, min_z, max_z):
    item["pose"] = {
        **(item.get("pose") or {}),
        "x_m": (min_x + max_x) / 2.0,
        "z_m": (min_z + max_z) / 2.0,
    }
    item["dimensions_m"] = {
        **(item.get("dimensions_m") or {}),
        "width_m": max_x - min_x,
        "depth_m": max_z - min_z,
    }


def _central_feature(snapshot, axis):
    feature_id = _PREFIX + "central_" + _axis_label(axis) + "_aisle"
    return next(
        (
            item
            for item in snapshot.get("site_features") or []
            if item.get("site_feature_id") == feature_id
        ),
        None,
    )


def _axis_label(axis):
    return "east_west" if axis == "east_west" else "north_south"


def _frame_id(snapshot):
    return ((snapshot.get("boundary") or {}).get("pose") or {}).get("coordinate_frame_id")


def _coordinate_from_edge(bounds, offset, edge):
    if edge == "west":
        return bounds["min_x_m"] + offset
    if edge == "east":
        return bounds["max_x_m"] - offset
    if edge == "south":
        return bounds["min_z_m"] + offset
    return bounds["max_z_m"] - offset


def _rectangle_inside(center_x, center_z, width, depth, bounds, margin):
    return (
        center_x - width / 2.0 >= bounds["min_x_m"] + margin - 1e-9
        and center_x + width / 2.0 <= bounds["max_x_m"] - margin + 1e-9
        and center_z - depth / 2.0 >= bounds["min_z_m"] + margin - 1e-9
        and center_z + depth / 2.0 <= bounds["max_z_m"] - margin + 1e-9
    )


def _valid_margin(margin, bounds):
    return margin >= 0.0 and 2.0 * margin < bounds["width_m"] and 2.0 * margin < bounds["depth_m"]


def _aisle_fits(aisle, bounds, margin):
    available = bounds["depth_m"] - 2.0 * margin if aisle["axis"] == "east_west" else bounds["width_m"] - 2.0 * margin
    return aisle["width_m"] > 0.0 and aisle["width_m"] < available


def _mention_matches_text(surface_text, text):
    surface = _normalized(surface_text)
    target = _normalized(text)
    if not surface or not target:
        return False
    if surface in target:
        return True
    tokens = tuple(
        token
        for token in _region_tokens(surface)
        if len(token) >= 3
    )
    target_tokens = set(_region_tokens(target))
    # Do not let connective words or one generic noun turn an unrelated
    # heading into a placement assignment.  The exact-span path above still
    # accepts a complete declaration where that is the user's wording.
    return len(set(tokens)) >= 2 and len(set(tokens) & target_tokens) >= 2


def _normalized(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _slug(value):
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", str(value or "").casefold())).strip("-")


def _sentences(text):
    return [
        part.strip()
        for part in re.split(r"(?<!\d)[.!?](?!\d)|\n+", str(text or ""))
        if part.strip()
    ]


def _mentions_wall_margin(text):
    if _wall_margin(text)[0] is not None:
        return True
    return bool(
        re.search(r"\bwall\s+(?:margin|clearance)\b", str(text or ""), re.IGNORECASE)
    )


def _mentions_aisle(text):
    return bool(re.search(r"\b(?:aisle|corridor)\b", text, re.IGNORECASE))


def _mentions_group_direction(text):
    return bool(re.search(r"\b(?:south|north|west|east)\s+of\s+(?:the\s+)?(?:aisle|corridor)\b", text, re.IGNORECASE))


def _mentions_site_feature(text):
    return _mentions_wall_margin(text) or _mentions_aisle(text) or bool(re.search(r"\bbuffers?\b", text, re.IGNORECASE))


def _first_sentence_matching(text, pattern):
    matcher = re.compile(pattern, re.IGNORECASE)
    return next((sentence for sentence in _sentences(text) if matcher.search(sentence)), "")


def _first_site_sentence(text):
    return _first_sentence_matching(text, r"\b\d+(?:\.\d+)?\s*(?:m|met(?:er|re)s?)\s*(?:east|north|x|by)")


def _first_spatial_sentence(text):
    return _first_sentence_matching(text, r"\b(?:wall|aisle|corridor|buffer|south|north|west|east)\b")


def _first_direction_heading_literal(text):
    headings = _group_headings(text)
    return _first_direction_heading_literal_from_items(headings)


def _first_direction_heading_literal_from_items(headings):
    return headings[0]["literal"] if headings else ""


def _declared_buffer_count(text):
    match = re.search(r"\b(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:fixed\s+)?(?:[\d.]+\s*m\s*(?:by|x|×)\s*[\d.]+\s*m\s+)?buffers?(?:\s+areas?)?\b", text, re.IGNORECASE)
    if not match:
        return None
    value = match.group("count").casefold()
    return int(value) if value.isdigit() else _NUMBER_WORDS.get(value)
