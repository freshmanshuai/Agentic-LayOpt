"""Compile source-backed workstation rows and reserved bays into site zones."""

from __future__ import annotations

import copy
import math
import re

from tool.context.instance_naming import declaration_labels


_PREFIX = "natural_site/workstation_bay_"
_COUNTS = {
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
_COUNT = r"(?:\d+|" + "|".join(_COUNTS) + r")"
_LAYOUT = re.compile(
    rf"\b(?P<rows>{_COUNT})\s+(?:east[\s\-‐‑‒–—−]*west\s+)?rows?\s+"
    rf"(?:of|with)\s+(?P<bays>{_COUNT})\s+"
    r"(?:bays?|workstations?|stations?)\s+(?:each|per\s+row)\b",
    re.IGNORECASE,
)
_ROW = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:(?P<ordinal>row\s+(?:\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth))|"
    r"(?P<word>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+row)\s*,?\s*"
    r"(?P<direction>west\s+to\s+east|east\s+to\s+west)\s*:\s*(?P<items>[^\n]+)"
)
_ROW_NEAREST_AISLE = re.compile(
    r"\brow\s+(?P<row>\d+|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b"
    r"[^.;\n]{0,100}\b(?:closest|nearest)\s+to\s+(?:the\s+)?(?:central\s+)?aisle\b",
    re.IGNORECASE,
)
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}
_MAINTENANCE = re.compile(
    r"\bfor\s+(?P<target>[^,.;]+),\s*(?:reserve|allow|provide|keep)\s+"
    r"(?P<front>\d+(?:\.\d+)?)\s*m\s+(?:at|for)\s+(?:its\s+)?(?:service\s+)?front,\s*"
    r"(?P<rear>\d+(?:\.\d+)?)\s*m\s+(?:at|for)\s+(?:the\s+)?rear,\s*and\s+"
    r"(?P<side>\d+(?:\.\d+)?)\s*m\s+on\s+each\s+side",
    re.IGNORECASE,
)
_CLEARANCE_BULLET = re.compile(
    r"(?im)^\s*[-*]\s*(?P<target>[^:\n]+):\s*"
    r"(?P<front>\d+(?:\.\d+)?)\s*m\s+at\s+(?:the\s+)?(?:service\s+)?front,\s*"
    r"(?P<rear>\d+(?:\.\d+)?)\s*m\s+at\s+(?:the\s+)?rear,\s*and\s*"
    r"(?P<side>\d+(?:\.\d+)?)\s*m\s+on\s+each\s+side\b[^\n]*"
)
_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOP = {
    "a", "an", "and", "the", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten", "of", "for", "machine",
    "machines", "center", "centers", "centre", "centres", "equipment",
    "station", "stations", "workstation", "workstations", "labeled",
    "labelled", "bay", "bays",
}


def compile_organized_workstations(
    user_message,
    site_snapshot,
    *,
    semantic_intent=None,
    equipment_instances=None,
    intent_id="INTENT_001",
):
    """Compile row and bay statements into constrained placement regions.

    Each item is bound through its resolved mention and stable quantity index.
    The compiler does not infer a catalog model from a label or row position.
    """

    text = str(user_message or "")
    snapshot = copy.deepcopy(site_snapshot or {})
    result = {
        "site_snapshot": snapshot,
        "instance_region_assignments": {},
        "workstation_bays": [],
        "unresolved_items": [],
        "provenance": [],
        "constraint_requirements": [],
        "literal_spans": {"organization": "", "maintenance": []},
    }
    _apply_maintenance_assumptions(
        text,
        semantic_intent or {},
        equipment_instances or [],
        result,
        intent_id,
    )

    layout_matches = list(_LAYOUT.finditer(text))
    layout_match = layout_matches[-1] if layout_matches else None
    all_row_lines = list(_ROW.finditer(text))
    later_row_lines = (
        [match for match in all_row_lines if match.start() > layout_match.start()]
        if layout_match
        else []
    )
    row_lines = later_row_lines or all_row_lines
    if not layout_match and not row_lines:
        return result
    if not layout_match:
        _unresolved(
            result,
            "layout_count",
            "site_ref.site_snapshot.zones",
            "The workstation rows need an explicit row count and bay count per row.",
            row_lines[0].group(0),
        )
        return result

    row_count = _count(layout_match.group("rows"))
    bays_per_row = _count(layout_match.group("bays"))
    parsed_rows = {}
    for match in row_lines:
        row_index = _row_index(match)
        items = _split_row_items(match.group("items"))
        if match.group("direction").casefold().replace(" ", "") == "easttowest":
            items.reverse()
        parsed_rows[row_index] = {"items": items, "literal": match.group(0)}

    expected_indices = list(range(1, row_count + 1))
    if sorted(parsed_rows) != expected_indices:
        _unresolved(
            result,
            "row_declarations",
            "site_ref.site_snapshot.zones",
            "Each declared row needs one west-to-east bay list.",
            layout_match.group(0),
        )
        return result
    if any(len(parsed_rows[index]["items"]) != bays_per_row for index in expected_indices):
        _unresolved(
            result,
            "bay_count_mismatch",
            "site_ref.site_snapshot.zones",
            "Each workstation row must list the stated number of bays.",
            " ".join(parsed_rows[index]["literal"] for index in expected_indices),
        )
        return result
    nearest_rows = {
        _ordinal_value(match.group("row"))
        for match in _ROW_NEAREST_AISLE.finditer(text)
    }
    nearest_rows.discard(None)
    if len(nearest_rows) > 1 or any(index not in expected_indices for index in nearest_rows):
        _unresolved(
            result,
            "row_order",
            "site_ref.site_snapshot.zones",
            "The row facing the central aisle must resolve to one declared row.",
            " ".join(match.group(0) for match in _ROW_NEAREST_AISLE.finditer(text)),
        )
        return result
    nearest_row = next(iter(nearest_rows), None)

    mentions = _resolved_mentions(semantic_intent or {})
    instances = list(equipment_instances or [])
    row_region = _south_placement_region(snapshot)
    if row_region is None:
        _unresolved(
            result,
            "missing_south_region",
            "site_ref.site_snapshot.zones",
            "A resolved south-side placement region is required before workstation bays can be located.",
            layout_match.group(0),
        )
        return result

    bindings = {}
    seen_instances = set()
    binding_failed = False
    for row_index in expected_indices:
        row = parsed_rows[row_index]
        for column_index, label in enumerate(row["items"], start=1):
            if _is_reserved(label):
                bindings[(row_index, column_index)] = None
                continue
            matches = _resolve_item(label, text, mentions, instances)
            if len(matches) != 1:
                binding_failed = True
                _unresolved(
                    result,
                    "bay_reference",
                    "equipment_instances.allowed_regions",
                    "A workstation bay reference must resolve to exactly one accepted equipment instance.",
                    row["literal"],
                )
                continue
            instance = matches[0]
            instance_id = str(instance.get("instance_id") or "")
            if not instance_id or instance_id in seen_instances:
                binding_failed = True
                _unresolved(
                    result,
                    "duplicate_instance_reference",
                    "equipment_instances.allowed_regions",
                    "One accepted equipment instance cannot occupy more than one workstation bay.",
                    row["literal"],
                )
                continue
            seen_instances.add(instance_id)
            bindings[(row_index, column_index)] = instance
    if binding_failed:
        return result

    region_bounds = _rect_bounds(row_region)
    if region_bounds is None:
        _unresolved(
            result,
            "invalid_south_region",
            "site_ref.site_snapshot.zones",
            "The south-side placement region needs positive rectangular dimensions.",
            layout_match.group(0),
        )
        return result
    min_x, max_x, min_z, max_z = region_bounds
    min_x = _east_of_west_blockers(snapshot, min_x, max_x, min_z, max_z)
    total_width = max_x - min_x
    total_depth = max_z - min_z
    bay_width = total_width / bays_per_row
    row_depths = _row_depth_requirements(expected_indices, parsed_rows, bindings)
    if row_count == 1:
        row_depths[1] = total_depth
    required_depth = sum(row_depths.values())
    if required_depth > total_depth + 1e-9:
        _unresolved(
            result,
            "row_space_requirements_exceed_region",
            "site_ref.site_snapshot.zones",
            "The source-specified machine footprints and case clearances do not fit in the south-side area with the current aisle width.",
            layout_match.group(0),
        )
        return result
    inter_row_gap = (
        max(0.0, total_depth - required_depth) / (row_count - 1)
        if row_count > 1
        else 0.0
    )
    band_depths = dict(row_depths)
    if nearest_row is not None:
        # Keep the front row at the aisle edge and give it the available slack.
        # Its service envelope remains contained while the rear row stays fixed
        # against the south wall during later aisle-width repairs.
        band_depths[nearest_row] += max(0.0, total_depth - required_depth)
        inter_row_gap = 0.0
    row_z_bounds = {}
    row_cursor = min_z
    south_to_north = [index for index in expected_indices if index != nearest_row] + ([nearest_row] if nearest_row else [])
    for position, row_index in enumerate(south_to_north):
        depth = band_depths[row_index]
        if position == len(south_to_north) - 1 and row_count > 1:
            south = max_z - depth
        else:
            south = row_cursor
        north = south + depth
        row_z_bounds[row_index] = (south, north)
        row_cursor = north + inter_row_gap

    if total_width <= 0.0 or total_depth <= 0.0 or bay_width <= 0.0 or any(value <= 0.0 for value in row_depths.values()):
        _unresolved(
            result,
            "row_geometry_out_of_range",
            "site_ref.site_snapshot.zones",
            "The site and reserved geometry leave no usable area for the declared workstation rows.",
            layout_match.group(0),
        )
        return result

    frame_id = _frame_id(snapshot)
    order_literal = next(
        (
            match.group(0).strip()
            for match in _ROW_NEAREST_AISLE.finditer(text)
            if _ordinal_value(match.group("row")) == nearest_row
        ),
        "",
    )
    row_literal = " ".join(
        [*(parsed_rows[index]["literal"] for index in expected_indices), *([order_literal] if order_literal else [])]
    )
    result["literal_spans"]["organization"] = row_literal
    _remove_prior_bay_zones(snapshot)
    for row_index in expected_indices:
        for column_index, label in enumerate(parsed_rows[row_index]["items"], start=1):
            instance = bindings[(row_index, column_index)]
            bay_id = f"{_PREFIX}row_{row_index}_column_{column_index}"
            west = min_x + (column_index - 1) * bay_width
            east = west + bay_width
            south, north = row_z_bounds[row_index]
            bay_depth = north - south
            center_x = (west + east) / 2.0
            center_z = (south + north) / 2.0
            reserved = instance is None
            if not reserved and not _fits_instance(instance, bay_width, bay_depth):
                _unresolved(
                    result,
                    "equipment_does_not_fit_bay",
                    "equipment_instances.allowed_regions",
                    "The accepted model footprint does not fit its source-assigned bay in either allowed cardinal orientation.",
                    parsed_rows[row_index]["literal"],
                )
                continue

            _upsert_zone(
                snapshot,
                _zone(
                    bay_id,
                    "forbidden_zone" if reserved else "workstation_bay",
                    label,
                    center_x,
                    center_z,
                    bay_width,
                    bay_depth,
                    frame_id,
                ),
            )
            region_id = None
            assigned_ids = []
            if instance:
                region_id = bay_id + "/placement"
                _upsert_zone(
                    snapshot,
                    _zone(
                        region_id,
                        "placement_region",
                        "placement region for " + label,
                        center_x,
                        center_z,
                        bay_width,
                        bay_depth,
                        frame_id,
                    ),
                )
                assigned_ids = [str(instance["instance_id"])]
                result["instance_region_assignments"][str(instance["instance_id"])] = {
                    "region_id": region_id,
                    "bay_id": bay_id,
                    "source_text": parsed_rows[row_index]["literal"],
                }
            bay = {
                "bay_id": bay_id,
                "label": label,
                "row_index": row_index,
                "column_index": column_index,
                "reserved": reserved,
                "pose": _pose(center_x, center_z, frame_id),
                "dimensions_m": {"width_m": bay_width, "depth_m": bay_depth},
                "bounds_m": {
                    "min_x_m": west,
                    "max_x_m": east,
                    "min_z_m": south,
                    "max_z_m": north,
                },
                "assigned_instance_ids": assigned_ids,
                "source_text": parsed_rows[row_index]["literal"],
            }
            result["workstation_bays"].append(bay)
            result["provenance"].append(
                {
                    "field_path": f"site_ref.site_snapshot.zones[{bay_id}]",
                    "source_type": "user_input",
                    "source_id": intent_id,
                    "source_field": parsed_rows[row_index]["literal"],
                    "confidence": 1.0,
                    "note": "Bay bounds derived from the source row order and resolved south placement region.",
                }
            )
    if any(item.get("item_id") == "organized_workstations_equipment_does_not_fit_bay" for item in result["unresolved_items"]):
        return result
    if len(result["workstation_bays"]) != row_count * bays_per_row:
        return result
    result["constraint_requirements"].append(
        {"constraint_type": "region_membership", "source_text": row_literal}
    )
    if any(bay["reserved"] for bay in result["workstation_bays"]):
        result["constraint_requirements"].append(
            {"constraint_type": "forbidden_zone_avoidance", "source_text": row_literal}
        )
    return result


def _apply_maintenance_assumptions(text, semantic_intent, equipment_instances, result, intent_id):
    mentions = _resolved_mentions(semantic_intent)
    requirements = [
        {
            "target": match.group("target"),
            "front": match.group("front"),
            "rear": match.group("rear"),
            "side": match.group("side"),
            "literal": match.group(0).strip(),
        }
        for match in _CLEARANCE_BULLET.finditer(text)
    ]
    requirements.extend(
        {
            "target": match.group("target"),
            "front": match.group("front"),
            "rear": match.group("rear"),
            "side": match.group("side"),
            "literal": _sentence_for_span(text, match.start(), match.end()),
        }
        for match in _MAINTENANCE.finditer(text)
    )
    for requirement in requirements:
        candidates = _resolve_clearance_targets(
            requirement["target"], text, mentions, equipment_instances
        )
        if not candidates:
            _unresolved(
                result,
                "maintenance_target",
                "equipment_instances.maintenance_clearance_m",
                "The case space requirement must resolve to one or more accepted equipment instances.",
                requirement["literal"],
            )
            continue
        value = {
            "front_m": float(requirement["front"]),
            "rear_m": float(requirement["rear"]),
            "left_m": float(requirement["side"]),
            "right_m": float(requirement["side"]),
            "top_m": None,
            "bottom_m": None,
            "source": "user_input",
            "confidence": 1.0,
        }
        for instance in candidates:
            instance["maintenance_clearance_m"] = copy.deepcopy(value)
            # The accepted case requirements supersede older experimental
            # maintenance and safety estimates on the working instance only.
            instance["safety_clearance_m"] = copy.deepcopy(value)
            result["literal_spans"]["maintenance"].append(requirement["literal"])
            for field_name in ("maintenance_clearance_m", "safety_clearance_m"):
                result["provenance"].append(
                    {
                        "field_path": f"equipment_instances[{instance.get('instance_id')}].{field_name}",
                        "source_type": "user_input",
                        "source_id": intent_id,
                        "source_field": requirement["literal"],
                        "confidence": 1.0,
                        "note": "User-confirmed case requirement; not manufacturer clearance data. It supersedes legacy experimental estimates on this working instance only.",
                    }
                )


def _resolve_clearance_targets(target, source_text, mentions, equipment_instances):
    target_key = _token_key(target)
    selected = []
    exact = _resolve_item(target, source_text, mentions, equipment_instances)
    if "ordinary" in target_key and "machining" in target_key:
        selected.extend(
            instance
            for instance in equipment_instances or []
            if _ordinary_machining_role(instance, mentions)
        )
    elif re.search(r"\bincluding\b", target, flags=re.IGNORECASE) or len(_category_terms(target)) > 1:
        categories = _category_terms(target.split("including", 1)[-1])
        for category in categories:
            selected.extend(_instances_for_category(category, mentions, equipment_instances))
    else:
        if exact:
            return exact
        return _instances_for_category(target, mentions, equipment_instances)
    unique = {}
    for instance in selected:
        instance_id = str(instance.get("instance_id") or "")
        if instance_id:
            unique[instance_id] = instance
    return [unique[key] for key in sorted(unique)]


def _ordinary_machining_role(instance, mentions):
    values = [str(instance.get("role") or "")]
    model_id = str(instance.get("model_asset_id") or "")
    values.extend(
        str(mention.get("surface_text") or "")
        for mention in mentions
        if str(mention.get("model_asset_id") or "") == model_id
    )
    key = " ".join(_token_key(value) for value in values)
    if "turn mill" in key or "turn-mill" in key:
        return False
    return any(token in key.split() for token in ("cnc", "vertical", "radial", "edm"))


def _category_terms(value):
    value = re.sub(r"\b(?:the|including)\b", " ", str(value or ""), flags=re.IGNORECASE)
    return [part.strip(" ,;.") for part in re.split(r",|\band\b", value, flags=re.IGNORECASE) if part.strip(" ,;.")]


def _instances_for_category(category, mentions, equipment_instances):
    category_tokens = set(_role_tokens(category)) - _STOP
    if not category_tokens:
        return []
    if "rack" in category_tokens:
        category_tokens = {"rack"}
    elif "cnc" in category_tokens:
        category_tokens = {"cnc"}
    if "assembly" in category_tokens:
        accepted = {"fixture", "assembly"}
    elif "test" in category_tokens or "testing" in category_tokens:
        accepted = {"test", "testing"}
    else:
        accepted = category_tokens
    result = []
    for instance in equipment_instances or []:
        values = [str(instance.get("role") or "")]
        model_id = str(instance.get("model_asset_id") or "")
        values.extend(
            str(mention.get("surface_text") or "")
            for mention in mentions
            if str(mention.get("model_asset_id") or "") == model_id
        )
        tokens = set()
        for value in values:
            tokens.update(_role_tokens(value))
        tokens -= _STOP
        if any(token in tokens for token in accepted) if accepted in ({"fixture", "assembly"}, {"test", "testing"}) else category_tokens.issubset(tokens):
            result.append(instance)
    return result


def _resolved_mentions(semantic_intent):
    return [
        item
        for item in (semantic_intent or {}).get("asset_mentions") or []
        if isinstance(item, dict)
        and str(item.get("surface_text") or "").strip()
        and str(item.get("model_asset_id") or "").strip()
    ]


def _resolve_item(item_text, source_text, mentions, equipment_instances):
    """Return an exact resolved mention/quantity binding for one source label."""

    candidates = []
    instances_by_model = {}
    for instance in equipment_instances or []:
        model_id = str(instance.get("model_asset_id") or "")
        if model_id:
            instances_by_model.setdefault(model_id, []).append(instance)
    for values in instances_by_model.values():
        values.sort(key=lambda item: (_positive_int(item.get("quantity_index")) or math.inf, str(item.get("instance_id") or "")))

    cursor_by_model = {}
    for mention in mentions:
        surface = str(mention.get("surface_text") or "").strip()
        model_id = str(mention.get("model_asset_id") or "").strip()
        if not _literal_present(source_text, surface):
            continue
        quantity = _positive_int(mention.get("quantity")) or 1
        indices = instances_by_model.get(model_id) or []
        start = cursor_by_model.get(model_id, 0)
        selected = indices[start : start + quantity]
        if len(selected) < quantity:
            continue
        cursor_by_model[model_id] = start + quantity
        labels = declaration_labels(surface, quantity)
        base_tokens = _role_tokens(surface, labels)
        target_tokens = _tokens(item_text)
        if quantity > 1:
            label_positions = [index for index, label in enumerate(labels) if _has_label(item_text, label)]
            if len(label_positions) != 1:
                continue
            label_position = label_positions[0]
            target_role = _role_tokens(item_text, (labels[label_position],))
            if not _roles_match(base_tokens, target_role):
                continue
            candidates.append(selected[label_position])
            continue
        mention_tokens = _tokens(surface)
        if not _roles_match(mention_tokens, target_tokens):
            continue
        # A single resolved declaration is safe only if this label names the
        # same literal mention, or the catalog has one matching unit in total.
        peer_mentions = [
            other
            for other in mentions
            if str(other.get("model_asset_id") or "") == model_id
            and _literal_present(source_text, str(other.get("surface_text") or ""))
        ]
        if len(peer_mentions) > 1 and not _tokens(item_text).issubset(mention_tokens):
            continue
        candidates.append(selected[0])

    unique = {}
    for item in candidates:
        key = str(item.get("instance_id") or "")
        if key:
            unique[key] = item

    if not unique:
        # Some existing-layout IRs carry stable user roles even when the
        # current semantic frame omits the original declaration text.
        for instance in equipment_instances or []:
            role = str(instance.get("role") or "").strip()
            if role and _instance_role_matches(role, item_text):
                unique[str(instance.get("instance_id") or "")] = instance
    return [item for key, item in unique.items() if key]


def _instance_role_matches(role, reference):
    if _token_key(role) == _token_key(reference):
        return True
    role_tokens = _tokens(role)
    reference_tokens = _tokens(reference)
    if (
        role_tokens
        and reference_tokens
        and len(role_tokens[-1]) == 1
        and role_tokens[-1].isalpha()
        and len(reference_tokens[-1]) == 1
        and reference_tokens[-1].isalpha()
        and role_tokens[-1] != reference_tokens[-1]
    ):
        return False
    return _roles_match(role_tokens, reference_tokens)


def _roles_match(left, right):
    left = set(left) - _STOP
    right = set(right) - _STOP
    if not left or not right:
        return False
    overlap = left & right
    return bool(overlap) and (left.issubset(right) or right.issubset(left) or len(overlap) >= 2)


def _role_tokens(value, labels=()):
    tokens = _tokens(value)
    label_tokens = {str(item).casefold() for item in labels}
    tokens = [token for token in tokens if token not in label_tokens]
    while tokens and tokens[0] in {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}:
        tokens.pop(0)
    tokens = [token[:-1] if len(token) > 4 and token.endswith("s") else token for token in tokens]
    return tokens


def _tokens(value):
    tokens = [token.casefold() for token in _TOKEN.findall(str(value or ""))]
    expanded = []
    for token in tokens:
        if token == "vmc":
            expanded.extend(("vertical", "machining", "center"))
        elif token == "edm":
            expanded.append("edm")
        else:
            expanded.append(token)
    return expanded


def _has_label(value, label):
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(str(label))}(?![a-z0-9])", str(value or ""), re.IGNORECASE))


def _literal_present(text, value):
    value = str(value or "").strip()
    if not value:
        return False
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(value).replace(r"\ ", r"\s+") + r"(?![a-z0-9])", str(text or ""), re.IGNORECASE))


def _south_placement_region(snapshot):
    zones = [
        item
        for item in snapshot.get("zones") or []
        if isinstance(item, dict)
        and str(item.get("zone_type") or "").casefold() in {"placement", "placement_region", "equipment_region"}
    ]
    named_south = [
        item for item in zones
        if "south" in (str(item.get("zone_id") or "") + " " + str(item.get("zone_name") or "")).casefold()
    ]
    candidates = named_south or zones
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(((item.get("pose") or {}).get("z_m") or 0.0)))


def _east_of_west_blockers(snapshot, min_x, max_x, min_z, max_z):
    start = min_x
    obstacles = []
    obstacles.extend(snapshot.get("site_features") or [])
    obstacles.extend(
        item for item in (snapshot.get("zones") or [])
        if str(item.get("zone_type") or "").casefold() in {"wall", "obstacle", "forbidden", "forbidden_zone", "non_placeable"}
    )
    for item in obstacles:
        bounds = _rect_bounds(item)
        if bounds is None:
            continue
        left, right, south, north = bounds
        source_name = " ".join(
            str(item.get(key) or "")
            for key in ("feature_name", "zone_name", "site_feature_id", "zone_id")
        ).casefold()
        touches_west_edge = left <= min_x + 1e-9
        is_west_buffer = "buffer" in source_name and (left + right) / 2.0 < (min_x + max_x) / 2.0
        if (
            str(item.get("feature_type") or item.get("zone_type") or "").casefold()
            not in {"wall", "obstacle", "forbidden", "forbidden_zone", "non_placeable", "column"}
            and not item.get("must_remain_clear")
        ):
            continue
        if (touches_west_edge or is_west_buffer) and right > start and south < max_z and north > min_z:
            start = min(max_x, right)
    return start


def _fits_instance(instance, bay_width, bay_depth):
    dimensions = ((instance.get("geometry") or {}).get("dimensions_m") or {})
    try:
        width = float(dimensions["width_m"])
        depth = float(dimensions["depth_m"])
    except (KeyError, TypeError, ValueError):
        return False
    if width <= 0.0 or depth <= 0.0:
        return False
    clearance = _case_clearance(instance)
    left = clearance["left_m"]
    right = clearance["right_m"]
    front = clearance["front_m"]
    rear = clearance["rear_m"]
    allowed = instance.get("allowed_orientations_deg") or [0, 90, 180, 270]
    for angle in allowed:
        try:
            angle = float(angle) % 360.0
        except (TypeError, ValueError):
            continue
        if angle % 180.0 == 90.0:
            placed_width, placed_depth = depth + front + rear, width + left + right
        else:
            placed_width, placed_depth = width + left + right, depth + front + rear
        if placed_width <= bay_width + 1e-9 and placed_depth <= bay_depth + 1e-9:
            return True
    return False


def _case_clearance(instance):
    value = instance.get("maintenance_clearance_m") or {}
    if value.get("source") != "user_input":
        return {key: 0.0 for key in ("front_m", "rear_m", "left_m", "right_m")}
    return {
        key: max(0.0, float(value.get(key) or 0.0))
        for key in ("front_m", "rear_m", "left_m", "right_m")
    }


def _row_depth_requirements(row_indices, parsed_rows, bindings):
    result = {}
    for row_index in row_indices:
        requirements = []
        for column_index, _label in enumerate(parsed_rows[row_index]["items"], start=1):
            instance = bindings[(row_index, column_index)]
            if instance is None:
                continue
            dimensions = ((instance.get("geometry") or {}).get("dimensions_m") or {})
            try:
                depth = float(dimensions["depth_m"])
            except (KeyError, TypeError, ValueError):
                continue
            clearance = _case_clearance(instance)
            requirements.append(depth + clearance["front_m"] + clearance["rear_m"])
        result[row_index] = max(requirements or [0.0])
    return result


def _remove_prior_bay_zones(snapshot):
    snapshot["zones"] = [
        item for item in (snapshot.get("zones") or [])
        if not str(item.get("zone_id") or "").startswith(_PREFIX)
    ]


def _zone(zone_id, zone_type, name, center_x, center_z, width, depth, frame_id):
    return {
        "zone_id": zone_id,
        "zone_type": zone_type,
        "zone_name": name,
        "pose": _pose(center_x, center_z, frame_id),
        "dimensions_m": {"width_m": width, "depth_m": depth, "height_m": 0.1},
        "allowed_categories": [],
        "forbidden_categories": [],
        "source": "computed",
        "source_model_asset_id": None,
        "source_node_ref": None,
        "confidence": 1.0,
    }


def _pose(x_m, z_m, frame_id):
    return {
        "x_m": x_m,
        "y_m": 0.0,
        "z_m": z_m,
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "yaw_deg": 0.0,
        "coordinate_frame_id": frame_id,
    }


def _rect_bounds(item):
    try:
        pose = item.get("pose") or {}
        dimensions = item.get("dimensions_m") or {}
        center_x = float(pose["x_m"])
        center_z = float(pose["z_m"])
        width = float(dimensions["width_m"])
        depth = float(dimensions["depth_m"])
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if width <= 0.0 or depth <= 0.0:
        return None
    yaw = float((item.get("pose") or {}).get("yaw_deg") or 0.0) % 180.0
    if yaw == 90.0:
        width, depth = depth, width
    return center_x - width / 2.0, center_x + width / 2.0, center_z - depth / 2.0, center_z + depth / 2.0


def _frame_id(snapshot):
    boundary = snapshot.get("boundary") or {}
    return ((boundary.get("pose") or {}).get("coordinate_frame_id")) or "site_layout_frame"


def _upsert_zone(snapshot, zone):
    zones = snapshot.setdefault("zones", [])
    for index, existing in enumerate(zones):
        if existing.get("zone_id") == zone["zone_id"]:
            zones[index] = zone
            return
    zones.append(zone)


def _split_row_items(value):
    content = str(value or "").strip().rstrip(".")
    if ";" in content or "|" in content:
        parts = re.split(r"\s*(?:;|\|)\s*", content)
    else:
        parts = re.split(r",\s*(?:and\s+)?", re.sub(r",\s+and\s+", ", ", content, flags=re.IGNORECASE))
    return [re.sub(r"^(?:and\s+)", "", item.strip(), flags=re.IGNORECASE) for item in parts if item.strip()]


def _row_index(match):
    value = (match.group("ordinal") or match.group("word") or "").casefold()
    words = value.split()
    token = words[-1] if words else ""
    if token.isdigit():
        return int(token)
    return _ORDINALS.get(token, 0)


def _ordinal_value(value):
    token = str(value or "").casefold()
    return int(token) if token.isdigit() else _ORDINALS.get(token)


def _count(value):
    value = str(value or "").casefold()
    return int(value) if value.isdigit() else _COUNTS.get(value)


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _is_reserved(value):
    return bool(re.search(r"\b(?:reserved|empty|vacant)\s+(?:workstation\s+)?bay\b", str(value or ""), re.IGNORECASE))


def _token_key(value):
    return " ".join(_tokens(value))


def _sentence_for_span(text, start, end):
    left = max(text.rfind(".", 0, start), text.rfind("!", 0, start), text.rfind("?", 0, start)) + 1
    stops = [position for marker in (".", "!", "?") if (position := text.find(marker, end)) >= 0]
    right = min(stops) + 1 if stops else len(text)
    return text[left:right].strip()


def _unresolved(result, item, field_path, description, source_text):
    record = {
        "item_id": "organized_workstations_" + item,
        "item_type": "missing_field",
        "field_path": field_path,
        "description": description,
        "severity": "blocking",
        "candidates": [],
        "resolution_policy": "ask_user",
    }
    if source_text:
        record["source_text"] = str(source_text)
    if not any(existing.get("item_id") == record["item_id"] for existing in result["unresolved_items"]):
        result["unresolved_items"].append(record)


__all__ = ["compile_organized_workstations"]
