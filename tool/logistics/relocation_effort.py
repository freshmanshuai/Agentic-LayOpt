"""User-supplied relocation effort, measured against the accepted initial layout."""
import math
import re

RELOCATION_EFFORT_OBJECTIVE = "minimize_relocation_effort"


def relocation_requested(text):
    return bool(re.search(r"\b(?:relocation|reconfiguration)\s+(?:effort|cost)\b", str(text), re.I))


def _words(text):
    return {w[:-1] if w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text).lower())}


def compile_relocation_parameters(text, context, asset_facts):
    """Compile explicitly written cost rows; never infer coefficients from CAD."""
    number = r"(?:\d+(?:\.\d*)?|\.\d+)"
    pattern = rf"^\s*[-*]?\s*([^:\n]+):\s*fixed effort\s+({number})\s*;\s*distance effort\s+({number})\s+per meter\b"
    rules = [(label.strip(), float(fixed), float(distance)) for label, fixed, distance in re.findall(pattern, text, re.I | re.M)]
    if not rules:
        raise ValueError("Please supply fixed and per-meter relocation-effort coefficients for the equipment categories.")
    reference = context.get("initial_layout") or (context.get("authoritative_context") or {}).get("initial_layout")
    if not reference or not reference.get("placements"):
        raise ValueError("Relocation effort needs an accepted initial layout.")
    facts = {a["model_asset_id"]: a for a in asset_facts}
    coefficients = {}
    for placement in reference["placements"]:
        fact = facts.get(placement["model_asset_id"], {})
        names = " ".join(str(fact.get(k) or "") for k in ["canonical_name", "equipment_type", "category"])
        tokens = _words(names)
        candidates = [(len(_words(label)), label, fixed, distance) for label, fixed, distance in rules if _words(label) <= tokens]
        candidates.sort(reverse=True)
        if not candidates or len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise ValueError(f"Please clarify which relocation-effort category applies to {fact.get('canonical_name') or 'this equipment'}.")
        _, label, fixed, distance = candidates[0]
        coefficients[placement["instance_id"]] = dict(fixed_effort=fixed, effort_per_meter=distance, category=label)
    return dict(reference_layout_id=reference.get("layout_id"),
                reference_positions={p["instance_id"]: dict(p["position"]) for p in reference["placements"]},
                coefficients=coefficients, distance_metric="manhattan_xz", units="planning_effort_index",
                basis="Explicit user planning assumptions; not actual monetary charges",
                reference_source="accepted_initial_layout")


def validate_relocation_parameters(parameters, instance_ids):
    ids = set(instance_ids)
    positions = parameters.get("reference_positions") or {}
    coefficients = parameters.get("coefficients") or {}
    if set(positions) != ids or set(coefficients) != ids:
        raise ValueError("Relocation reference and coefficients must cover the existing equipment exactly.")
    for key in ids:
        if any(not math.isfinite(float(positions[key][axis])) for axis in ["x_m", "z_m"]):
            raise ValueError("Relocation reference coordinates must be finite.")
        for field in ["fixed_effort", "effort_per_meter"]:
            value = float(coefficients[key][field])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Relocation coefficients must be finite and nonnegative.")


def relocation_effort(layout, parameters):
    """Position changes only; orientation and AGV mission travel are not charged."""
    placements = layout.get("placements") or []
    validate_relocation_parameters(parameters, [p["instance_id"] for p in placements])
    total = 0.0
    for item in placements:
        key = item["instance_id"]
        old = parameters["reference_positions"][key]
        delta = sum(abs(float(item["position"][axis]) - float(old[axis])) for axis in ["x_m", "z_m"])
        if delta > 1e-8:
            coefficient = parameters["coefficients"][key]
            total += coefficient["fixed_effort"] + coefficient["effort_per_meter"] * delta
    return total
