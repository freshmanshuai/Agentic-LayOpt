"""Prefer visually identified operator fronts toward an aisle when feasible.

This is a layout preference, not an OEM interface or a new hard constraint.
Unknown fronts and equipment with fixed orientations remain unchanged.
"""
import copy
import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def model_front_annotations():
    path = Path(__file__).resolve().parents[2] / "rag_dataset/model_front_annotations.json"
    return json.loads(path.read_text(encoding="utf-8"))["models"]


def model_front_vector(model_asset_id, *, annotations=None):
    models = annotations if annotations is not None else model_front_annotations()
    entry = (models or {}).get(str(model_asset_id or "")) or {}
    vectors = entry.get("front_vectors_xz") or []
    if not vectors:
        return None
    try:
        x, z = float(vectors[0][0]), float(vectors[0][1])
    except (IndexError, TypeError, ValueError):
        return None
    length = math.hypot(x, z)
    return (x / length, z / length) if length > 1e-9 else None


def preferred_yaw_for_direction(model_asset_id, direction_x=0.0, direction_z=1.0, *, allowed_orientations=None, annotations=None):
    """Return the cardinal pose that points an audited front toward a direction."""

    models = annotations if annotations is not None else model_front_annotations()
    entry = (models or {}).get(str(model_asset_id or "")) or {}
    vectors = entry.get("front_vectors_xz") or []
    if not vectors:
        return None
    choices = [float(value) % 360.0 for value in (allowed_orientations or [0, 90, 180, 270])]
    choices = [value for value in choices if value % 90.0 == 0.0]
    if not choices:
        return None
    target_length = math.hypot(float(direction_x), float(direction_z))
    if target_length <= 1e-9:
        return None
    target_x, target_z = float(direction_x) / target_length, float(direction_z) / target_length

    def score(yaw):
        angle = math.radians(yaw)
        cosine, sine = math.cos(angle), math.sin(angle)
        return max(
            (
                (cosine * float(vector[0]) - sine * float(vector[1])) * target_x
                + (sine * float(vector[0]) + cosine * float(vector[1])) * target_z
            ) / max(math.hypot(float(vector[0]), float(vector[1])), 1e-9)
            for vector in vectors
        )

    scores = {yaw: score(yaw) for yaw in choices}
    best = max(scores.values())
    return min((yaw for yaw in choices if scores[yaw] >= best - 1e-9), key=lambda value: (value,))


def case_clearance_margins(clearance, front_vector_xz, yaw_deg):
    """Map front/rear/left/right allowances onto world x/z sides."""

    clearance = clearance or {}
    front = max(0.0, float(clearance.get("front_m") or 0.0))
    rear = max(0.0, float(clearance.get("rear_m") or 0.0))
    left = max(0.0, float(clearance.get("left_m") or 0.0))
    right = max(0.0, float(clearance.get("right_m") or 0.0))
    vector = front_vector_xz or (0.0, 1.0)
    fx, fz = float(vector[0]), float(vector[1])
    length = math.hypot(fx, fz)
    if length <= 1e-9:
        fx, fz = 0.0, 1.0
    else:
        fx, fz = fx / length, fz / length
    directions = {
        "front": (fx, fz, front),
        "rear": (-fx, -fz, rear),
        "right": (fz, -fx, right),
        "left": (-fz, fx, left),
    }
    angle = math.radians(float(yaw_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    margins = {"left_m": 0.0, "right_m": 0.0, "south_m": 0.0, "north_m": 0.0}
    for dx, dz, value in directions.values():
        world_x = cosine * dx - sine * dz
        world_z = sine * dx + cosine * dz
        if abs(world_x) >= abs(world_z):
            key = "left_m" if world_x < 0 else "right_m"
        else:
            key = "south_m" if world_z < 0 else "north_m"
        margins[key] = max(margins[key], value)
    return margins


def improve_aisle_facing(layout, problem, *, annotations=None, validator=None, objectives=None):
    from tool.layout.layout_solvers import layout_violations, _placement, _oriented_dimensions
    if validator is None:
        validator = layout_violations
    if annotations is None:
        annotations = model_front_annotations()
    result = copy.deepcopy(layout)
    report = dict(basis="Visually inspected calibrated operator fronts; feasible orientation preference", changes=[], skipped=[])
    if validator(result, problem):
        report["skipped"].append(dict(reason="Input layout is infeasible; no cosmetic rotation can substitute for repair."))
        return result, report
    corridors = [f for f in result.get("site",{}).get("site_features",[]) if f.get("feature_type") in {"corridor","aisle"} and f.get("pose") and f.get("dimensions_m")]
    if not corridors:
        return result, report
    instances = {i["instance_id"]:i for i in problem.get("equipment_instances",[])}
    locked = set((problem.get("optimization_scope") or {}).get("locked_instances") or [])
    for constraint in (problem.get("constraints") or {}).get("hard") or []:
        if constraint.get("constraint_type") in {"fixed_pose","fixed_position","orientation_lock","transfer_workcell_alignment"}:
            locked.update(r.get("target_id") for r in constraint.get("target_refs") or [])
    for index,placement in enumerate(result["placements"]):
        key=placement["instance_id"]; instance=instances.get(key)
        front=annotations.get(placement["model_asset_id"])
        if not instance or not front or key in locked:
            continue
        x,z=placement["position"]["x_m"],placement["position"]["z_m"]
        directions=[]
        for corridor in corridors:
            center=corridor["pose"]; size=corridor["dimensions_m"]
            dx=max(center["x_m"]-size["width_m"]/2,min(x,center["x_m"]+size["width_m"]/2))-x
            dz=max(center["z_m"]-size["depth_m"]/2,min(z,center["z_m"]+size["depth_m"]/2))-z
            directions.append((math.hypot(dx,dz),dx,dz))
        length,dx,dz=min(directions)
        if length<1e-9:
            continue
        def score(yaw):
            c,s=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
            return max(((c*fx-s*fz)*dx+(s*fx+c*fz)*dz)/length/math.hypot(fx,fz)
                       for fx,fz in front["front_vectors_xz"])
        old=float(placement.get("rotation",{}).get("yaw_deg",0))%360
        allowed=instance.get("allowed_orientations_deg") or [0,90,180,270]
        # Only cardinal rotations have exact rectangular envelopes here.
        candidates=[float(a)%360 for a in allowed if float(a)%90==0 and score(a)>score(old)+1e-8]
        candidates.sort(key=lambda a:(-score(a),min((a-old)%360,(old-a)%360)))
        accepted=False
        for yaw in candidates:
            w,d,h=_oriented_dimensions(instance,yaw)
            posed=_placement(instance,w,d,h,x,z,yaw_deg=yaw)
            candidate=copy.deepcopy(result)
            candidate["placements"][index].update({field:posed[field] for field in ("rotation","yaw_deg","dimensions_m","footprint")})
            feasible = not validator(candidate,problem)
            if feasible and objectives is not None:
                before,after=objectives(result),objectives(candidate)
                feasible=all(after[k]<=value+1e-8 for k,value in before.items())
            if feasible:
                result=candidate;accepted=True
                report["changes"].append(dict(instance_id=key,from_yaw_deg=old,to_yaw_deg=yaw,from_alignment=score(old),to_alignment=score(yaw)))
                break
        if candidates and not accepted:
            report["skipped"].append(dict(instance_id=key,reason="Better-facing cardinal poses failed existing feasibility or objective-preservation checks."))
    return result,report
