"""Facility areas remain spatial flow endpoints, separate from catalog equipment."""
import copy
import re


def existing_facility_surface(surface, features):
    """Recognize collective references to known floor features, not hardware."""
    features = features or []
    if not features:
        return False
    words = {w[:-1] if len(w)>3 and w.endswith("s") else w for w in re.findall(r"[a-z]+", str(surface).lower())}
    words -= {"the","a","an","one","two","both","existing","fixed","and","or"}
    types = {f.get("feature_type") for f in features}
    if words and words <= {"receiving","dispatch","shipping","point","door","gate","entrance","exit","area","location"}:
        receiving = bool(words & {"receiving","entrance"})
        dispatch = bool(words & {"dispatch","shipping","exit"})
        return (receiving or dispatch) and (not receiving or "entrance" in types) and (not dispatch or "exit" in types)
    if "buffer" in words and words <= {"buffer","inbound","outbound","area","zone","floor"}:
        return "buffer_area" in types or any(facility_role(f.get("feature_name")) in {"inbound buffer","outbound buffer"} for f in features)
    if words & {"aisle","corridor"} and words <= {"aisle","corridor","central","logistic","main","lane"}:
        return "corridor" in types
    return False


def facility_role(surface):
    text = " ".join(re.findall(r"[a-z]+", str(surface).lower()))
    if "buffer station" in text:
        return None
    if re.search(r"\binbound buffer\b", text):
        return "inbound buffer"
    if re.search(r"\boutbound buffer\b", text):
        return "outbound buffer"
    if text in {"receiving", "receiving door", "receiving gate", "entry", "entrance"}:
        return "receiving"
    if text in {"dispatch", "dispatch door", "dispatch gate", "exit", "shipping"}:
        return "dispatch"
    return None


def declared_facility_roles(text):
    roles = set()
    for role in ["inbound buffer", "outbound buffer"]:
        if re.search(r"\b" + role + r"\b", str(text), re.I):
            roles.add(role)
    if re.search(r"\breceiving\s+(?:door|gate)\b", str(text), re.I):
        roles.add("receiving")
    if re.search(r"\bdispatch\s+(?:door|gate)\b", str(text), re.I):
        roles.add("dispatch")
    return roles


def resolve_facility_ref(features, name):
    role = facility_role(name)
    matches = []
    for feature in features or []:
        label = feature.get("feature_name") or feature.get("name") or ""
        kind = feature.get("feature_type")
        identified = facility_role(label)
        if not identified:
            identified = "receiving" if kind == "entrance" else "dispatch" if kind == "exit" else None
        if role and role == identified:
            matches.append(feature.get("site_feature_id"))
    return matches[0] if len(matches) == 1 else None


def fixed_flow_endpoints(layout, flp_ir):
    snapshot = ((flp_ir or {}).get("site_ref") or {}).get("site_snapshot") or {}
    features = ((layout or {}).get("site") or {}).get("site_features") or snapshot.get("site_features") or []
    return {f["site_feature_id"]: dict(position=copy.deepcopy(f["pose"])) for f in features
            if f.get("site_feature_id") and f.get("pose") and f.get("feature_type") in {"entrance", "exit", "buffer_area", "forbidden_zone"} and (f.get("feature_type") != "forbidden_zone" or facility_role(f.get("feature_name")))}


def derive_aligned_doors(snapshot, text):
    """Door center and width follow the explicitly aligned central aisle."""
    source = str(text or "")
    if not re.search(r"\balign\w*\b[^.]*\b(?:doors?|aisle)\b", source, re.I):
        return
    corridors = [f for f in snapshot.get("site_features") or [] if f.get("feature_type") == "corridor"]
    if len(corridors) != 1:
        return
    boundary = snapshot["boundary"];size = boundary["dimensions_m"];center = boundary.get("pose") or {}
    aisle = corridors[0];opening = aisle["dimensions_m"]["depth_m"]
    for label, kind, direction, sign in [("receiving", "entrance", "west", -1), ("dispatch", "exit", "east", 1)]:
        if not re.search(rf"\b{label}\s+(?:door|gate)\b[^.]*?\b{direction}\s+wall\b", source, re.I):
            continue
        feature_id = "natural_site/" + label + "_door"
        feature = dict(site_feature_id=feature_id,feature_type=kind,feature_name=label,
                       pose=dict(x_m=float(center.get("x_m", 0))+sign*float(size["width_m"])/2,
                                 y_m=0,z_m=aisle["pose"]["z_m"],roll_deg=0,pitch_deg=0,yaw_deg=0,
                                 coordinate_frame_id=boundary["pose"].get("coordinate_frame_id", "site_frame")),
                       dimensions_m=dict(width_m=.2,depth_m=opening,height_m=3.6),must_remain_clear=True,
                       source="computed",confidence=1.0)
        existing = next((f for f in snapshot["site_features"] if f.get("site_feature_id") == feature_id), None)
        if existing is None:
            snapshot["site_features"].append(feature)
