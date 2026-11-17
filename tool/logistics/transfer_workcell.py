"""Compile a manual conveyor transfer cell from grounded material-flow roles.

No catalog ports are invented. End alignment and the operator strip are an
explicitly accepted planning proposal, distinct from the source CAD geometry.
"""
from __future__ import annotations
import copy
import math
import re


_FUNCTIONAL_GENERIC_ROLE_WORDS = {
    "a", "an", "the", "one", "two", "three", "four", "five", "and",
    "work", "workstation", "workstations", "station", "stations",
    "machine", "machines", "equipment", "assembly", "flexible", "small",
    "storage", "unit", "units",
}


def interface_questions(text):
    source = str(text)
    if not re.search(r"\bconveyor\b", source, re.I) or not all(re.search(rf"\b{role}\s+table\b", source, re.I) for role in ("loading", "unloading")):
        return []
    if not re.search(r"\bmanual\b|\boperators?\b", source, re.I):
        return ["Should operators lift containers between the tables and the conveyor, or is automatic level transfer required? The retrieved geometry alone does not establish a working automatic interface."]
    if not re.search(r"(?:opposite|input|infeed|upstream)[^.\n]{0,100}\bends?\b|\bat (?:the )?(?:input|infeed|upstream) end\b", source, re.I):
        return ["May I place the loading and unloading tables at opposite short ends of the conveyor, align their centers with it, and reserve a continuous operator strip on the aisle side? For manual lifting I propose a 0.20 m end gap and a 1.0 m operator strip as case planning assumptions, not manufacturer specifications. Please also confirm that the cell should run parallel to the central aisle."]
    return []


def compile_transfer_cell(problem, text):
    if not re.search(r"\bmanual\b|\boperators?\b", str(text), re.I):
        return
    if not re.search(r"opposite short ends|input end|infeed end|upstream end", str(text), re.I):
        return
    instances = problem.get("equipment_instances") or []
    by_id = {p["instance_id"]:p for p in instances}
    def role(p):
        return str(p.get("role") or "").casefold()
    conveyors = [p for p in instances if "conveyor" in role(p)]
    # Grounded flow relations can establish a conveyor when its role is a CAD name.
    for candidate in instances:
        if "conveyor" in str(candidate.get("model_asset_id","")).lower() and candidate not in conveyors:
            conveyors.append(candidate)
    for conveyor in conveyors:
        key = conveyor["instance_id"]
        before = {f["from_instance_id"] for f in problem.get("flow_relations") or [] if f.get("to_instance_id")==key}
        after = {f["to_instance_id"] for f in problem.get("flow_relations") or [] if f.get("from_instance_id")==key}
        inputs = [i for i in before if i in by_id and "table" in role(by_id[i])]
        outputs = [i for i in after if i in by_id and "table" in role(by_id[i])]
        if len(inputs)!=1 or len(outputs)!=1:
            continue
        parameters = dict(planning_kind="manual_conveyor_cell", conveyor_instance_id=key,
            input_table_instance_id=inputs[0],output_table_instance_id=outputs[0],
            end_gap_m=.20,operator_strip_m=1.,
            basis="Agent planning proposal accepted in the user clarification; manual lifting, no OEM docking claim",
            source_text=str(text))
        if re.search(r"parallel to (?:the )?central aisle",str(text),re.I):
            for member in (key,inputs[0],outputs[0]):
                by_id[member]["allowed_orientations_deg"]=[0.]
        constraints = problem.setdefault("constraints",{}).setdefault("hard",[])
        if not any((c.get("parameters") or {}).get("planning_kind")=="manual_conveyor_cell" for c in constraints):
            constraints.append(dict(constraint_id=f"constraint_hard_{len(constraints)+1:03d}",constraint_type="alignment",
                target_refs=[dict(target_type="equipment_instance",target_id=i) for i in (key,inputs[0],outputs[0])],
                parameters=parameters,source="user_input",evidence_refs=["source_request.raw_user_message"]))


def compile_functional_subregions(problem, text):
    """Resolve explicit quadrant role lists against grounded instance aliases."""
    snapshot=(problem.get("site_ref") or {}).get("site_snapshot") or {}
    size=(snapshot.get("boundary") or {}).get("dimensions_m") or {}
    if not size: return
    width,depth=float(size["width_m"]),float(size["depth_m"])
    aisles=[f for f in snapshot.get("site_features") or [] if f.get("feature_type") in {"corridor","aisle"}]
    north_edge=max((float((f.get("pose") or {}).get("z_m",0))+float(f["dimensions_m"]["depth_m"])/2 for f in aisles),default=0.)
    for match in re.finditer(r"(?im)^\s*(Northwest|Northeast)\s*:\s*([^\n]+)", str(text)):
        direction,literal=match.group(1).lower(),match.group(2)
        key="functional/"+direction
        a,b=(-width/2,0.) if direction=="northwest" else (0.,width/2)
        region=copy.deepcopy(next(iter(snapshot.get("zones") or []),{}))
        region.update(zone_id=key,zone_type="placement_region",zone_name="Assembly and transfer" if direction=="northwest" else "Test and finishing",
            pose=dict(x_m=(a+b)/2,y_m=0.,z_m=(north_edge+depth/2)/2,roll_deg=0.,pitch_deg=0.,yaw_deg=0.,coordinate_frame_id=None),
            dimensions_m=dict(width_m=b-a,depth_m=depth/2-north_edge,height_m=.1),
            allowed_categories=[],forbidden_categories=[],source="computed",source_model_asset_id=None,source_node_ref=None,confidence=1.)
        zones=snapshot.setdefault("zones",[])
        zones[:]=[z for z in zones if z.get("zone_id")!=key];zones.append(region)
        for item in problem.get("equipment_instances") or []:
            label=str(item.get("role") or "").strip()
            if label and _functional_role_listed(label, literal):
                item["allowed_regions"]=[key]


def _functional_role_listed(role, literal):
    """Match a short heading alias to its longer source-backed equipment role."""
    role_tokens = [token.casefold() for token in re.findall(r"[a-z0-9]+", str(role or ""), re.I)]
    if not role_tokens:
        return False
    role_label = _functional_terminal_label(role_tokens)
    entries = re.split(r"\s*(?:;|,|\band\b)\s*", str(literal or ""), flags=re.I)
    for entry in entries:
        alias_tokens = [token.casefold() for token in re.findall(r"[a-z0-9]+", entry, re.I)]
        if not alias_tokens:
            continue
        alias_label = _functional_terminal_label(alias_tokens)
        if alias_label and role_label and alias_label != role_label:
            continue
        required = [
            token for token in alias_tokens
            if token not in _FUNCTIONAL_GENERIC_ROLE_WORDS
        ]
        if required and all(token in role_tokens for token in required):
            return True
    return False


def _functional_terminal_label(tokens):
    token = tokens[-1] if tokens else ""
    return token if token.isdigit() or len(token) == 1 and token.isalpha() else None


def transfer_cells(problem):
    return [c["parameters"] for c in (problem.get("constraints") or {}).get("hard",[])
            if (c.get("parameters") or {}).get("planning_kind")=="manual_conveyor_cell"]


def _refresh(item):
    x,z=item["position"]["x_m"],item["position"]["z_m"]
    w,d=item["dimensions_m"]["width_m"],item["dimensions_m"]["depth_m"]
    item["footprint"].update(min_x=x-w/2,max_x=x+w/2,min_z=z-d/2,max_z=z+d/2,width_m=w,depth_m=d)


def align_transfer_cells(layout, problem, *, ignore_locked=False):
    by_id = {p["instance_id"]:p for p in layout.get("placements") or []}
    facts = {p["instance_id"]:p for p in problem.get("equipment_instances") or []}
    for cell in transfer_cells(problem):
        key=cell["conveyor_instance_id"]
        if key not in by_id: continue
        conveyor=by_id[key]; yaw=float(conveyor["rotation"]["yaw_deg"])
        if abs(yaw/90-round(yaw/90))>1e-8: continue
        angle=math.radians(yaw); ux,uz=round(math.cos(angle)),round(-math.sin(angle))
        long_size=float((facts[key].get("geometry") or {})["dimensions_m"]["width_m"])
        for field,sign in (("input_table_instance_id",-1),("output_table_instance_id",1)):
            table=by_id.get(cell[field])
            if not table: continue
            raw=facts[cell[field]]["geometry"]["dimensions_m"]
            along=float(raw["width_m"])
            distance=long_size/2+cell["end_gap_m"]+along/2
            if facts[cell[field]].get("locked") and not ignore_locked:
                continue
            table["position"]["x_m"]=conveyor["position"]["x_m"]+sign*distance*ux
            table["position"]["z_m"]=conveyor["position"]["z_m"]+sign*distance*uz
            table["rotation"]["yaw_deg"]=yaw;table["yaw_deg"]=yaw
            rotated=round(yaw/90)%2
            table["dimensions_m"].update(width_m=float(raw["depth_m"] if rotated else raw["width_m"]),
                                         depth_m=float(raw["width_m"] if rotated else raw["depth_m"]))
            _refresh(table)
    return layout


def workcell_regions(layout, problem):
    by_id={p["instance_id"]:p for p in layout.get("placements") or []};rows=[]
    for cell in transfer_cells(problem):
        ids=[cell[k] for k in ("conveyor_instance_id","input_table_instance_id","output_table_instance_id")]
        if not all(i in by_id for i in ids): continue
        footprints=[by_id[i]["footprint"] for i in ids]
        a,b,c,d=min(f["min_x"] for f in footprints),max(f["max_x"] for f in footprints),min(f["min_z"] for f in footprints),max(f["max_z"] for f in footprints)
        # Explicitly parallel to the east-west aisle; aisle-facing strip is south.
        rows.append(dict(kind="manual_transfer_cell",member_instance_ids=ids,min_x=a,max_x=b,min_z=c,max_z=d,
            operator_area=dict(min_x=a,max_x=b,min_z=c-cell["operator_strip_m"],max_z=c),parameters=cell))
    return rows


def transfer_cell_violations(layout, problem):
    actual={p["instance_id"]:p for p in layout.get("placements") or []}
    # Generate the required alignment independently of mutation policy. A
    # locked endpoint remains in place during normalization, but a mismatched
    # locked pose still violates the connected-cell requirement.
    aligned=align_transfer_cells(copy.deepcopy(layout),problem,ignore_locked=True)
    violations=[]
    for expected in aligned.get("placements") or []:
        item=actual[expected["instance_id"]]
        delta=sum(abs(item["position"][a]-expected["position"][a]) for a in ("x_m","z_m"))
        if delta>1e-6 or abs(item["rotation"]["yaw_deg"]-expected["rotation"]["yaw_deg"])>1e-6:
            violations.append(dict(violation_type="transfer_end_alignment",instances=[item["instance_id"]],deficit_m=delta))
    for region in workcell_regions(layout,problem):
        area=region["operator_area"]
        for zone in (layout.get("site") or {}).get("non_placeable_zones") or []:
            if min(area["max_x"],zone["max_x_m"])-max(area["min_x"],zone["min_x_m"])>1e-8 and min(area["max_z"],zone["max_z_m"])-max(area["min_z"],zone["min_z_m"])>1e-8:
                violations.append(dict(violation_type="transfer_operator_area_reserved",instances=region["member_instance_ids"],deficit_m=1.))
        for key,item in actual.items():
            if key in region["member_instance_ids"]: continue
            f=item["footprint"]
            if min(f["max_x"],area["max_x"])-max(f["min_x"],area["min_x"])>1e-8 and min(f["max_z"],area["max_z"])-max(f["min_z"],area["min_z"])>1e-8:
                violations.append(dict(violation_type="transfer_operator_area",instances=[key],deficit_m=1.))
    return violations
