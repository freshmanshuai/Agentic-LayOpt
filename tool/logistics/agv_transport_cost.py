"""A declared AGV shuttle cost on collision-free rectilinear paths.

The roadmap uses a conservative square enclosing the calibrated vehicle body.
External stops are planning handoff locations, not manufacturer docking ports.
No timing, traffic, battery, or payload feasibility is implied by this model.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from tool.logistics.facility_logistics import fixed_flow_endpoints

AGV_TRANSPORT_OBJECTIVE = "minimize_agv_transport_cost"


def agv_cost_requested(text):
    return bool(re.search(r"\bAGV\s+(?:transport(?:ation)?|travel|operating)\s+cost\b", str(text), re.I))


def compile_agv_parameters(text):
    """Rates and task policy must be declared in the ordinary user conversation."""
    number = r"(\d+(?:\.\d+)?)"
    loaded = re.search(r"loaded travel(?: costs?)?\s+" + number + r"\s+(?:cost )?index units? per meter", text, re.I)
    empty_pattern = r"empty travel(?: costs?)?\s+" + number
    empty = re.search(
        empty_pattern + r"\s+(?:cost )?index units? per meter",
        text,
        re.I,
    )
    if not empty:
        shorthand = re.search(empty_pattern, text, re.I)
        if shorthand:
            sentence_start = max(
                text.rfind(".", 0, shorthand.start()),
                text.rfind("\n", 0, shorthand.start()),
            ) + 1
            sentence_ends = [
                position
                for position in (
                    text.find(".", shorthand.end()),
                    text.find("\n", shorthand.end()),
                )
                if position >= 0
            ]
            sentence_end = min(sentence_ends) if sentence_ends else len(text)
            sentence = text[sentence_start:sentence_end]
            if re.search(
                r"loaded travel(?: costs?)?\s+\d+(?:\.\d+)?\s+"
                r"(?:cost )?index units? per meter[^\n]*empty travel(?: costs?)?\s+\d+(?:\.\d+)?\s*$",
                sentence,
                re.I,
            ):
                empty = shorthand
    if not loaded or not empty:
        raise ValueError("Please specify the assumed loaded and empty travel cost indices per meter; no actual operating rates are available.")
    if not re.search(r"(?:empty|unloaded)[^.\n]{0,60}return[^.\n]{0,50}(?:pickup|pick-up|collection|origin)", text, re.I):
        raise ValueError("Should each loaded delivery be followed by an empty return to its pickup point, or should tasks be sequenced? The current shuttle model requires an explicit empty-return policy.")
    if not re.search(r"(?:one|1)\s+batch\s+per\s+(?:trip|mission|delivery)", text, re.I):
        raise ValueError("Please specify batches per AGV delivery. The current model supports one batch per trip.")
    return dict(loaded_cost_per_meter=float(loaded.group(1)), empty_cost_per_meter=float(empty.group(1)),
                mission_policy="loaded_then_empty_return", batches_per_trip=1,
                include_daily_home_trip=True, units="planning_transport_cost_index_per_day",
                basis="Declared case assumptions, not monetary charges or measured energy consumption",
                body_model="circumscribed_square", stop_policy="aisle_facing_external_manual_handoff")


def validate_agv_parameters(parameters):
    for name in ("loaded_cost_per_meter", "empty_cost_per_meter"):
        value = float(parameters[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError("AGV travel cost indices must be finite and nonnegative.")
    if parameters.get("mission_policy") != "loaded_then_empty_return" or parameters.get("batches_per_trip") != 1:
        raise ValueError("Unsupported AGV task policy; do not silently approximate a sequenced fleet schedule.")


def agv_parameters(problem):
    return next((o.get("parameters") or {} for o in problem.get("objectives") or []
                 if o.get("objective_type") == AGV_TRANSPORT_OBJECTIVE), None)


def _rect(item):
    p = item.get("position") or item.get("pose") or {}
    d = item.get("dimensions_m") or {}
    return (float(p["x_m"])-float(d["width_m"])/2, float(p["x_m"])+float(d["width_m"])/2,
            float(p["z_m"])-float(d["depth_m"])/2, float(p["z_m"])+float(d["depth_m"])/2)


def _inputs(layout, problem):
    placements = {p["instance_id"]: p for p in layout.get("placements") or []}
    flows = [f for f in problem.get("flow_relations") or [] if f.get("transport_mode") == "agv"]
    vehicle_ids = {f.get("carrier_instance_id") for f in flows}
    if None in vehicle_ids:
        raise ValueError("Every AGV task needs a grounded carrier instance.")
    features=(layout.get("site") or {}).get("site_features") or ((problem.get("site_ref") or {}).get("site_snapshot") or {}).get("site_features") or []
    endpoints = dict(placements, **{f["site_feature_id"]:f for f in features if f.get("pose") and f.get("dimensions_m")})
    # Loaded buffers are occupied storage; the AGV stops outside their marked area.
    obstacles = tuple(_rect(p) for key, p in endpoints.items() if key not in vehicle_ids and
                      (key in placements or p.get("feature_type") in {"buffer_area","forbidden_zone","column","obstacle"}))
    from tool.logistics.transfer_workcell import workcell_regions
    obstacles += tuple((r["operator_area"]["min_x"],r["operator_area"]["max_x"],r["operator_area"]["min_z"],r["operator_area"]["max_z"]) for r in workcell_regions(layout,problem))
    site = layout.get("site") or {}
    width, depth = float(site["width_m"]), float(site["depth_m"])
    result = []
    for vehicle_id in sorted(vehicle_ids):
        vehicle = placements[vehicle_id]
        size = vehicle["dimensions_m"]
        radius = math.hypot(float(size["width_m"]), float(size["depth_m"]))/2
        assigned = [f for f in flows if f["carrier_instance_id"] == vehicle_id]
        station_ids = sorted({f[k] for f in assigned for k in ("from_instance_id", "to_instance_id")})
        stops = []
        for key in station_ids:
            if endpoints[key].get("feature_type") in {"entrance","exit"}:
                p=endpoints[key]["pose"]
                stops.append((math.copysign(width/2-radius-1e-5,float(p["x_m"])),float(p["z_m"])))
                continue
            a, b, c, d = _rect(endpoints[key])
            # A layout planning stop just outside the aisle-facing body face.
            # Tiny epsilon separates contact from occupied geometry; it is not a service clearance.
            z = c-radius-1e-5 if (c+d)/2 >= 0 else d+radius+1e-5
            stops.append(((a+b)/2, z))
        home = (float(vehicle["position"]["x_m"]), float(vehicle["position"]["z_m"]))
        result.append((vehicle_id, station_ids, assigned, home, tuple(stops), radius, width, depth, obstacles))
    if not result:
        raise ValueError("AGV transport cost requires actual AGV-assigned transfers.")
    return result


@lru_cache(maxsize=96)
def _roadmap(width, depth, radius, obstacles, terminals):
    """Exact rectilinear shortest paths for this conservative body envelope.

    Coordinate compression contains every expanded obstacle edge and every exact
    terminal. Edges cross only consecutive coordinates and are checked at their
    midpoint, so a narrow obstacle cannot disappear between sampled grid cells.
    """
    xmin, xmax, zmin, zmax = -width/2+radius, width/2-radius, -depth/2+radius, depth/2-radius
    expanded = [(a-radius, b+radius, c-radius, d+radius) for a,b,c,d in obstacles]
    xs = np.array(sorted({xmin, xmax, *(x for x,z in terminals),
                          *(x for a,b,c,d in expanded for x in (a,b) if xmin <= x <= xmax)}))
    zs = np.array(sorted({zmin, zmax, *(z for x,z in terminals),
                          *(z for a,b,c,d in expanded for z in (c,d) if zmin <= z <= zmax)}))
    xx, zz = np.meshgrid(xs, zs)
    def free(x, z):
        valid = (x >= xmin-1e-9)&(x <= xmax+1e-9)&(z >= zmin-1e-9)&(z <= zmax+1e-9)
        for a,b,c,d in expanded:
            valid &= ~((x > a+1e-9)&(x < b-1e-9)&(z > c+1e-9)&(z < d-1e-9))
        return valid
    mask = free(xx, zz)
    index = np.arange(xx.size).reshape(xx.shape)
    horizontal = mask[:,:-1]&mask[:,1:]&free((xx[:,:-1]+xx[:,1:])/2, zz[:,:-1])
    vertical = mask[:-1,:]&mask[1:,:]&free(xx[:-1,:], (zz[:-1,:]+zz[1:,:])/2)
    u = np.r_[index[:,:-1][horizontal], index[:-1,:][vertical]]
    v = np.r_[index[:,1:][horizontal], index[1:,:][vertical]]
    weights = np.r_[np.broadcast_to(np.diff(xs), horizontal.shape)[horizontal],
                    np.broadcast_to(np.diff(zs)[:,None], vertical.shape)[vertical]]
    graph = coo_matrix((np.r_[weights,weights], (np.r_[u,v],np.r_[v,u])), shape=(xx.size,xx.size)).tocsr()
    terminal_nodes = np.array([int(np.searchsorted(zs,z))*len(xs)+int(np.searchsorted(xs,x)) for x,z in terminals])
    distances, predecessors = dijkstra(graph, directed=False, indices=terminal_nodes, return_predecessors=True)
    result = distances[:,terminal_nodes].copy()
    for i, node in enumerate(terminal_nodes):
        if not mask.flat[node]:
            result[i,:] = np.inf; result[:,i] = np.inf
    return result, predecessors, terminal_nodes, xs, zs


def evaluate_agv_cost(layout, problem, *, include_paths=False):
    parameters = agv_parameters(problem)
    validate_agv_parameters(parameters or {})
    loaded_rate, empty_rate = (float(parameters[k]) for k in ("loaded_cost_per_meter", "empty_cost_per_meter"))
    totals = dict(loaded_distance_m_per_day=0., empty_distance_m_per_day=0., home_and_reposition_distance_m_per_day=0., cost_index_per_day=0.)
    reports = []; unreachable = []
    for vehicle, station_ids, assigned, home, stops, radius, width, depth, obstacles in _inputs(layout, problem):
        terminals = (home,)+stops
        distances, predecessors, terminal_nodes, xs, zs = _roadmap(width,depth,radius,obstacles,terminals)
        lookup = {key:i+1 for i,key in enumerate(station_ids)}
        def waypoints(i,j):
            if not math.isfinite(float(distances[i,j])): return []
            node=int(terminal_nodes[j]);path=[]
            while node>=0:
                path.append(dict(x_m=float(xs[node%len(xs)]),z_m=float(zs[node//len(xs)])))
                if node==terminal_nodes[i]:break
                node=int(predecessors[i,node])
            return path[::-1]
        rows = []; weighted_distance = 0.
        for f in assigned:
            source, target = f["from_instance_id"], f["to_instance_id"]
            i,j = lookup[source], lookup[target]
            distance = float(distances[i,j]); trips = float(f.get("flow_weight",1))
            if not math.isfinite(distance):
                unreachable.append(dict(vehicle=vehicle,source=source,target=target)); continue
            weighted_distance += trips*distance
            row = dict(source=source,target=target,trips_per_day=trips,loaded_distance_m=distance,
                       empty_return_distance_m=distance,cost_index_per_day=trips*distance*(loaded_rate+empty_rate))
            if include_paths:
                row["loaded_waypoints"] = waypoints(i,j)
            rows.append(row)
        # The shuttle jobs may run in any order. Start/end at the same closest
        # pickup; every intervening job returns to its own origin. Repositioning
        # BETWEEN different pickup origins is therefore still necessary.
        # Visit pickup groups in first occurrence in the declared product
        # routes. This is an explicit dispatch policy, not an optimal schedule.
        origins = list(dict.fromkeys(lookup[f["from_instance_id"]] for f in assigned))
        itinerary = [0]+origins+[0]
        reposition = sum(float(distances[i,j]) for i,j in zip(itinerary,itinerary[1:]))
        if not math.isfinite(reposition):
            unreachable.append(dict(vehicle=vehicle,source="home/origin",target="home/origin"))
            reposition = 0.
        cost = weighted_distance*(loaded_rate+empty_rate)+reposition*empty_rate
        totals["loaded_distance_m_per_day"] += weighted_distance
        totals["empty_distance_m_per_day"] += weighted_distance+reposition
        totals["home_and_reposition_distance_m_per_day"] += reposition
        totals["cost_index_per_day"] += cost
        reports.append(dict(vehicle=vehicle,body_envelope_half_size_m=radius,home=home,
                            station_stops={key:dict(x_m=stops[i][0],z_m=stops[i][1]) for i,key in enumerate(station_ids)},
                            grouped_pickup_order=[station_ids[i-1] for i in origins],
                            daily_reposition_distance_m=reposition,
                            empty_reposition_legs=[dict(source="home" if i==0 else station_ids[i-1],target="home" if j==0 else station_ids[j-1],distance_m=float(distances[i,j]),waypoints=waypoints(i,j)) for i,j in zip(itinerary,itinerary[1:])] if include_paths else [],
                            missions=rows,cost_index_per_day=cost))
    return dict(**totals,feasible=not unreachable,unreachable=unreachable,vehicles=reports,
                assumptions=parameters,limitations="Static body-clearance paths and declared grouped shuttle policy only; no traffic, timing, energy, battery, or payload validation.")


def agv_transport_cost(layout, problem):
    report = evaluate_agv_cost(layout, problem)
    # Finite search penalty is kept separate from acceptance: unreachable
    # transfers are also emitted as hard violations by the shared validator.
    return report["cost_index_per_day"] + len(report["unreachable"])*1e8


def agv_access_required(problem):
    return any((c.get("parameters") or {}).get("planning_kind")=="agv_external_access"
               for c in (problem.get("constraints") or {}).get("hard",[]))


def agv_access_issues(layout, problem):
    issues=[]
    for vehicle,station_ids,assigned,home,stops,radius,width,depth,obstacles in _inputs(layout,problem):
        distance,*_= _roadmap(width,depth,radius,obstacles,(home,)+stops)
        for i,key in enumerate(station_ids,1):
            if not math.isfinite(float(distance[0,i])):
                issues.append(dict(vehicle=vehicle,source="home",target=key))
    return issues
