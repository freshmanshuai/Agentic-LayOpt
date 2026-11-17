"""Small explicit updates to an accepted workshop conversation."""
import copy
import re
from collections import Counter

from tool.context.session_engineering_context import inventory_role_surface_key


def retain_confirmed_role_declarations(
    frame,
    grounding,
    previous_mentions,
    *,
    confirmed_asset_choices=None,
    allow_partial_inventory_merge=False,
):
    """Keep station names when a clarification only groups their catalog types.

    Both inventories must resolve to the same models and counts in actual RAG
    results. Changed quantities, changed models, and unresolved choices pass
    through untouched. This retains aliases, not layout or engineering rules.
    """
    previous = previous_mentions or []
    current = (frame or {}).get("asset_mentions") or []
    fold = lambda value: " ".join(str(value or "").casefold().split())
    matches = {fold(m.get("surface_text")):m for m in (grounding or {}).get("asset_matches") or []}

    if allow_partial_inventory_merge:
        previous_by_role = {}
        for index, item in enumerate(previous):
            if not isinstance(item, dict) or not item.get("surface_text"):
                continue
            previous_by_role.setdefault(
                inventory_role_surface_key(item.get("surface_text")), []
            ).append((index, item))

        choices_by_title = {}
        for choice in confirmed_asset_choices or []:
            if not isinstance(choice, dict):
                continue
            role_key = inventory_role_surface_key(choice.get("surface_text"))
            model_id = str(choice.get("model_asset_id") or "").strip()
            display_label = str(choice.get("display_label") or "")
            title = display_label.split(" (", 1)[0].strip()
            title_key = _catalog_title_key(title)
            prior_entries = previous_by_role.get(role_key) or []
            if len(prior_entries) != 1:
                continue
            prior_index, prior = prior_entries[0]
            match_entries = [
                item
                for item in (grounding or {}).get("asset_matches") or []
                if isinstance(item, dict)
                and inventory_role_surface_key(item.get("surface_text")) == role_key
            ]
            match = match_entries[0] if len(match_entries) == 1 else {}
            candidates = {
                str(candidate.get("model_asset_id") if isinstance(candidate, dict) else candidate)
                for candidate in match.get("candidate_model_asset_ids") or []
            }
            if (
                not prior
                or not title_key
                or not model_id
                or match.get("selected_model_asset_id") != model_id
                or model_id not in candidates
            ):
                continue
            choices_by_title.setdefault(title_key, []).append(
                (prior_index, prior, model_id)
            )

        covered_previous_indices = set()
        choice_merge_is_complete = bool(current) and bool(previous)
        current_choice_titles = set()
        has_direct_role_mentions = False
        current_quantity_total = 0
        for mention in current:
            title_key = _catalog_title_key(mention.get("surface_text"))
            choices = choices_by_title.get(title_key) or []
            try:
                current_quantity = int(mention.get("quantity") or 1)
            except (TypeError, ValueError):
                current_quantity = 0
            current_quantity_total += current_quantity
            if choices:
                current_choice_titles.add(title_key)
                choice_indices = {index for index, _prior, _model in choices}
                selected_models = {model for _index, _prior, model in choices}
                expected_quantity = sum(
                    int(prior.get("quantity") or 1)
                    for index, prior, _model in choices
                    if index in choice_indices
                )
                if (
                    len(choice_indices) != len(choices)
                    or len(selected_models) != 1
                    or current_quantity != expected_quantity
                    or covered_previous_indices.intersection(choice_indices)
                ):
                    choice_merge_is_complete = False
                    break
                covered_previous_indices.update(choice_indices)
                continue

            has_direct_role_mentions = True
            role_key = inventory_role_surface_key(mention.get("surface_text"))
            prior_entries = previous_by_role.get(role_key) or []
            if len(prior_entries) != 1:
                choice_merge_is_complete = False
                break
            prior_index, prior = prior_entries[0]
            expected_quantity = int(prior.get("quantity") or 1)
            prior_match = matches.get(fold(prior.get("surface_text"))) or {}
            selected_model_id = str(
                prior_match.get("selected_model_asset_id") or ""
            ).strip()
            candidate_ids = {
                str(candidate.get("model_asset_id") if isinstance(candidate, dict) else candidate)
                for candidate in prior_match.get("candidate_model_asset_ids") or []
            }
            if (
                current_quantity != expected_quantity
                or prior_index in covered_previous_indices
                or not selected_model_id
                or selected_model_id not in candidate_ids
            ):
                choice_merge_is_complete = False
                break
            covered_previous_indices.add(prior_index)

        complete_role_merge = bool(
            choice_merge_is_complete
            and len(covered_previous_indices) == len(previous)
            and current_quantity_total
            == sum(
                int(item.get("quantity") or 1)
                for item in previous
                if isinstance(item, dict)
            )
        )
        catalog_only_choice_merge = bool(
            choice_merge_is_complete
            and not has_direct_role_mentions
            and current_choice_titles == set(choices_by_title)
        )
        choice_merge_is_complete = (
            complete_role_merge or catalog_only_choice_merge
        )

        if choice_merge_is_complete:
            roles = [
                {
                    "surface_text": str(item["surface_text"]),
                    "quantity": int(item.get("quantity") or 1),
                }
                for item in previous
                if isinstance(item, dict) and item.get("surface_text")
            ]
            if roles != current:
                out = copy.deepcopy(frame)
                out["asset_mentions"] = roles
                return out, copy.deepcopy(grounding), True

    if not previous or not current or any(not m.get("model_asset_id") for m in previous):
        return frame,grounding,False
    totals = Counter(); examples = {}
    for mention in current:
        match = matches.get(fold(mention.get("surface_text")),{})
        selected = match.get("selected_model_asset_id")
        status = str(match.get("decision") or match.get("ambiguity_status") or "").lower()
        if not selected or status not in {"resolved","resolve"}:
            return frame,grounding,False
        totals[selected] += int(mention.get("quantity") or 1)
        examples[selected] = match
    prior_totals = Counter()
    for mention in previous:
        prior_totals[mention["model_asset_id"]] += int(mention.get("quantity") or 1)
    if totals != prior_totals:
        return frame,grounding,False
    roles = [dict(surface_text=m["surface_text"],quantity=int(m.get("quantity") or 1)) for m in previous]
    if roles == [dict(surface_text=m["surface_text"],quantity=int(m.get("quantity") or 1)) for m in current]:
        return frame,grounding,False
    out,evidence = copy.deepcopy(frame),copy.deepcopy(grounding)
    out["asset_mentions"] = roles
    for mention in previous:
        key=fold(mention["surface_text"])
        if key in matches:
            continue
        alias = copy.deepcopy(examples[mention["model_asset_id"]])
        alias.update(surface_text=mention["surface_text"],decision_source="retained_confirmed_role",
                     retained_from_surface=examples[mention["model_asset_id"]]["surface_text"])
        evidence["asset_matches"].append(alias)
    return out,evidence,True


def _catalog_title_key(value):
    label = str(value or "").split(" (", 1)[0].casefold()
    tokens = re.findall(r"[a-z0-9]+", label)
    while tokens and tokens[0] in {
        "a", "an", "the", "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten",
    }:
        tokens.pop(0)
    return " ".join(tokens)


def inherited_objective_parameters(context, objective_type):
    sources = [context.get("last_flp_ir"), context.get("authoritative_flp_ir"),
               (context.get("authoritative_context") or {}).get("authoritative_flp_ir")]
    for source in sources:
        for objective in (source or {}).get("objectives") or []:
            if objective.get("objective_type") == objective_type and objective.get("parameters"):
                return copy.deepcopy(objective["parameters"])
    return None


def service_duty_transfers(text, base_problem, edit_ir):
    """Resolve explicit 'transfer X's tool-service duties to Y' deltas.

    Do not infer service reassignment merely because a cabinet is removed.
    Destination models must already have been grounded in the typed edit.
    """
    pattern = r"\btransfer\s+(?:all\s+of\s+)?([^.;\n]+?)['’]s\s+tool[- ]service\s+duties\s+to\s+([^.;\n]+)"
    clauses = re.findall(pattern, text, re.I)
    if not clauses:
        return []
    directory = [(i.get("role", "").casefold().strip(), i["instance_id"])
                 for i in base_problem.get("equipment_instances") or []]
    removed = {op.get("instance_id") for op in edit_ir["operations"] if op["op_type"] == "remove"}
    for op in edit_ir["operations"]:
        if op["op_type"] == "add" and int(op["parameters"].get("quantity") or 1) == 1:
            directory.append((op["parameters"].get("role", "").casefold().strip(),op["instance_id"]))
    result = []
    for source, target in clauses:
        def resolve(name):
            name = re.sub(r"^the\s+", "", name.strip(), flags=re.I).casefold()
            matches = {key for role,key in directory if role == name}
            if len(matches) != 1:
                raise ValueError(f"Please clarify the tool-service station named {name}.")
            return matches.pop()
        old, new = resolve(source), resolve(target)
        if old == new or new in removed:
            raise ValueError("The replacement tool-service station must remain in the workshop.")
        result.append((old,new))
    return result


def apply_service_duty_transfers(base_problem, edited_problem, transfers):
    ids = {i["instance_id"] for i in edited_problem["equipment_instances"]}
    for old,new in transfers:
        if new not in ids:
            raise ValueError("The new tool-service station was not created.")
        inherited = [f for f in base_problem.get("flow_relations") or []
                     if f.get("flow_type") == "tool_service" and old in (f.get("from_instance_id"),f.get("to_instance_id"))]
        if not inherited:
            raise ValueError("The named source has no accepted tool-service duties to transfer.")
        edited_problem["flow_relations"] = [f for f in edited_problem.get("flow_relations") or []
            if not (f.get("flow_type") == "tool_service" and old in (f.get("from_instance_id"),f.get("to_instance_id")))]
        for flow in inherited:
            row = copy.deepcopy(flow)
            for field in ("from_instance_id","to_instance_id"):
                if row.get(field) == old:
                    row[field] = new
            edited_problem["flow_relations"].append(row)
