"""Helpers for durable, natural-language engineering context in a session."""

import re


_SECTION_HEADINGS = {
    "equipment inventory and organization",
    "evaluation assumptions",
    "facility and work areas",
    "production and logistics",
    "case requirements",
}

_INVENTORY_COUNTS = {
    "a", "an", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
}
_INVENTORY_COUNT_VALUES = {
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
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


def persist_engineering_context(session, incoming_context):
    """Update the saved brief only when the caller supplied the field.

    ``None`` means the request omitted the field. An empty string is an explicit
    replacement that clears the brief.
    """

    if incoming_context is not None:
        if not isinstance(incoming_context, str):
            raise TypeError("engineering_context must be a natural-language string.")
        session["engineering_context"] = incoming_context.strip()
    session.setdefault("engineering_context", "")
    return session["engineering_context"]


def engineering_context_source_history(history, engineering_context):
    """Add a source record for literal-slot checks while keeping it distinct."""

    result = list(history or [])
    if str(engineering_context or "").strip():
        result.append(
            {
                "role": "engineering_context",
                "text": str(engineering_context).strip(),
            }
        )
    return result


def engineering_inventory_section(engineering_context):
    """Return the clearly labeled equipment inventory portion of the brief."""

    lines = str(engineering_context or "").splitlines()
    start = None
    collected = []
    for line in lines:
        heading = line.strip().strip("#* ").casefold()
        if heading in {
            "equipment inventory and organization",
            "equipment inventory",
            "inventory",
        }:
            start = True
            continue
        if start and heading in _SECTION_HEADINGS:
            break
        if start:
            collected.append(line)
    return "\n".join(collected).strip()


def engineering_inventory_declarations(engineering_context):
    """Return inventory declarations without the row and bay organization text."""

    section = engineering_inventory_section(engineering_context)
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", section) if part.strip()]
    declarations = [
        paragraph
        for paragraph in paragraphs
        if re.search(
            r"(?im)^\s*(?:use\s+\w+\s+main\s+process\s+units|add\s+\w+\s+auxiliary\s+objects)\b",
            paragraph,
        )
    ]
    return "\n\n".join(declarations) if declarations else section


def engineering_inventory_items(engineering_context):
    """Split the declared inventory into its named equipment entries."""

    declarations = engineering_inventory_declarations(engineering_context) or str(
        engineering_context or ""
    ).strip()
    if not declarations:
        return []
    items = []
    for paragraph in re.split(r"\n\s*\n", declarations):
        paragraph = re.sub(
            r"(?im)^\s*(?:use\s+\w+\s+main\s+process\s+units|add\s+\w+\s+auxiliary\s+objects)\s*:\s*",
            "",
            paragraph,
        )
        for clause in paragraph.split(";"):
            for sentence in re.split(r"\.\s+(?=[A-Z])", clause):
                sentence = re.sub(
                    r"(?i)^\s*(?:south|north)\s*:\s*", "", sentence
                )
                sentence = sentence.strip(" \t\r\n.*")
                if not sentence or re.match(r"(?i)^use\s+freestanding\b", sentence):
                    continue
                items.append(sentence)
    return items


def _folded(value):
    return " ".join(str(value or "").casefold().split())


def inventory_role_surface_key(value):
    """Fold only a leading count/article used as a quantity declaration.

    The prefix must be a separate whitespace-delimited token, so class names
    such as ``three-axis`` remain intact. Labels and the rest of the role stay
    unchanged, keeping A/B instances distinct.
    """

    raw = str(value or "").strip()
    match = re.match(
        r"^(?P<prefix>a|an|the|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|\d+)\s+(?P<body>.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        prefix = match.group("prefix").casefold()
        raw_prefix = raw.split(None, 1)[0]
        if prefix not in {"a", "an", "the"} or raw_prefix.islower():
            raw = match.group("body")
    return _folded(raw)


def _leading_inventory_count(value):
    match = re.match(
        r"^\s*(?P<count>a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty|\d+)\s+",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    token = match.group("count").casefold()
    raw_token = str(value or "").strip().split(None, 1)[0]
    if token in {"a", "an", "the"} and not raw_token.islower():
        return None
    return int(token) if token.isdigit() else _INVENTORY_COUNT_VALUES[token]


def _inventory_surface_evidence_key(surface, inventory_items, *, quantity=None):
    indices = inventory_surface_item_indices(surface, inventory_items)
    if len(indices) != 1:
        return ("surface", inventory_role_surface_key(surface), quantity)
    index = indices[0]
    surface_count = _leading_inventory_count(surface)
    source_count = _leading_inventory_count(inventory_items[index])
    count = surface_count if surface_count is not None else source_count
    resolved_quantity = quantity if quantity is not None else count
    return (
        "inventory",
        index,
        inventory_role_surface_key(surface),
        count,
        resolved_quantity,
    )


def _contains_phrase(text, phrase):
    text = _folded(text)
    phrase = _folded(phrase)
    if not text or not phrase:
        return False
    return re.search(
        rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text
    ) is not None


def _inventory_tokens(value):
    tokens = re.findall(r"[a-z0-9]+", _folded(value))
    while tokens and (tokens[0] in _INVENTORY_COUNTS or tokens[0].isdigit()):
        tokens.pop(0)
    tokens = [token for token in tokens if token not in {"labeled", "labelled"}]
    singulars = {
        "centers": "center",
        "machines": "machine",
        "stations": "station",
        "workstations": "workstation",
        "racks": "rack",
        "cabinets": "cabinet",
        "tables": "table",
        "units": "unit",
        "conveyors": "conveyor",
        "agvs": "agv",
    }
    return [singulars.get(token, token) for token in tokens]


def _inventory_item_matches_surface(surface, item):
    surface_tokens = _inventory_tokens(surface)
    if not surface_tokens:
        return False
    surface_set = set(surface_tokens)
    surface_labels = surface_set & {"a", "b", "c", "d", "e", "f"}
    item_tokens = _inventory_tokens(item)
    item_set = set(item_tokens)
    if not surface_set <= item_set:
        return False
    item_labels = item_set & {"a", "b", "c", "d", "e", "f"}
    if len(item_labels) > 1 and not surface_labels:
        return False
    surface_core = surface_set - {"a", "b", "c", "d", "e", "f", "and"}
    item_core = item_set - {"a", "b", "c", "d", "e", "f", "and"}
    return len(surface_core) >= min(2, len(item_core))


def inventory_surface_item_indices(surface, inventory_items):
    """Match a mention to one complete declared role, including labeled units.

    Ambiguous generic phrases such as “storage rack” do not identify either of
    two qualified inventory roles. Row labels and evaluation categories are
    excluded because matching uses the declared inventory entries only.
    """

    matches = [
        index
        for index, item in enumerate(inventory_items or [])
        if _inventory_item_matches_surface(surface, item)
    ]
    return matches if len(matches) == 1 else []


def inventory_surface_changed(surface, previous_inventory_context, current_context):
    """Report whether this surface's declared inventory item changed."""

    if previous_inventory_context is None:
        return True
    previous_items = engineering_inventory_items(previous_inventory_context)
    current_items = engineering_inventory_items(current_context)
    previous_matches = {
        _folded(item)
        for item in previous_items
        if _inventory_item_matches_surface(surface, item)
    }
    current_matches = {
        _folded(item)
        for item in current_items
        if _inventory_item_matches_surface(surface, item)
    }
    return previous_matches != current_matches


def inventory_context_surface_keys(linguistic_frame, engineering_context):
    items = engineering_inventory_items(engineering_context)
    return {
        _inventory_surface_evidence_key(
            item.get("surface_text"),
            items,
            quantity=item.get("quantity"),
        )
        for item in (linguistic_frame or {}).get("asset_mentions") or []
        if isinstance(item, dict)
        and item.get("surface_text")
        and inventory_surface_item_indices(item.get("surface_text"), items)
    }


def reusable_confirmed_asset_choices(
    confirmed_asset_choices,
    linguistic_frame,
    engineering_context,
    *,
    inventory_context_changed=False,
    previous_inventory_context=None,
):
    choices = list(confirmed_asset_choices or [])
    if not inventory_context_changed:
        return choices
    changed_surface_keys = {
        _folded(item.get("surface_text"))
        for item in (linguistic_frame or {}).get("asset_mentions") or []
        if isinstance(item, dict)
        and item.get("surface_text")
        and inventory_surface_item_indices(
            item.get("surface_text"),
            engineering_inventory_items(engineering_context),
        )
        and inventory_surface_changed(
            item.get("surface_text"),
            previous_inventory_context,
            engineering_context,
        )
    }
    return [
        choice
        for choice in choices
        if not isinstance(choice, dict)
        or (
            _folded(choice.get("surface_text")) not in changed_surface_keys
            and not inventory_surface_changed(
                choice.get("surface_text"),
                previous_inventory_context,
                engineering_context,
            )
        )
    ]


def new_model_operation_surfaces(linguistic_frame, user_message):
    """Return literal new-model surfaces for ADD/REPLACE operations only."""

    frame = linguistic_frame if isinstance(linguistic_frame, dict) else {}
    message = str(user_message or "")
    result = []
    for operation in frame.get("edit_operations") or []:
        if not isinstance(operation, dict):
            continue
        op_type = str(operation.get("op_type") or "").casefold()
        if op_type not in {"add", "replace"}:
            continue
        surface = str(operation.get("new_asset_surface_text") or "").strip()
        if op_type == "add" and not surface:
            surface = str(operation.get("surface_text") or "").strip()
        if op_type == "replace" and not surface:
            old_surface = re.sub(
                r"^(?:(?:a|an|the|existing)\s+)+",
                "",
                _folded(operation.get("surface_text")),
            )
            sources = [str(operation.get("source_text") or ""), message]
            for mention in frame.get("asset_mentions") or []:
                candidate = str(mention.get("surface_text") or "").strip() if isinstance(mention, dict) else ""
                candidate_key = re.sub(
                    r"^(?:(?:a|an|the|existing)\s+)+", "", _folded(candidate)
                )
                if (
                    candidate
                    and candidate_key != old_surface
                    and any(_contains_phrase(source, candidate) for source in sources)
                ):
                    surface = candidate
                    break
        if surface and _contains_phrase(message, surface):
            if _folded(surface) not in {_folded(item) for item in result}:
                result.append(surface)
    return result


def _surface_values(item):
    value = item.get("stable_alias")
    if isinstance(value, dict):
        return [str(alias) for alias in value.values() if str(alias or "").strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(alias) for alias in value if str(alias or "").strip()]
    return [str(value)] if str(value or "").strip() else []


def new_equipment_retrieval_surfaces(
    linguistic_frame,
    *,
    user_message,
    engineering_context,
    has_accepted_layout,
    instance_directory=None,
    cached_grounding=None,
    retrieved_surfaces=None,
    confirmed_asset_choices=None,
    inventory_context_changed=False,
    previous_inventory_context=None,
):
    """Select only inventory that needs actual Engineering RAG retrieval.

    The labeled inventory is retrieved for an initial Generate. On later turns,
    accepted instance aliases and cached evidence are reused. Add/replace
    operations can request new evidence. Category names in evaluation
    assumptions never count as inventory.
    """

    frame = linguistic_frame if isinstance(linguistic_frame, dict) else {}
    message = str(user_message or "")
    folded_message = _folded(message)
    context = str(engineering_context or "")
    inventory = engineering_inventory_section(context)
    inventory_items = engineering_inventory_items(context)
    has_inventory_section = bool(inventory)

    candidates = []

    def add(surface):
        surface = str(surface or "").strip()
        if surface and _folded(surface) not in {_folded(value) for value in candidates}:
            candidates.append(surface)

    intent_type = str(frame.get("intent_type") or "").casefold()
    fresh_generate = intent_type in {
        "generate_feasible_layout",
        "optimize_layout_from_scratch",
        "generate",
        "optimize_from_scratch",
    }
    if not has_accepted_layout or inventory_context_changed or fresh_generate:
        for mention in frame.get("asset_mentions") or []:
            if not isinstance(mention, dict):
                continue
            surface = str(mention.get("surface_text") or "").strip()
            in_request = bool(surface and _contains_phrase(folded_message, surface))
            in_inventory = bool(
                surface
                and has_inventory_section
                and inventory_surface_item_indices(surface, inventory_items)
            )
            use_current_request = (
                not has_accepted_layout
                or inventory_context_changed
                or fresh_generate
            )
            if (use_current_request and in_request) or in_inventory or (
                not has_inventory_section and (not has_accepted_layout or fresh_generate) and surface
            ):
                add(surface)

    explicit_new_model_surfaces = new_model_operation_surfaces(frame, message)
    for surface in explicit_new_model_surfaces:
        add(surface)

    mention_quantities = {
        _folded(item.get("surface_text")): max(
            1, int(item.get("quantity") or 1)
        )
        for item in frame.get("asset_mentions") or []
        if isinstance(item, dict) and item.get("surface_text")
    }
    inventory_candidates = {
        key: set(inventory_surface_item_indices(surface, inventory_items))
        for surface in candidates
        if (key := _folded(surface))
    }
    candidates = [
        surface
        for surface in candidates
        if not any(
            other != _folded(surface)
            and inventory_candidates.get(_folded(surface))
            & inventory_candidates.get(other, set())
            and mention_quantities.get(other, 0)
            > mention_quantities.get(_folded(surface), 0)
            for other in inventory_candidates
        )
    ]

    accepted_aliases = {
        inventory_role_surface_key(alias)
        for item in instance_directory or []
        if isinstance(item, dict)
        for alias in _surface_values(item)
    }
    previous_inventory_items = (
        engineering_inventory_items(previous_inventory_context)
        if previous_inventory_context is not None
        else inventory_items
    )
    grounded_surfaces = {
        _inventory_surface_evidence_key(
            match.get("surface_text"),
            previous_inventory_items,
            quantity=match.get("quantity"),
        )
        for match in (cached_grounding or {}).get("asset_matches") or []
        if isinstance(match, dict) and match.get("surface_text")
    }
    retrieved = {
        _inventory_surface_evidence_key(surface, previous_inventory_items)
        for surface in retrieved_surfaces or []
    }
    confirmed_surfaces = set()
    confirmed_titles = set()
    for choice in confirmed_asset_choices or []:
        if not isinstance(choice, dict):
            continue
        if choice.get("surface_text"):
            confirmed_surfaces.add(
                inventory_role_surface_key(choice.get("surface_text"))
            )
        display_label = str(choice.get("display_label") or "")
        base_label = display_label.split(" (", 1)[0].strip()
        if base_label:
            confirmed_titles.add(inventory_role_surface_key(base_label))
    current_inventory_surfaces = {
        _inventory_surface_evidence_key(
            item.get("surface_text"),
            inventory_items,
            quantity=item.get("quantity"),
        )
        for item in frame.get("asset_mentions") or []
        if isinstance(item, dict)
        and item.get("surface_text")
        and inventory_context_changed
        and inventory_surface_changed(
            item.get("surface_text"), previous_inventory_context, context
        )
    }

    result = []
    explicit_new_model_keys = {
        _folded(surface) for surface in explicit_new_model_surfaces
    }
    mention_quantity_by_surface = {
        _folded(item.get("surface_text")): max(1, int(item.get("quantity") or 1))
        for item in frame.get("asset_mentions") or []
        if isinstance(item, dict) and item.get("surface_text")
    }
    for surface in candidates:
        exact_key = _folded(surface)
        role_key = inventory_role_surface_key(surface)
        current_key = _inventory_surface_evidence_key(
            surface,
            inventory_items,
            quantity=mention_quantity_by_surface.get(exact_key),
        )
        explicitly_confirmed = (
            role_key in confirmed_surfaces or role_key in confirmed_titles
        )
        force_new_model_retrieval = (
            exact_key in explicit_new_model_keys and not explicitly_confirmed
        )
        changed_inventory = current_key in current_inventory_surfaces
        if explicitly_confirmed and not changed_inventory:
            continue
        if force_new_model_retrieval or changed_inventory:
            result.append(surface)
            continue
        if role_key in accepted_aliases:
            continue
        if current_key in retrieved or current_key in grounded_surfaces:
            continue
        result.append(surface)
    return result
