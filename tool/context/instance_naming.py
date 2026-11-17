"""Names supplied by the user distinguish repeated copies of a catalog model."""
import re


def declaration_labels(surface, quantity):
    if quantity < 2:
        return ()
    text = str(surface or "").strip()
    explicit = re.search(r"\b(?:labeled|labelled)\s+(.+)$", text, re.I)
    if explicit:
        labels = [token.casefold() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", explicit.group(1)) if token.casefold() not in {"and", "or"}]
    else:
        # Bare single-letter labels, as in "two machining centers A and B".
        suffix = re.search(r"\b([A-Za-z](?:\s*(?:,|\band\b|\bor\b)\s*[A-Za-z])+)$", text)
        labels = re.findall(r"\b[A-Za-z]\b", suffix.group(1)) if suffix else []
        labels = [label.casefold() for label in labels]
    return tuple(labels) if len(labels) == quantity and len(set(labels)) == quantity else ()


def declaration_role(surface, quantity, local_index):
    text = re.sub(r"^\s*(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+", "", str(surface), flags=re.I)
    labels = declaration_labels(surface, quantity)
    if labels:
        text = re.sub(r"\s+(?:labeled|labelled)\s+.+$", "", text, flags=re.I)
        text = re.sub(r"\s+[A-Za-z](?:\s*(?:,|\band\b|\bor\b)\s*[A-Za-z])+$", "", text)
    text = re.sub(r"\b(centers|machines|workstations|tables|racks|cabinets|AGVs)$", lambda m: m.group(1)[:-1], text, flags=re.I)
    return text.strip() + (" " + labels[local_index-1].upper() if labels else "" if quantity == 1 else " " + str(local_index))
