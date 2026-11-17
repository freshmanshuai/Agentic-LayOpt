import copy
import hashlib
import json
import math


RECEIPT_SCHEMA_VERSION = "3.0"


class ViewportReceiptError(ValueError):
    pass


def canonical_json_bytes(value):
    """Return a Python/JavaScript-stable canonical JSON representation.

    JavaScript has one numeric type, so a browser round-trip converts ``2.0``
    to ``2``.  Integral finite floats are normalized before hashing to keep an
    otherwise identical frozen layout stable across that boundary.
    """

    try:
        payload = json.dumps(
            _normalize_json_numbers(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ViewportReceiptError(f"Value is not canonical JSON: {exc}") from exc
    return payload.encode("utf-8")


def _normalize_json_numbers(value):
    if isinstance(value, dict):
        return {str(key): _normalize_json_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_numbers(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ViewportReceiptError("Viewport identity contains a non-finite number.")
        if value == 0.0:
            return 0
        if value.is_integer():
            return int(value)
    return value


def layout_sha256(layout):
    if not isinstance(layout, dict):
        raise ViewportReceiptError("Layout snapshot must be an object.")
    return hashlib.sha256(canonical_json_bytes(layout)).hexdigest()


def expected_viewport_instances(layout):
    """Return movable and fixed instances that the viewport must account for."""

    expected = []
    for placement in layout.get("placements") or []:
        if not isinstance(placement, dict) or not placement.get("instance_id"):
            continue
        expected.append(
            {
                "instance_id": str(placement["instance_id"]),
                "model_asset_id": _optional_id(placement.get("model_asset_id")),
                "fixed": False,
            }
        )
    site = layout.get("site") or {}
    for item in site.get("fixed_equipment_instances") or []:
        if not isinstance(item, dict) or not item.get("instance_id"):
            continue
        expected.append(
            {
                "instance_id": str(item["instance_id"]),
                "model_asset_id": _optional_id(
                    item.get("model_asset_id") or item.get("source_model_asset_id")
                ),
                "fixed": True,
            }
        )
    duplicate_ids = _duplicates(item["instance_id"] for item in expected)
    if duplicate_ids:
        raise ViewportReceiptError(
            "Layout has duplicate viewport instance IDs: " + ", ".join(duplicate_ids)
        )
    return sorted(expected, key=lambda item: item["instance_id"])


def verify_viewport_receipt(
    layout,
    receipt,
    *,
    expected_turn_id=None,
    expected_layout_sha256=None,
    expected_render_token=None,
):
    """Verify a browser receipt against the exact layout snapshot sent to Three.js."""

    if not isinstance(layout, dict) or not layout.get("layout_id"):
        raise ViewportReceiptError("A layout with layout_id is required.")
    if not isinstance(receipt, dict):
        raise ViewportReceiptError("Viewport receipt must be an object.")

    expected_snapshot = copy.deepcopy(layout)
    expected_sha = expected_layout_sha256 or layout_sha256(expected_snapshot)
    if expected_sha != layout_sha256(expected_snapshot):
        raise ViewportReceiptError("Expected layout_sha256 does not match the layout snapshot.")

    received_snapshot = receipt.get("layout_snapshot")
    snapshot_matches = isinstance(received_snapshot, dict)
    if snapshot_matches:
        try:
            snapshot_matches = layout_sha256(received_snapshot) == expected_sha
        except ViewportReceiptError:
            snapshot_matches = False

    expected_layout_id = str(layout["layout_id"])
    received_layout_id = str(receipt.get("layout_id") or "")
    received_sha = str(receipt.get("layout_sha256") or "")
    received_turn_id = _integer_or_none(receipt.get("turn_id"))
    required_turn_id = (
        _integer_or_none(expected_turn_id)
        if expected_turn_id is not None
        else received_turn_id
    )
    turn_matches = required_turn_id is not None and received_turn_id == required_turn_id
    received_render_token = receipt.get("render_token")
    render_token_matches = (
        expected_render_token is None
        or (
            isinstance(received_render_token, str)
            and received_render_token == str(expected_render_token)
        )
    )

    expected_instances = expected_viewport_instances(layout)
    expected_by_id = {item["instance_id"]: item for item in expected_instances}
    records, invalid_record_ids = _normalize_instance_records(
        receipt.get("instance_records") or []
    )
    record_by_id = {item["instance_id"]: item for item in records}
    duplicate_record_ids = _duplicates(
        item.get("instance_id")
        for item in receipt.get("instance_records") or []
        if isinstance(item, dict)
    )

    expected_ids = sorted(expected_by_id)
    declared_expected_ids = _unique_ids(
        receipt.get("expected_instance_ids") or []
    )
    rendered_ids = _unique_ids(receipt.get("rendered_instance_ids") or [])
    declared_missing_ids = _unique_ids(
        receipt.get("missing_instance_ids") or []
    )
    failed_instance_ids = _unique_ids(receipt.get("failed_instance_ids") or [])
    failed_asset_ids = _unique_ids(receipt.get("failed_model_asset_ids") or [])
    record_ids = sorted(record_by_id)
    missing_ids = sorted(set(expected_ids) - set(rendered_ids))
    unexpected_ids = sorted(set(rendered_ids) - set(expected_ids))
    missing_record_ids = sorted(set(rendered_ids) - set(record_ids))
    unexpected_record_ids = sorted(set(record_ids) - set(expected_ids))
    unexpected_failed_ids = sorted(set(failed_instance_ids) - set(expected_ids))

    mismatched_model_ids = []
    mismatched_fixed_ids = []
    mismatched_render_source_ids = []
    invalid_geometry_ids = list(invalid_record_ids)
    invalid_visibility_ids = []
    for instance_id, record in record_by_id.items():
        expected = expected_by_id.get(instance_id)
        if expected is None:
            continue
        if record.get("model_asset_id") != expected.get("model_asset_id"):
            mismatched_model_ids.append(instance_id)
        if record.get("fixed") != expected.get("fixed"):
            mismatched_fixed_ids.append(instance_id)
        expected_source = (
            "model"
            if expected.get("model_asset_id")
            else "geometry_placeholder" if expected.get("fixed") else "model_required"
        )
        if record.get("render_source") != expected_source:
            mismatched_render_source_ids.append(instance_id)
        if (
            record.get("mesh_count", 0) <= 0
            or not record.get("finite_world_bounds")
            or not _finite_bounds(record.get("world_bounds"))
        ):
            invalid_geometry_ids.append(instance_id)
        if (
            record.get("frustum_intersection") is not True
            or not _positive_finite(
                record.get("screen_projection_area_px2")
            )
            or not _ratio_below_one(record.get("clipping_ratio"))
        ):
            invalid_visibility_ids.append(instance_id)

    renderer = str(receipt.get("renderer") or "unknown")
    schema_matches = str(receipt.get("schema_version") or "") == RECEIPT_SCHEMA_VERSION
    render_status = str(receipt.get("render_status") or "UNKNOWN").upper()
    browser = receipt.get("browser") if isinstance(receipt.get("browser"), dict) else {}
    webgl = receipt.get("webgl") if isinstance(receipt.get("webgl"), dict) else {}
    frame = receipt.get("frame") if isinstance(receipt.get("frame"), dict) else {}
    model_load_reports = (
        copy.deepcopy(receipt.get("model_load_reports"))
        if isinstance(receipt.get("model_load_reports"), list)
        else []
    )
    asset_load_failures = (
        copy.deepcopy(receipt.get("asset_load_failures"))
        if isinstance(receipt.get("asset_load_failures"), list)
        else None
    )
    failed_assets_from_failures = _unique_ids(
        item.get("model_asset_id")
        for item in asset_load_failures or []
        if isinstance(item, dict)
    )
    viewport_visibility = (
        copy.deepcopy(receipt.get("viewport_visibility"))
        if isinstance(receipt.get("viewport_visibility"), dict)
        else {}
    )
    visibility_intersecting = _unique_ids(
        viewport_visibility.get("intersecting_instance_ids") or []
    )
    visibility_non_intersecting = _unique_ids(
        viewport_visibility.get("non_intersecting_instance_ids") or []
    )
    visibility_summary_valid = (
        visibility_intersecting
        == sorted(
            instance_id
            for instance_id in expected_ids
            if instance_id not in set(invalid_visibility_ids)
        )
        and visibility_non_intersecting
        == sorted(set(expected_ids) - set(visibility_intersecting))
        and (
            not expected_ids
            or _positive_finite(
                viewport_visibility.get(
                    "total_screen_projection_area_px2"
                )
            )
        )
    )
    browser_recorded = bool(browser.get("user_agent"))
    webgl_valid = (
        bool(webgl.get("version"))
        and webgl.get("context_lost") is False
        and _integer_or_none(webgl.get("gl_error")) == 0
    )
    frame_valid = (
        (_integer_or_none(frame.get("width_px")) or 0) > 0
        and (_integer_or_none(frame.get("height_px")) or 0) > 0
        and (_integer_or_none(frame.get("nonzero_rgb_pixel_count")) or 0) > 0
        and (
            not expected_ids
            or (_integer_or_none(frame.get("distinct_rgb_sample_count")) or 0)
            >= 2
        )
        and _is_sha256(frame.get("pixel_sha256"))
        and _is_sha256(frame.get("screenshot_sha256"))
    )

    layout_matches = received_layout_id == expected_layout_id
    renderer_matches = renderer.casefold() == "threejs"
    sha_matches = received_sha == expected_sha
    accepted_render_statuses = {"RENDERED"}
    if not expected_ids:
        accepted_render_statuses.add("EMPTY")

    hard_mismatch = any(
        (
            not layout_matches,
            not schema_matches,
            not renderer_matches,
            not sha_matches,
            not snapshot_matches,
            not turn_matches,
            not render_token_matches,
            render_status not in accepted_render_statuses,
            declared_expected_ids != expected_ids,
            declared_missing_ids != missing_ids,
            bool(unexpected_ids),
            bool(unexpected_record_ids),
            bool(unexpected_failed_ids),
            bool(duplicate_record_ids),
            bool(mismatched_model_ids),
            bool(mismatched_fixed_ids),
            bool(mismatched_render_source_ids),
            bool(invalid_visibility_ids),
            asset_load_failures is None,
            failed_assets_from_failures != failed_asset_ids,
            not visibility_summary_valid,
            receipt.get("empty_frame") is not False and bool(expected_ids),
            not browser_recorded,
            not webgl_valid,
            not frame_valid,
        )
    )
    if hard_mismatch:
        status = "MISMATCH"
    elif any(
        (
            render_status == "PARTIAL",
            bool(missing_ids),
            bool(missing_record_ids),
            bool(failed_instance_ids),
            bool(failed_asset_ids),
            bool(invalid_geometry_ids),
        )
    ):
        status = "PARTIAL"
    else:
        status = "VERIFIED"

    identity_payload = {
        "layout_sha256": expected_sha,
        "turn_id": required_turn_id,
        "render_version": receipt.get("render_version"),
        "render_token": receipt.get("render_token"),
        "renderer": renderer,
        "schema_version_matches": schema_matches,
        "render_status": render_status,
        "instance_records": records,
        "failed_instance_ids": failed_instance_ids,
        "failed_model_asset_ids": failed_asset_ids,
        "asset_load_failures": asset_load_failures,
        "viewport_visibility": viewport_visibility,
        "model_load_reports": model_load_reports,
        "frame": frame,
    }
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_id": hashlib.sha256(canonical_json_bytes(identity_payload)).hexdigest(),
        "status": status,
        "schema_version_matches": schema_matches,
        "renderer": renderer,
        "renderer_matches": renderer_matches,
        "render_status": render_status,
        "layout_id": expected_layout_id,
        "received_layout_id": received_layout_id or None,
        "layout_id_matches": layout_matches,
        "layout_sha256": expected_sha,
        "received_layout_sha256": received_sha or None,
        "layout_sha256_matches": sha_matches,
        "layout_snapshot_matches": snapshot_matches,
        "layout_snapshot": expected_snapshot,
        "turn_id": required_turn_id,
        "received_turn_id": received_turn_id,
        "turn_id_matches": turn_matches,
        "render_version": receipt.get("render_version"),
        "render_token": receipt.get("render_token"),
        "expected_render_token": expected_render_token,
        "render_token_matches": render_token_matches,
        "expected_instance_count": len(expected_ids),
        "expected_instance_ids": expected_ids,
        "declared_expected_instance_ids": declared_expected_ids,
        "rendered_instance_count": len(rendered_ids),
        "rendered_instance_ids": rendered_ids,
        "missing_instance_ids": missing_ids,
        "declared_missing_instance_ids": declared_missing_ids,
        "unexpected_instance_ids": unexpected_ids,
        "missing_instance_record_ids": missing_record_ids,
        "unexpected_instance_record_ids": unexpected_record_ids,
        "duplicate_instance_record_ids": duplicate_record_ids,
        "unexpected_failed_instance_ids": unexpected_failed_ids,
        "mismatched_model_instance_ids": sorted(set(mismatched_model_ids)),
        "mismatched_fixed_instance_ids": sorted(set(mismatched_fixed_ids)),
        "mismatched_render_source_instance_ids": sorted(
            set(mismatched_render_source_ids)
        ),
        "invalid_geometry_instance_ids": sorted(set(invalid_geometry_ids)),
        "invalid_visibility_instance_ids": sorted(
            set(invalid_visibility_ids)
        ),
        "failed_instance_ids": failed_instance_ids,
        "failed_model_asset_ids": failed_asset_ids,
        "model_load_reports": model_load_reports,
        "asset_load_failures": asset_load_failures or [],
        "viewport_visibility": viewport_visibility,
        "visibility_summary_valid": visibility_summary_valid,
        "empty_frame": receipt.get("empty_frame"),
        "instance_records": records,
        "browser": copy.deepcopy(browser),
        "webgl": copy.deepcopy(webgl),
        "frame": copy.deepcopy(frame),
        "browser_recorded": browser_recorded,
        "webgl_valid": webgl_valid,
        "frame_valid": frame_valid,
    }


def _normalize_instance_records(values):
    records = []
    invalid_ids = []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict) or not item.get("instance_id"):
            invalid_ids.append("<missing_instance_id>")
            continue
        instance_id = str(item["instance_id"])
        try:
            mesh_count = int(item.get("mesh_count") or 0)
        except (TypeError, ValueError):
            mesh_count = 0
            invalid_ids.append(instance_id)
        records.append(
            {
                "instance_id": instance_id,
                "model_asset_id": _optional_id(item.get("model_asset_id")),
                "fixed": bool(item.get("fixed")),
                "render_source": str(item.get("render_source") or "model"),
                "mesh_count": mesh_count,
                "finite_world_bounds": bool(item.get("finite_world_bounds")),
                "world_bounds": copy.deepcopy(item.get("world_bounds")),
                "frustum_intersection": item.get(
                    "frustum_intersection"
                )
                is True,
                "screen_projection_area_px2": _float_or_none(
                    item.get("screen_projection_area_px2")
                ),
                "clipping_ratio": _float_or_none(
                    item.get("clipping_ratio")
                ),
            }
        )
    return sorted(records, key=lambda item: item["instance_id"]), invalid_ids


def _finite_bounds(bounds):
    if not isinstance(bounds, dict):
        return False
    values = []
    for endpoint in ("min", "max"):
        point = bounds.get(endpoint)
        if not isinstance(point, dict):
            return False
        values.extend(point.get(axis) for axis in ("x", "y", "z"))
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return False
    return all(math.isfinite(value) for value in numbers) and all(
        numbers[index] <= numbers[index + 3] for index in range(3)
    )


def _optional_id(value):
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def _integer_or_none(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_sha256(value):
    value = str(value or "")
    return len(value) == 64 and all(char in "0123456789abcdefABCDEF" for char in value)


def _float_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_finite(value):
    number = _float_or_none(value)
    return number is not None and number > 0.0


def _ratio_below_one(value):
    number = _float_or_none(value)
    return number is not None and 0.0 <= number < 1.0


def _duplicates(values):
    seen = set()
    duplicates = set()
    for value in values:
        if value is None:
            continue
        value = str(value)
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _unique_ids(values):
    return sorted(
        {
            str(value).strip()
            for value in values
            if value is not None and str(value).strip()
        }
    )
