"""Lightweight fail-closed checks before invoking the Three.js browser."""

from __future__ import annotations

import math


class ViewportPrecheck:
    """Check render inputs that do not require WebGL."""

    def run(
        self,
        layout,
        *,
        expected_instance_ids,
        expected_model_by_instance=None,
        available_model_asset_ids=None,
    ):
        placements = list((layout or {}).get("placements") or [])
        fixed_equipment = list(
            ((layout or {}).get("site") or {}).get(
                "fixed_equipment_instances"
            )
            or []
        )
        render_instances = [
            *((item, False) for item in placements),
            *((item, True) for item in fixed_equipment),
        ]
        expected = [str(item) for item in expected_instance_ids or []]
        actual = [
            str(item.get("instance_id"))
            for item, _fixed in render_instances
            if item.get("instance_id")
        ]
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        duplicates = sorted(
            item for item in set(actual) if actual.count(item) > 1
        )
        expected_models = expected_model_by_instance or {}
        available_models = (
            set(available_model_asset_ids)
            if available_model_asset_ids is not None
            else None
        )
        invalid_model_mappings = []
        invalid_positions = []
        camera_invisible = []
        records = []

        for placement, fixed in render_instances:
            instance_id = placement.get("instance_id")
            model_asset_id = placement.get("model_asset_id") or placement.get(
                "source_model_asset_id"
            )
            expected_model = expected_models.get(instance_id)
            if (
                not model_asset_id
                or (expected_model and model_asset_id != expected_model)
                or (
                    available_models is not None
                    and model_asset_id not in available_models
                )
            ):
                invalid_model_mappings.append(
                    {
                        "instance_id": instance_id,
                        "model_asset_id": model_asset_id,
                        "expected_model_asset_id": expected_model,
                    }
                )
            position = placement.get("position") or placement.get("pose") or {}
            if isinstance(position.get("position"), dict):
                position = position["position"]
            dimensions = placement.get("dimensions_m") or (
                placement.get("geometry") or {}
            ).get("dimensions_m") or {}
            finite_position = all(
                _finite(position.get(key)) for key in ("x_m", "z_m")
            )
            finite_dimensions = all(
                _positive_finite(dimensions.get(key))
                for key in ("width_m", "depth_m", "height_m")
            )
            if not finite_position or not finite_dimensions:
                invalid_positions.append(instance_id)
            footprint = placement.get("footprint") or _footprint_from_pose(
                position,
                dimensions,
            )
            projected_area = _estimated_projection_area(footprint)
            frustum_intersection = finite_position and finite_dimensions
            if not frustum_intersection or projected_area <= 0.0:
                camera_invisible.append(instance_id)
            records.append(
                {
                    "instance_id": instance_id,
                    "model_asset_id": model_asset_id,
                    "fixed": fixed,
                    "position_data_valid": finite_position,
                    "dimensions_valid": finite_dimensions,
                    "frustum_intersection_expected": frustum_intersection,
                    "estimated_projection_area": projected_area,
                }
            )

        errors = []
        if missing:
            errors.append({"code": "VIEWPORT_MISSING_INSTANCE", "instance_ids": missing})
        if unexpected:
            errors.append(
                {"code": "VIEWPORT_UNEXPECTED_INSTANCE", "instance_ids": unexpected}
            )
        if duplicates:
            errors.append(
                {"code": "VIEWPORT_DUPLICATE_INSTANCE", "instance_ids": duplicates}
            )
        if invalid_model_mappings:
            errors.append(
                {
                    "code": "VIEWPORT_MODEL_MAPPING_INVALID",
                    "instances": invalid_model_mappings,
                }
            )
        if invalid_positions:
            errors.append(
                {
                    "code": "VIEWPORT_POSITION_DATA_INVALID",
                    "instance_ids": sorted(item for item in invalid_positions if item),
                }
            )
        if camera_invisible:
            errors.append(
                {
                    "code": "VIEWPORT_CAMERA_VISIBILITY_INVALID",
                    "instance_ids": sorted(item for item in camera_invisible if item),
                }
            )
        return {
            "schema_version": "viewport-precheck-v1",
            "status": "VALID" if not errors else "INVALID",
            "valid": not errors,
            "expected_instance_ids": sorted(expected),
            "observed_instance_ids": sorted(actual),
            "missing_instance_ids": missing,
            "unexpected_instance_ids": unexpected,
            "duplicate_instance_ids": duplicates,
            "instance_records": records,
            "errors": errors,
        }


def _estimated_projection_area(footprint):
    try:
        width = float(footprint["max_x"]) - float(footprint["min_x"])
        depth = float(footprint["max_z"]) - float(footprint["min_z"])
        return max(0.0, width * depth)
    except (KeyError, TypeError, ValueError):
        return 0.0


def _footprint_from_pose(position, dimensions):
    try:
        x_m = float(position["x_m"])
        z_m = float(position["z_m"])
        width_m = float(dimensions["width_m"])
        depth_m = float(dimensions["depth_m"])
    except (KeyError, TypeError, ValueError):
        return {}
    return {
        "min_x": x_m - width_m / 2.0,
        "max_x": x_m + width_m / 2.0,
        "min_z": z_m - depth_m / 2.0,
        "max_z": z_m + depth_m / 2.0,
    }


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive_finite(value):
    return _finite(value) and float(value) > 0.0
