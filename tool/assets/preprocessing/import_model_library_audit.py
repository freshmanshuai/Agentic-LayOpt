"""Import audited library records into the live Engineering RAG dataset.

The importer is deliberately metadata-only. It does not build an embedding
index, copy runtime assets, or alter benchmark fixtures. Source geometry that
does not have an approved case calibration remains unavailable for planning.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from tool.assets.preprocessing.model_naming import normalize_asset_names


IMPORT_ID = "model-library-audit-20261015"
CALIBRATION_ID = "candidate-calibration-20261015"
MACHINE_CALIBRATION_ID = "machine-calibration-20261015"
LIBRARY_PREFIX = "产线模型库"
INDEPENDENT_STL_IDS = {"M102", "M103", "M104"}
MOVED_FBX_IDS = {"M059", "M062", "M063", "M070", "M071"}
PLANNING_FIELDS = {
    "planning_fields.ports": "Ports require manual annotation or consistent model node naming conventions.",
    "planning_fields.maintenance_clearance_m": "Maintenance clearance needs a rule library or manual annotation.",
    "planning_fields.safety_clearance_m": "Safety clearance needs a rule library or manual annotation.",
    "planning_fields.process_capabilities": "Process capabilities need authoritative equipment documentation or manual annotation.",
}


# These translations are restricted to the audited source filename and, where
# noted, the associated visual review. They deliberately make no claims about
# capability, tolerance, throughput, payload, or safety certification.
EXISTING_MACHINE_SPECS = {
    "M057": ("CNC machining center", "cnc_machining_center", "machining_equipment", "The filename identifies a CNC machining-center model."),
    "M058": ("Six-axis CNC machining center for plastic oil-bottle machining", "six_axis_cnc_machining_center", "machining_equipment", "The filename identifies a six-axis CNC machining-center model for plastic oil-bottle machining."),
    "M059": ("Horizontal milling machine", "horizontal_milling_machine", "machining_equipment", "The filename identifies a horizontal milling-machine model."),
    "M060": ("Haas CNC machining center", "haas_cnc_machining_center", "machining_equipment", "The filename identifies a Haas CNC machining-center model."),
    "M061": ("Micro machining center", "micro_machining_center", "machining_equipment", "The filename identifies a micro machining-center model."),
    "M062": ("Radial drilling machine", "radial_drilling_machine", "machining_equipment", "The filename identifies a radial drilling-machine model."),
    "M063": ("Enclosed CNC machine", "enclosed_cnc_machine", "machining_equipment", "The generic CNC filename and audited preview identify an enclosed CNC-machine model."),
    "M064": ("Turn-mill machining center", "turn_mill_machining_center", "machining_equipment", "The filename identifies a turn-mill machining-center model."),
    "M065": ("Vertical machining center 1100 x 900", "vertical_machining_center", "machining_equipment", "The filename identifies a vertical machining center labeled 1100 x 900."),
    "M066": ("Vertical machining center GA-V918C", "vertical_machining_center", "machining_equipment", "The filename identifies a vertical machining center labeled GA-V918C."),
    "M067": ("Bare vertical machining-center model", "vertical_machining_center_bare_model", "machining_equipment", "The filename identifies a bare-machine model of a vertical machining center."),
    "M068": ("Vertical four-axis CNC milling machine", "vertical_four_axis_cnc_milling_machine", "machining_equipment", "The filename identifies a vertical four-axis CNC milling-machine model."),
    "M069": ("High-speed vertical turn-mill machining center", "high_speed_vertical_turn_mill_machining_center", "machining_equipment", "The filename identifies a high-speed vertical turn-mill machining-center model."),
    "M070": ("Gantry CNC cutting machine", "gantry_cnc_cutting_machine", "machining_equipment", "The filename identifies a gantry CNC cutting-machine model."),
    "M071": ("Gantry milling machine", "gantry_milling_machine", "machining_equipment", "The filename identifies a gantry milling-machine model."),
}


# Legacy FBX records were indexed with their Chinese source labels but a generic
# ``fbx_scene`` identity.  These are conservative English recoveries from the
# audited filenames.  They improve retrieval and clarification wording without
# asserting capacity, accuracy, throughput, payload, or safety performance.
# Factory_Unit records are intentionally excluded: their independently verified
# identities, scopes, planning fields, and geometry remain untouched.
LEGACY_AUDITED_SPECS = {
    "M002": ("0.5M transfer station", "equipment", "material_handling", "transfer_station", "The filename identifies a 0.5M transfer-station model; it does not establish a physical dimension or operating capability."),
    "M023": ("Industrial kitbash scene", "composite_scene", "scene_assembly", "industrial_kitbash_scene", "The source filename identifies an industrial kitbash scene, not a verified standalone machine."),
    "M025": ("H5 chain automatic assembly line", "production_line", "assembly_line", "h5_chain_automatic_assembly_line", "The filename identifies an H5 chain-type automatic assembly-line model."),
    "M026": ("Mixmaster model", "composite_scene", "unknown", "mixmaster_model", "The source filename is Mixmaster; its industrial function is not verified."),
    "M027": ("SMD component test-and-sorting machine", "equipment", "electronics_test_equipment", "smd_component_test_sorting_machine", "The filename identifies an integrated SMD component testing-and-sorting machine model."),
    "M028": ("Semiconductor automatic taping machine", "equipment", "semiconductor_packaging_equipment", "semiconductor_automatic_taping_machine", "The filename identifies a semiconductor automatic taping-machine model."),
    "M029": ("Semiconductor cooling and thermal platform", "equipment", "semiconductor_process_equipment", "semiconductor_thermal_platform", "The filename identifies a semiconductor cooling and thermal-platform model."),
    "M030": ("Semiconductor component pad-printing and packaging machine", "equipment", "semiconductor_packaging_equipment", "semiconductor_pad_printing_packaging_machine", "The filename identifies a semiconductor component pad-printing and packaging-machine model."),
    "M031": ("Semiconductor heat-sink fastening machine", "equipment", "semiconductor_assembly_equipment", "semiconductor_heat_sink_fastening_machine", "The filename identifies a semiconductor heat-sink fastening-machine model."),
    "M032": ("Semiconductor wafer automatic labeling machine", "equipment", "semiconductor_packaging_equipment", "semiconductor_wafer_labeling_machine", "The filename identifies a semiconductor wafer automatic labeling-machine model."),
    "M033": ("Semiconductor inspection equipment", "equipment", "semiconductor_inspection_equipment", "semiconductor_inspection_equipment", "The filename identifies semiconductor inspection equipment; the inspection method is not specified."),
    "M034": ("Semiconductor integrated inspection, coding, taping, and sorting machine", "equipment", "semiconductor_test_sorting_equipment", "semiconductor_integrated_inspection_coding_taping_sorting_machine", "The filename identifies an integrated semiconductor inductance-inspection, inkjet-coding, taping, and sorting machine model."),
    "M035": ("Semiconductor circuit-board loading and unloading equipment", "equipment", "semiconductor_material_handling", "semiconductor_circuit_board_loading_unloading_equipment", "The filename identifies semiconductor circuit-board loading and unloading equipment."),
    "M036": ("Semiconductor integrated-equipment model", "equipment", "semiconductor_equipment", "semiconductor_integrated_equipment_model", "The filename identifies a semiconductor integrated-equipment model; its specific process function is not verified."),
    "M037": ("Four-tank wafer cleaning and etching machine", "equipment", "semiconductor_process_equipment", "four_tank_wafer_cleaning_etching_machine", "The filename identifies a four-tank wafer cleaning and etching-machine model."),
    "M038": ("Disc vacuum packaging machine", "equipment", "packaging_equipment", "disc_vacuum_packaging_machine", "The filename identifies a disc vacuum packaging-machine model."),
    "M039": ("Expansion machine", "equipment", "industrial_equipment", "expansion_machine", "The filename identifies an expansion-machine model; the particular expansion process is not specified."),
    "M040": ("Automatic die-attach forming assembly", "production_line", "semiconductor_assembly_line", "automatic_die_attach_forming_assembly", "The filename identifies an automatic die-attach forming assembly; it is retained as an assembly-level model rather than a verified standalone station."),
    "M041": ("Industrial factory showroom scene 015", "composite_scene", "workshop_scene", "industrial_factory_showroom_scene", "The filename identifies an industrial factory showroom scene, not a single layoutable machine."),
    "M043": ("Four-column hydraulic press", "equipment", "forming_equipment", "four_column_hydraulic_press", "The filename identifies a four-column hydraulic press model."),
    "M044": ("Fixture workbench", "equipment", "integration_workstation", "fixture_workbench", "The filename identifies a fixture-workbench model."),
    "M045": ("Installation platform", "equipment", "work_surface", "installation_platform", "The filename identifies an installation-platform model; its process capability is not specified."),
    "M046": ("Tool cabinet", "equipment", "storage", "tool_cabinet", "The filename identifies a tool-cabinet model."),
    "M047": ("Robot linear-slide workstation", "equipment", "robotic_workstation", "robot_linear_slide_workstation", "The filename identifies a robot linear-slide workstation model."),
    "M048": ("Equipment enclosure", "equipment", "industrial_enclosure", "equipment_enclosure", "The filename identifies an equipment enclosure or cabinet model, not a verified production station."),
    "M049": ("Hydraulic power unit", "equipment", "hydraulic_equipment", "hydraulic_power_unit", "The filename identifies a hydraulic power-unit model."),
    "M050": ("EDM forming machine", "equipment", "machining_equipment", "edm_forming_machine", "The filename identifies an electrical-discharge machining forming-machine model."),
    "M051": ("EDM drilling machine", "equipment", "machining_equipment", "edm_drilling_machine", "The filename identifies an electrical-discharge machining drilling-machine model."),
    "M052": ("Pipe cutting and beveling machine", "equipment", "fabrication_equipment", "pipe_cutting_beveling_machine", "The filename identifies a pipe cutting and beveling-machine model."),
    "M053": ("Rolling machine", "equipment", "forming_equipment", "rolling_machine", "The filename identifies a rolling-machine model."),
    "M054": ("Conical hopper", "equipment", "material_handling", "conical_hopper", "The filename identifies a conical hopper model."),
    "M055": ("Five-unit integrated server-cabinet group", "composite_scene", "data_center_scene", "server_cabinet_group_scene", "The filename identifies an integrated group of five server cabinets, not a single standalone cabinet."),
    "M056": ("Single server cabinet", "equipment", "data_center_equipment", "server_cabinet", "The filename identifies a single server-cabinet model."),
    "M106": ("PCB placing-machine assembly scene", "composite_scene", "electronics_assembly_scene", "pcb_placing_machine_assembly_scene", "The source file calls this a PCB placing-machine assembly; it is retained as an assembly-level model rather than a verified standalone production station."),
    "M108": ("Rotary assembly line", "production_line", "assembly_line", "rotary_assembly_line", "The filename identifies a rotary assembly-line model."),
    "M110": ("Steel-platform production-line layout scene", "composite_scene", "production_line_scene", "steel_platform_production_line_layout_scene", "The filename identifies a full production-line layout on a steel platform, not a single layoutable machine."),
}


NEW_ASSET_SPECS = {
    "M072": ("asset/glb/m072_agv", "Automated guided vehicle", "equipment", "material_handling", "automated_guided_vehicle", "The filename identifies an automated guided vehicle (AGV) model."),
    "M073": ("asset/glb/m073_energy_monitoring_gateway", "Energy monitoring gateway", "equipment", "electrical_monitoring", "energy_monitoring_gateway", "The filename identifies an energy-monitoring gateway device."),
    "M074": ("asset/glb/m074_communications_service_monitor", "Communications service monitor", "equipment", "electrical_monitoring", "communications_service_monitor", "The filename identifies a communications service-monitor device."),
    "M075": ("asset/glb/m075_keyboard_video_mouse_switch", "Keyboard-video-mouse switch", "composite_scene", "electronics_component", "keyboard_video_mouse_switch", "The filename identifies a keyboard-video-mouse switch component, not a production machine."),
    "M076": ("asset/glb/m076_uhf_tunable_filter", "UHF tunable filter", "composite_scene", "electronics_component", "uhf_tunable_filter", "The filename identifies a UHF tunable-filter component, not a production machine."),
    "M077": ("asset/glb/m077_vhf_tunable_filter", "VHF tunable filter", "composite_scene", "electronics_component", "vhf_tunable_filter", "The filename identifies a VHF tunable-filter component, not a production machine."),
    "M078": ("asset/glb/m078_tool_conveyor_line", "Tool conveyor-line component", "production_line", "line_component", "tool_conveyor_line_component", "The filename identifies a tool conveyor-line component; standalone machine function is not verified."),
    "M079": ("asset/glb/m079_conveyor_2", "Conveyor 2", "production_line", "conveyor", "conveyor_line_segment", "The filename identifies a conveyor-line segment."),
    "M080": ("asset/glb/m080_floor_surface", "Floor surface", "site", "site_component", "floor_surface", "The filename identifies a floor-surface model, not a machine or an authoritative site boundary."),
    "M081": ("asset/glb/m081_electronics_workshop_scene", "Electronics workshop scene", "composite_scene", "scene_assembly", "electronics_workshop_scene", "The filename identifies an assembled electronics-workshop scene, not a single layoutable machine."),
    "M082": ("asset/glb/m082_large_temperature_chamber", "Large temperature chamber", "equipment", "test_environment", "large_temperature_chamber", "The filename identifies a large temperature-chamber model."),
    "M083": ("asset/glb/m083_worker_figure_1", "Worker figure 1", "composite_scene", "human_figure", "worker_figure", "The filename identifies a human worker figure, not a machine."),
    "M084": ("asset/glb/m084_worker_figure_2", "Worker figure 2", "composite_scene", "human_figure", "worker_figure", "The filename identifies a human worker figure, not a machine."),
    "M085": ("asset/glb/m085_temporary_storage_visual_scene", "Temporary-storage visual scene", "composite_scene", "semantic_mismatch", "visual_transfer_mechanism_scene", "The filename says temporary storage rack, but the visual audit shows a robot or transfer mechanism; it is retained as a composite scene rather than a verified standalone rack or machine."),
    "M086": ("asset/glb/m086_flexible_test_workstation", "Flexible test workstation", "equipment", "test_station", "flexible_test_workstation", "The filename identifies a flexible test-workstation model."),
    "M087": ("asset/glb/m087_table", "Table", "equipment", "work_surface", "table", "The filename identifies a table or work-surface model; machining or test capability is not verified."),
    "M088": ("asset/glb/m088_small_temperature_chamber", "Small temperature chamber", "equipment", "test_environment", "small_temperature_chamber", "The filename identifies a small temperature-chamber model."),
    "M089": ("asset/glb/m089_laser_marking_station", "Laser marking station", "equipment", "marking_station", "laser_marking_station", "The filename identifies a laser-marking station model."),
    "M090": ("asset/glb/m090_material_model", "Material model", "composite_scene", "material_component", "material_model", "The filename identifies a material model, not processing equipment."),
    "M091": ("asset/glb/m091_belt_conveyor", "Belt conveyor", "production_line", "conveyor", "belt_conveyor_line", "The filename identifies a belt-conveyor line component."),
    "M092": ("asset/glb/m092_line_side_storage_unit", "Line-side storage unit", "equipment", "storage", "line_side_storage_unit", "The filename identifies a line-side storage-unit model."),
    "M093": ("asset/glb/m093_fixture_workstation", "Fixture workstation", "equipment", "integration_workstation", "fixture_workstation", "The filename identifies a fixture-workstation model."),
    "M094": ("asset/glb/m094_touchscreen", "Touchscreen", "composite_scene", "electronics_component", "touchscreen", "The filename identifies a touchscreen component, not a standalone production machine."),
    "M095": ("asset/glb/m095_unspecified_equipment", "Unspecified equipment model", "composite_scene", "unknown", "unspecified_equipment_model", "The source filename is only 'equipment 1'; a machine function is not identified."),
    "M096": ("asset/glb/m096_multi_object_rack_scene", "Multi-object rack scene", "composite_scene", "semantic_mismatch", "multi_object_rack_scene", "The filename says rack, but the visual audit shows a multi-object rack scene; it is not exposed as a verified standalone storage-rack choice."),
    "M097": ("asset/glb/m097_ultrasonic_welding_machine", "Ultrasonic welding machine", "equipment", "joining_equipment", "ultrasonic_welding_machine", "The filename identifies an ultrasonic welding-machine model."),
    "M098": ("asset/glb/m098_conveyor_line", "Conveyor line", "production_line", "conveyor", "conveyor_line", "The filename identifies a conveyor-line component."),
    "M099": ("asset/glb/m099_conveyor_line_2", "Conveyor line 2", "production_line", "conveyor", "conveyor_line", "The filename identifies a second conveyor-line component."),
    "M100": ("asset/glb/m100_conveyor_line_3", "Conveyor line 3", "production_line", "conveyor", "conveyor_line", "The filename identifies a third conveyor-line component."),
    "M101": ("asset/glb/m101_aluminum_profile_support", "Aluminum-profile support", "composite_scene", "support_structure", "aluminum_profile_support", "The filename identifies an aluminum-profile support structure, not a machine."),
    "M102": ("asset/stl/m102_worktable_child", "Worktable subunit", "equipment", "work_surface", "worktable_subunit", "The filename identifies a worktable subunit; machining or test capability is not verified."),
    "M103": ("asset/stl/m103_work_table", "Work table", "equipment", "work_surface", "work_table", "The filename identifies a work-table model."),
    "M104": ("asset/stl/m104_storage_rack", "Storage rack", "equipment", "storage", "storage_rack", "The filename identifies a storage-rack model."),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="rag_dataset")
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--geometry-inspection", required=True)
    parser.add_argument("--calibrations", required=True)
    parser.add_argument("--machine-calibrations")
    parser.add_argument("--baseline-assets")
    parser.add_argument("--report-path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = import_audited_library(
        Path(args.dataset_dir),
        Path(args.inventory),
        Path(args.geometry_inspection),
        Path(args.calibrations),
        machine_calibration_path=(
            Path(args.machine_calibrations)
            if args.machine_calibrations
            else None
        ),
        baseline_assets_path=(
            Path(args.baseline_assets)
            if args.baseline_assets
            else None
        ),
        report_path=Path(args.report_path) if args.report_path else None,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def import_audited_library(
    dataset_dir,
    inventory_path,
    geometry_inspection_path,
    calibration_path,
    *,
    machine_calibration_path=None,
    baseline_assets_path=None,
    report_path=None,
    dry_run=False,
    timestamp=None,
):
    """Apply an audited inventory import and return a validation summary."""

    timestamp = timestamp or _timestamp()
    inventory = _read_json(inventory_path)
    geometry_rows = _read_csv_by_id(geometry_inspection_path)
    calibrations = _read_json(calibration_path)
    machine_calibrations = (
        _read_json(machine_calibration_path)
        if machine_calibration_path
        else []
    )
    inventory_by_id = {item["id"]: item for item in inventory}
    calibration_by_id = _merged_calibrations(calibrations, machine_calibrations)
    baseline_by_id = {
        asset["model_asset_id"]: asset
        for asset in (
            _read_json(baseline_assets_path)
            if baseline_assets_path
            else []
        )
    }
    _validate_import_sources(inventory, geometry_rows, calibration_by_id)

    assets_path = dataset_dir / "model_assets.json"
    assets_jsonl_path = dataset_dir / "model_assets.jsonl"
    chunks_path = dataset_dir / "rag_chunks.jsonl"
    unresolved_path = dataset_dir / "unresolved_items.jsonl"
    manifest_path = dataset_dir / "build_manifest.json"
    assets = _read_json(assets_path)
    chunks = _read_jsonl(chunks_path)
    unresolved = _read_jsonl(unresolved_path)
    manifest = _read_json(manifest_path)
    _validate_existing_dataset(assets, chunks)

    assets_by_id = {asset["model_asset_id"]: asset for asset in assets}
    changed_asset_ids = set()
    moved_paths = []
    legacy_identity_repaired_asset_ids = []
    for audit_id in MOVED_FBX_IDS:
        row = inventory_by_id[audit_id]
        asset = assets_by_id[row["rag_id"]]
        previous_uri = _update_moved_source_path(asset, row, timestamp)
        moved_paths.append(
            {
                "audit_id": audit_id,
                "model_asset_id": asset["model_asset_id"],
                "previous_file_uri": previous_uri,
                "current_file_uri": _library_uri(row),
            }
        )
        changed_asset_ids.add(asset["model_asset_id"])

    for audit_id, spec in LEGACY_AUDITED_SPECS.items():
        row = inventory_by_id[audit_id]
        asset = assets_by_id[row["rag_id"]]
        _apply_identity(asset, row, spec, timestamp)
        changed_asset_ids.add(asset["model_asset_id"])
        legacy_identity_repaired_asset_ids.append(asset["model_asset_id"])

    for audit_id, spec in EXISTING_MACHINE_SPECS.items():
        row = inventory_by_id[audit_id]
        asset = assets_by_id[row["rag_id"]]
        _apply_identity(asset, row, spec, timestamp)
        changed_asset_ids.add(asset["model_asset_id"])

    added_asset_ids = []
    catalog_new_asset_ids = set()
    for audit_id, spec in NEW_ASSET_SPECS.items():
        row = inventory_by_id[audit_id]
        asset_id = spec[0]
        if asset_id in assets_by_id:
            asset = assets_by_id[asset_id]
        else:
            asset = _new_asset(row, geometry_rows[audit_id], spec, timestamp)
            assets.append(asset)
            assets_by_id[asset_id] = asset
            added_asset_ids.append(asset_id)
        _apply_identity(asset, row, spec[1:], timestamp)
        changed_asset_ids.add(asset_id)
        catalog_new_asset_ids.add(asset_id)

    calibrated_asset_ids = []
    for audit_id, calibration in calibration_by_id.items():
        row = inventory_by_id[audit_id]
        asset_id = calibration.get("old_rag_id") or NEW_ASSET_SPECS[audit_id][0]
        asset = assets_by_id.get(asset_id)
        if asset is None:
            raise ValueError(f"Calibration {audit_id} has no catalog asset {asset_id}")
        _apply_case_calibration(
            asset,
            row,
            calibration,
            timestamp,
            baseline_asset=baseline_by_id.get(asset_id),
            is_new_asset=asset_id in catalog_new_asset_ids,
        )
        calibrated_asset_ids.append(asset_id)
        changed_asset_ids.add(asset_id)

    assets = _ordered_assets(assets)
    changed_asset_ids.update(normalize_asset_names(assets))
    assets_by_id = {asset["model_asset_id"]: asset for asset in assets}
    chunks = _update_chunks(chunks, assets, changed_asset_ids, timestamp)
    unresolved = _updated_unresolved(assets, unresolved, catalog_new_asset_ids)
    manifest = _updated_manifest(
        manifest,
        assets,
        chunks,
        unresolved,
        added_asset_ids=added_asset_ids,
        catalog_new_asset_ids=catalog_new_asset_ids,
        calibrated_asset_ids=calibrated_asset_ids,
        moved_paths=moved_paths,
        timestamp=timestamp,
    )
    summary = validate_imported_dataset(
        assets,
        chunks,
        unresolved,
        manifest,
        dataset_dir / "schemas",
        calibration_by_id=calibration_by_id,
        inventory_by_id=inventory_by_id,
    )
    result = {
        "status": "OK",
        "asset_count": len(assets),
        "rag_chunk_count": len(chunks),
        "unresolved_item_count": len(unresolved),
        "added_asset_count": len(catalog_new_asset_ids),
        "added_glb_count": sum(asset_id.startswith("asset/glb/") for asset_id in catalog_new_asset_ids),
        "added_stl_count": sum(asset_id.startswith("asset/stl/") for asset_id in catalog_new_asset_ids),
        "newly_inserted_asset_count": len(added_asset_ids),
        "moved_fbx_path_count": len(moved_paths),
        "legacy_identity_repaired_asset_count": len(set(legacy_identity_repaired_asset_ids)),
        "legacy_identity_repaired_asset_ids": sorted(set(legacy_identity_repaired_asset_ids)),
        "calibrated_asset_count": len(set(calibrated_asset_ids)),
        "user_confirmed_calibrated_asset_count": sum(
            calibration.get("geometry_status") == "user_confirmed_case_calibration"
            for calibration in calibration_by_id.values()
        ),
        "batch_scale_calibrated_asset_count": sum(
            calibration.get("geometry_status") == "case_calibration_using_approved_batch_scale"
            for calibration in calibration_by_id.values()
        ),
        "uncalibrated_new_asset_count": len(catalog_new_asset_ids - set(calibrated_asset_ids)),
        "added_asset_ids": sorted(catalog_new_asset_ids),
        "calibrated_asset_ids": sorted(set(calibrated_asset_ids)),
        "moved_paths": moved_paths,
        "validation": summary,
    }
    if not dry_run:
        _write_json(assets_path, assets)
        _write_jsonl(assets_jsonl_path, assets)
        _write_jsonl(chunks_path, chunks)
        _write_jsonl(unresolved_path, unresolved)
        _write_json(manifest_path, manifest)
        if report_path:
            _write_report(report_path, result)
    return result


def validate_imported_dataset(
    assets,
    chunks,
    unresolved,
    manifest,
    schemas_dir,
    *,
    calibration_by_id,
    inventory_by_id,
):
    asset_ids = [asset["model_asset_id"] for asset in assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("model_asset_id values must be unique")
    chunk_asset_ids = [chunk["model_asset_id"] for chunk in chunks]
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    if set(asset_ids) != set(chunk_asset_ids) or len(chunk_asset_ids) != len(set(chunk_asset_ids)):
        raise ValueError("Every catalog asset must have exactly one RAG chunk")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("RAG chunk IDs must be unique")
    unresolved_ids = [item["item_id"] for item in unresolved]
    if len(unresolved_ids) != len(set(unresolved_ids)):
        raise ValueError("Unresolved item IDs must be unique")
    if manifest.get("asset_count") != len(assets) or manifest.get("rag_chunk_count") != len(chunks):
        raise ValueError("Manifest asset or chunk counts are inconsistent")
    if manifest.get("unresolved_item_count") != len(unresolved):
        raise ValueError("Manifest unresolved count is inconsistent")
    if manifest.get("asset_ids") != asset_ids:
        raise ValueError("Manifest asset IDs are inconsistent")

    validator = Draft202012Validator(
        _read_json(schemas_dir / "asset" / "model_engineering_asset.schema.json"),
        registry=_schema_registry(schemas_dir),
    )
    schema_errors = []
    for asset in assets:
        for error in validator.iter_errors(asset):
            schema_errors.append(f"{asset['model_asset_id']}: {error.message}")
    if schema_errors:
        raise ValueError("Asset schema validation failed: " + "; ".join(schema_errors[:8]))

    by_id = {asset["model_asset_id"]: asset for asset in assets}
    for audit_id, calibration in calibration_by_id.items():
        asset_id = calibration.get("old_rag_id") or NEW_ASSET_SPECS[audit_id][0]
        asset = by_id[asset_id]
        expected = calibration["after_candidate_m"]
        actual = (asset.get("geometry") or {}).get("dimensions_m") or {}
        for dimension in ("width_m", "depth_m", "height_m"):
            if abs(float(actual[dimension]) - float(expected[dimension])) > 1e-9:
                raise ValueError(f"Calibration mismatch for {asset_id}.{dimension}")
        calibration_metadata = ((asset.get("extraction") or {}).get("raw_metadata") or {}).get("other") or {}
        if "case_calibration_20261015" not in calibration_metadata:
            raise ValueError(f"Calibration provenance is missing for {asset_id}")

    for audit_id, spec in NEW_ASSET_SPECS.items():
        asset = by_id[spec[0]]
        if audit_id not in calibration_by_id:
            dimensions = (asset.get("geometry") or {}).get("dimensions_m") or {}
            if any(dimensions.get(field) is not None for field in ("width_m", "depth_m", "height_m")):
                raise ValueError(f"Uncalibrated asset {asset['model_asset_id']} has planning dimensions")
            item_id = f"unresolved/{asset['model_asset_id']}/geometry_scale"
            if item_id not in set(unresolved_ids):
                raise ValueError(f"Uncalibrated asset {asset['model_asset_id']} lacks geometry review")

    for audit_id in MOVED_FBX_IDS:
        row = inventory_by_id[audit_id]
        asset = by_id[row["rag_id"]]
        if (asset.get("model_files") or [{}])[0].get("file_uri") != _library_uri(row):
            raise ValueError(f"Moved source path was not applied for {asset['model_asset_id']}")
    return {
        "asset_schema_valid": True,
        "unique_asset_ids": len(asset_ids),
        "unique_chunk_ids": len(chunk_ids),
        "unique_unresolved_ids": len(unresolved_ids),
    }


def _validate_import_sources(inventory, geometry_rows, calibration_by_id):
    by_id = {item["id"]: item for item in inventory}
    if len(by_id) != len(inventory):
        raise ValueError("Audit inventory IDs must be unique")
    glb = [item for item in inventory if item.get("format", "").casefold() == ".glb"]
    if len(glb) != 30:
        raise ValueError(f"Expected 30 new GLB source records, found {len(glb)}")
    selected = set(item["id"] for item in glb) | INDEPENDENT_STL_IDS
    if selected != set(NEW_ASSET_SPECS):
        raise ValueError("New asset classification does not cover exactly the audited GLB and independent STL set")
    for audit_id in selected:
        if audit_id not in by_id or audit_id not in geometry_rows:
            raise ValueError(f"Audit row or geometry inspection is missing for {audit_id}")
    for audit_id in MOVED_FBX_IDS | set(EXISTING_MACHINE_SPECS):
        if audit_id not in by_id or not by_id[audit_id].get("rag_id"):
            raise ValueError(f"Existing catalog mapping is missing for {audit_id}")
    for audit_id, calibration in calibration_by_id.items():
        if audit_id not in by_id:
            raise ValueError(f"Calibration source {audit_id} is absent from inventory")
        if "after_candidate_m" not in calibration:
            raise ValueError(f"Calibration {audit_id} lacks effective dimensions")


def _validate_existing_dataset(assets, chunks):
    asset_ids = [asset.get("model_asset_id") for asset in assets]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("Existing dataset contains duplicate asset IDs")
    chunk_ids = [chunk.get("model_asset_id") for chunk in chunks]
    if set(asset_ids) != set(chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Existing dataset must have one RAG chunk per asset")


def _calibrations_by_audit_id(calibrations):
    result = {}
    for calibration in calibrations:
        audit_id = calibration.get("audit_id")
        if not audit_id or audit_id in result:
            raise ValueError("Calibration audit IDs must be present and unique")
        result[audit_id] = calibration
    return result


def _merged_calibrations(primary_calibrations, machine_calibrations):
    """Prefer individually approved role calibrations over batch-scale variants."""

    result = _calibrations_by_audit_id(primary_calibrations)
    for calibration in result.values():
        calibration["_calibration_source_id"] = CALIBRATION_ID
        calibration["geometry_status"] = "user_confirmed_case_calibration"
        calibration["dimension_claim_scope"] = (
            "case_calibration_not_manufacturer_specification"
        )
    for calibration in machine_calibrations:
        audit_id = calibration.get("audit_id")
        if not audit_id:
            raise ValueError("Machine calibration audit IDs must be present")
        if audit_id in result:
            continue
        normalized = copy.deepcopy(calibration)
        normalized["old_rag_id"] = normalized.get("model_asset_id")
        normalized["_calibration_source_id"] = MACHINE_CALIBRATION_ID
        normalized["geometry_status"] = "case_calibration_using_approved_batch_scale"
        normalized["dimension_claim_scope"] = (
            "case_calibration_not_manufacturer_specification"
        )
        result[audit_id] = normalized
    return result


def _new_asset(row, geometry_row, spec, timestamp):
    asset_id, english_name, asset_scope, category, equipment_type, function_summary = spec
    source_format = row["format"].lstrip(".").casefold()
    source_frame_id = f"{asset_id}/{source_format}_frame"
    native_bounds = _native_bounds(geometry_row)
    return {
        "schema_version": "0.2",
        "model_asset_id": asset_id,
        "asset_scope": asset_scope,
        "asset_name": _display_name(english_name, row["name"]),
        "asset_description": _description(function_summary),
        "library_id": "library/production_model_library",
        "site_id": None,
        "category": category,
        "equipment_type": equipment_type,
        "aliases": _unique([english_name, row["name"]]),
        "model_files": [
            {
                "source_file_id": f"{asset_id}/source/1",
                "file_uri": _library_uri(row),
                "asset_format": source_format,
                "native_extension": row["format"].casefold(),
                "file_role": "source_model",
                "checksum": None,
                "source_unit": "unknown",
                "file_status": "available",
                "metadata": {
                    "author": None,
                    "source_system": "model_library_audit_20261015",
                    "created_at": None,
                    "updated_at": timestamp,
                },
            }
        ],
        "derived_files": [],
        "extraction": {
            "extraction_status": "requires_manual_review",
            "extractor_name": "tool.assets.preprocessing.import_model_library_audit",
            "extractor_version": "1.0",
            "extraction_method": "scene_parser" if source_format == "glb" else "mesh_parser",
            "extracted_at": timestamp,
            "source_format": source_format,
            "source_unit": "unknown",
            "unit_scale_to_m": None,
            "axis_mapping": {
                "source_up_axis": "unknown",
                "target_up_axis": "unknown",
                "layout_x_from_source_axis": "unknown",
                "layout_y_from_source_axis": "unknown",
                "layout_z_from_source_axis": "unknown",
            },
            "extraction_quality": {
                "geometry_confidence": None,
                "scale_confidence": None,
                "origin_confidence": None,
                "port_confidence": 0.0,
                "site_feature_confidence": 0.0,
                "requires_manual_review": True,
                "warnings": [
                    "The audit recorded native scene extents, but no approved meter-scale calibration is available; planning geometry is withheld.",
                    "Identity is based on the source filename and visual audit only; process capacity, tolerance, throughput, and safety behavior are not verified.",
                ],
            },
            "raw_metadata": {
                "solidworks": None,
                "fbx": None,
                "three_ds_max": None,
                "c4d": None,
                "other": {
                    "source_inventory": _inventory_reference(row),
                    "audit_native_bounds": native_bounds,
                },
            },
        },
        "coordinate_frames": [
            _source_frame(source_frame_id, asset_scope),
        ],
        "geometry": _unknown_geometry(source_frame_id),
        "planning_fields": _empty_planning_fields(),
        "site_asset_fields": _empty_site_asset_fields(),
        "manual_overrides": {"has_override": False, "override_fields": []},
        "provenance": [
            _provenance("model_files", "model_file", f"inventory:{row['id']}", row["relative_path"], 1.0, "Primary source file from the 20261015 library audit."),
            _provenance("geometry", "model_extraction", f"geometry_inspection:{row['id']}", "loaded native extents", None, "Native extents were inspected, but their planning-meter scale is unresolved."),
            _provenance("asset_name", "model_file", f"inventory:{row['id']}", row["name"], 0.8, "English identity is a source-filename translation and broad visual classification only."),
        ],
        "metadata": {
            "version": "0.2.1",
            "status": "draft",
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    }


def _apply_identity(asset, row, spec, timestamp):
    if len(spec) == 5:
        english_name, asset_scope, category, equipment_type, function_summary = spec
    elif len(spec) == 4:
        english_name, equipment_type, category, function_summary = spec
        asset_scope = "equipment"
    else:
        raise ValueError(f"Invalid identity specification for {row['id']}")
    raw_metadata = ((asset.get("extraction") or {}).get("raw_metadata") or {})
    other = raw_metadata.get("other") if isinstance(raw_metadata.get("other"), dict) else {}
    other.setdefault(
        "original_extraction_before_case_calibration",
        {
            "source_format": (asset.get("extraction") or {}).get("source_format"),
            "source_unit": (asset.get("extraction") or {}).get("source_unit"),
            "unit_scale_to_m": (asset.get("extraction") or {}).get("unit_scale_to_m"),
            "axis_mapping": copy.deepcopy((asset.get("extraction") or {}).get("axis_mapping")),
        },
    )
    other.setdefault(
        "legacy_identity_before_model_library_import_20261015",
        {
            "asset_name": asset.get("asset_name"),
            "asset_description": asset.get("asset_description"),
            "equipment_type": asset.get("equipment_type"),
            "aliases": copy.deepcopy(asset.get("aliases") or []),
        },
    )
    raw_metadata["other"] = other
    asset.setdefault("extraction", {})["raw_metadata"] = raw_metadata
    asset["asset_name"] = _display_name(english_name, row["name"])
    asset["asset_description"] = _description(function_summary)
    asset["asset_scope"] = asset_scope
    asset["equipment_type"] = equipment_type
    asset["category"] = category
    asset["aliases"] = _unique([
        *(asset.get("aliases") or []),
        english_name,
        row["name"],
        equipment_type,
    ])
    _replace_provenance(
        asset,
        _provenance(
            "asset_name",
            "model_file",
            f"inventory:{row['id']}",
            row["relative_path"],
            0.8,
            "English identity is a source-filename translation and broad visual classification only.",
        ),
    )
    asset.setdefault("metadata", {})["updated_at"] = timestamp


def _update_moved_source_path(asset, row, timestamp):
    files = asset.get("model_files") or []
    if not files:
        raise ValueError(f"No source file is available for {asset['model_asset_id']}")
    previous_uri = files[0].get("file_uri")
    files[0]["file_uri"] = _library_uri(row)
    files[0]["file_status"] = "available"
    files[0].setdefault("metadata", {})["updated_at"] = timestamp
    raw_metadata = ((asset.get("extraction") or {}).get("raw_metadata") or {})
    other = raw_metadata.get("other") if isinstance(raw_metadata.get("other"), dict) else {}
    other["source_path_relocation_20261015"] = {
        "previous_file_uri": previous_uri,
        "current_file_uri": _library_uri(row),
        "audit_id": row["id"],
    }
    raw_metadata["other"] = other
    asset.setdefault("extraction", {})["raw_metadata"] = raw_metadata
    _replace_provenance(
        asset,
        _provenance(
            "model_files[0].file_uri",
            "model_file",
            f"inventory:{row['id']}",
            row["relative_path"],
            1.0,
            f"Source path relocated from {previous_uri} to the audited current library path.",
        ),
    )
    asset.setdefault("metadata", {})["updated_at"] = timestamp
    return previous_uri


def _apply_case_calibration(
    asset,
    row,
    calibration,
    timestamp,
    *,
    baseline_asset=None,
    is_new_asset=False,
):
    raw_metadata = ((asset.get("extraction") or {}).get("raw_metadata") or {})
    other = raw_metadata.get("other") if isinstance(raw_metadata.get("other"), dict) else {}
    existing_calibration = other.get("case_calibration_20261015")
    if baseline_asset is not None:
        original_geometry = copy.deepcopy(baseline_asset.get("geometry"))
    elif is_new_asset:
        # New audit records have no earlier catalog geometry in the preserved
        # baseline.  On a re-import, the current geometry is already the case
        # calibration, so it must not become its own claimed source geometry.
        original_geometry = _unknown_geometry(
            _source_frame_id(asset, row["format"].lstrip(".").casefold())
        )
    elif isinstance(existing_calibration, dict) and existing_calibration.get("original_geometry"):
        original_geometry = copy.deepcopy(existing_calibration["original_geometry"])
    else:
        original_geometry = copy.deepcopy(asset.get("geometry"))
    preserved_calibration = (
        {
            key: value
            for key, value in existing_calibration.items()
            if key in {"first_recorded_at", "original_geometry"}
        }
        if isinstance(existing_calibration, dict)
        else {}
    )
    other["case_calibration_20261015"] = {
        **preserved_calibration,
        "first_recorded_at": preserved_calibration.get("first_recorded_at", timestamp),
        "geometry_status": calibration.get("geometry_status"),
        "user_confirmed": calibration.get("geometry_status") == "user_confirmed_case_calibration",
        "not_manufacturer_dimension_truth": True,
        "audit_id": calibration["audit_id"],
        "key": calibration.get("key"),
        "label": calibration.get("label"),
        "source_relative": calibration.get("source_relative"),
        "basis_grade": calibration.get("basis_grade"),
        "basis": calibration.get("basis"),
        "calibration_batch": calibration.get("calibration_batch"),
        "uniform_scale_from_loaded_scene": calibration.get("uniform_scale_from_loaded_scene"),
        "rotation_x_deg": calibration.get("rotation_x_deg"),
        "loaded_scene_transform": {
            "uniform_scale": calibration.get("uniform_scale_from_loaded_scene"),
            "rotation_x_deg": calibration.get("rotation_x_deg"),
            "coordinate_normalization": calibration.get("coordinate_normalization"),
        },
        "before_loaded_native": copy.deepcopy(calibration.get("before_loaded_native")),
        "original_geometry": original_geometry,
        "effective_dimensions_m": copy.deepcopy(calibration["after_candidate_m"]),
    }
    raw_metadata["other"] = other
    extraction = asset.setdefault("extraction", {})
    extraction["raw_metadata"] = raw_metadata
    source_format = extraction.get("source_format") or row["format"].lstrip(".").casefold()
    extraction["extraction_status"] = "requires_manual_review"
    extraction.setdefault("source_format", source_format)
    extraction.setdefault("source_unit", "unknown")
    extraction.setdefault("unit_scale_to_m", None)
    extraction.setdefault(
        "axis_mapping",
        {
            "source_up_axis": "unknown",
            "target_up_axis": "unknown",
            "layout_x_from_source_axis": "unknown",
            "layout_y_from_source_axis": "unknown",
            "layout_z_from_source_axis": "unknown",
        },
    )
    quality = extraction.setdefault("extraction_quality", {})
    confidence = _calibration_confidence(calibration)
    quality.update(
        {
            "geometry_confidence": confidence,
            "scale_confidence": confidence,
            "origin_confidence": min(confidence, 0.6),
            "port_confidence": quality.get("port_confidence") or 0.0,
            "site_feature_confidence": quality.get("site_feature_confidence") or 0.0,
            "requires_manual_review": True,
        }
    )
    warnings = list(quality.get("warnings") or [])
    note = _calibration_note(calibration)
    if note not in warnings:
        warnings.append(note)
    quality["warnings"] = warnings

    source_frame_id = _source_frame_id(asset, source_format)
    calibration_frame_id = f"{asset['model_asset_id']}/case_calibration_20261015_frame"
    _ensure_source_frame(asset, source_frame_id)
    _ensure_calibration_frame(asset, calibration_frame_id, source_frame_id, calibration)
    asset["geometry"] = _calibrated_geometry(
        calibration["after_candidate_m"],
        calibration_frame_id,
        confidence,
    )
    overrides = asset.setdefault("manual_overrides", {"has_override": False, "override_fields": []})
    overrides["has_override"] = True
    override_fields = overrides.setdefault("override_fields", [])
    override_fields[:] = [
        item
        for item in override_fields
        if item.get("field_path") not in {"geometry", "extraction.unit_scale_to_m"}
    ]
    override_fields.extend(
        [
            {
                "field_path": "geometry",
                "override_reason": note,
                "overridden_by": calibration.get("_calibration_source_id", CALIBRATION_ID),
                "overridden_at": timestamp,
            },
            {
                "field_path": "extraction.unit_scale_to_m",
                "override_reason": "Calibration transform retained with the original geometry in raw metadata.",
                "overridden_by": calibration.get("_calibration_source_id", CALIBRATION_ID),
                "overridden_at": timestamp,
            },
        ]
    )
    _replace_provenance(
        asset,
        _provenance(
            "geometry",
            "manual_override",
            calibration.get("_calibration_source_id", CALIBRATION_ID),
            calibration.get("source_relative"),
            confidence,
            note,
        ),
    )
    asset.setdefault("metadata", {})["updated_at"] = timestamp


def _update_chunks(chunks, assets, changed_asset_ids, timestamp):
    by_asset = {chunk["model_asset_id"]: chunk for chunk in chunks}
    if len(by_asset) != len(chunks):
        raise ValueError("RAG chunk model asset IDs must be unique")
    next_number = max((_chunk_number(chunk["chunk_id"]) for chunk in chunks), default=0) + 1
    for asset in assets:
        asset_id = asset["model_asset_id"]
        if asset_id in by_asset and asset_id not in changed_asset_ids:
            continue
        chunk_id = by_asset[asset_id]["chunk_id"] if asset_id in by_asset else f"chunk/model_asset/{next_number:04d}"
        if asset_id not in by_asset:
            next_number += 1
        chunk = _asset_chunk(asset, chunk_id, timestamp)
        if asset_id in by_asset:
            index = next(index for index, item in enumerate(chunks) if item["model_asset_id"] == asset_id)
            chunks[index] = chunk
        else:
            chunks.append(chunk)
        by_asset[asset_id] = chunk
    return chunks


def _asset_chunk(asset, chunk_id, timestamp):
    geometry = asset.get("geometry") or {}
    dimensions = geometry.get("dimensions_m") or {}
    calibration = (
        (((asset.get("extraction") or {}).get("raw_metadata") or {}).get("other") or {}).get("case_calibration_20261015")
    )
    if all(dimensions.get(key) is not None for key in ("width_m", "depth_m", "height_m")):
        geometry_text = (
            f"effective planning geometry: {dimensions['width_m']}m x "
            f"{dimensions['depth_m']}m x {dimensions['height_m']}m"
        )
        if isinstance(calibration, dict):
            status = calibration.get("geometry_status")
            if status == "user_confirmed_case_calibration":
                geometry_text += "; user-confirmed case calibration, not manufacturer dimensional truth"
            elif status == "case_calibration_using_approved_batch_scale":
                geometry_text += "; case calibration using an approved batch scale, not manufacturer dimensional truth"
            else:
                geometry_text += "; case calibration, not manufacturer dimensional truth"
    else:
        geometry_text = "planning geometry unavailable because source-native scale remains uncalibrated"
    warnings = ((asset.get("extraction") or {}).get("extraction_quality") or {}).get("warnings") or []
    planning = asset.get("planning_fields") or {}
    retrieval_text = " ".join(
        [
            f"Asset {asset.get('asset_name')} ({asset['model_asset_id']}) is a {asset.get('asset_scope')} with category {asset.get('category')} and equipment_type {asset.get('equipment_type')}.",
            str(asset.get("asset_description") or ""),
            f"Aliases: {', '.join(asset.get('aliases') or [])}.",
            f"Source format: {(asset.get('model_files') or [{}])[0].get('asset_format')}; {geometry_text}.",
            f"Planning facts available: ports={len(planning.get('ports') or [])}, process_capabilities={len(planning.get('process_capabilities') or [])}.",
            "Warnings: " + " ".join(str(item) for item in warnings),
        ]
    ).strip()
    return {
        "chunk_id": chunk_id,
        "source_type": "model_asset",
        "model_asset_id": asset["model_asset_id"],
        "asset_name": asset.get("asset_name"),
        "chunk_type": "asset_summary",
        "retrieval_text": retrieval_text,
        "structured_facts": {
            "asset_scope": asset.get("asset_scope"),
            "category": asset.get("category"),
            "equipment_type": asset.get("equipment_type"),
            "aliases": copy.deepcopy(asset.get("aliases") or []),
            "geometry": copy.deepcopy(asset.get("geometry")),
            "planning_fields": copy.deepcopy(planning),
            "extraction": {
                "status": (asset.get("extraction") or {}).get("extraction_status"),
                "source_format": (asset.get("extraction") or {}).get("source_format"),
                "source_unit": (asset.get("extraction") or {}).get("source_unit"),
                "quality": copy.deepcopy((asset.get("extraction") or {}).get("extraction_quality") or {}),
            },
        },
        "provenance": copy.deepcopy(asset.get("provenance") or []),
        "created_at": timestamp,
    }


def _updated_unresolved(assets, existing, added_asset_ids):
    preserved = {
        item["item_id"]: item
        for item in existing
        if not _is_managed_unresolved(item)
    }
    for asset in assets:
        for field_path, description in PLANNING_FIELDS.items():
            if _is_missing(_get_path(asset, field_path)):
                item_id = f"unresolved/{asset['model_asset_id']}/{field_path.replace('.', '_')}"
                preserved[item_id] = {
                    "item_id": item_id,
                    "model_asset_id": asset["model_asset_id"],
                    "item_type": "missing_field",
                    "field_path": field_path,
                    "description": description,
                    "severity": "warning",
                    "candidates": [],
                    "resolution_policy": "manual_review",
                }
    for asset_id in added_asset_ids:
        asset = next(asset for asset in assets if asset["model_asset_id"] == asset_id)
        dimensions = (asset.get("geometry") or {}).get("dimensions_m") or {}
        if any(dimensions.get(key) is None for key in ("width_m", "depth_m", "height_m")):
            item_id = f"unresolved/{asset_id}/geometry_scale"
            preserved[item_id] = {
                "item_id": item_id,
                "model_asset_id": asset_id,
                "item_type": "extraction_low_confidence",
                "field_path": "geometry.dimensions_m",
                "description": "Source-native extents were inspected, but no approved meter-scale calibration is available for planning geometry.",
                "severity": "warning",
                "candidates": [],
                "resolution_policy": "manual_review",
            }
    return [preserved[item_id] for item_id in sorted(preserved)]


def _updated_manifest(manifest, assets, chunks, unresolved, *, added_asset_ids, catalog_new_asset_ids, calibrated_asset_ids, moved_paths, timestamp):
    result = copy.deepcopy(manifest)
    result["generated_at"] = timestamp
    result["asset_count"] = len(assets)
    result["rag_chunk_count"] = len(chunks)
    result["unresolved_item_count"] = len(unresolved)
    result["asset_ids"] = [asset["model_asset_id"] for asset in assets]
    result["model_library_import_20261015"] = {
        "inventory_source": "rag_dataset/sources/model_library_audit_20261015/inventory.json",
        "geometry_source": "rag_dataset/sources/model_library_audit_20261015/geometry_inspection.csv",
        "calibration_source": "rag_dataset/sources/candidate_calibration_20261015/calibration_results.json",
        "added_asset_ids": sorted(catalog_new_asset_ids),
        "newly_inserted_asset_ids_on_latest_run": list(added_asset_ids),
        "moved_fbx_paths": copy.deepcopy(moved_paths),
        "calibrated_asset_ids": sorted(set(calibrated_asset_ids)),
        "uncalibrated_new_asset_ids": sorted(catalog_new_asset_ids - set(calibrated_asset_ids)),
        "calibration_statement": "User-confirmed role calibrations and approved-batch-scale machine alternatives are case assumptions, not manufacturer dimension or capability claims.",
        "clearance_statement": "No new maintenance or safety clearance annotations were created by this import.",
    }
    return result


def _calibrated_geometry(candidate, frame_id, confidence):
    dimensions = {
        key: float(candidate[key])
        for key in ("width_m", "depth_m", "height_m")
    }
    minimum = candidate.get("min") or [
        -dimensions["width_m"] / 2.0,
        0.0,
        -dimensions["depth_m"] / 2.0,
    ]
    maximum = candidate.get("max") or [
        dimensions["width_m"] / 2.0,
        dimensions["height_m"],
        dimensions["depth_m"] / 2.0,
    ]
    return {
        "shape_type": "rectangle",
        "dimensions_m": dimensions,
        "footprint": {
            "shape_type": "rectangle",
            "width_m": dimensions["width_m"],
            "depth_m": dimensions["depth_m"],
            "source": "manual_input",
            "confidence": confidence,
        },
        "bounding_box_m": {
            "min": _vector(minimum),
            "max": _vector(maximum),
            "coordinate_frame_id": frame_id,
        },
        "origin_policy": "center_floor",
        "geometry_source": "manual_input",
        "confidence": confidence,
    }


def _unknown_geometry(frame_id):
    return {
        "shape_type": "rectangle",
        "dimensions_m": {"width_m": None, "depth_m": None, "height_m": None},
        "footprint": {
            "shape_type": "rectangle",
            "width_m": None,
            "depth_m": None,
            "source": "model_extraction",
            "confidence": None,
        },
        "bounding_box_m": {
            "min": _vector([None, None, None]),
            "max": _vector([None, None, None]),
            "coordinate_frame_id": frame_id,
        },
        "origin_policy": "unknown",
        "geometry_source": "model_extraction",
        "confidence": None,
    }


def _empty_planning_fields():
    return {
        "allowed_orientations_deg": [],
        "maintenance_clearance_m": _empty_clearance(),
        "safety_clearance_m": _empty_clearance(),
        "ports": [],
        "compatible_upstream_categories": [],
        "compatible_downstream_categories": [],
        "allowed_zone_types": [],
        "forbidden_zone_types": [],
        "process_capabilities": [],
    }


def _empty_clearance():
    return {
        "front_m": None,
        "rear_m": None,
        "left_m": None,
        "right_m": None,
        "top_m": None,
        "bottom_m": None,
        "source": "computed",
        "confidence": None,
    }


def _empty_site_asset_fields():
    return {
        "boundary": None,
        "extracted_site_features": [],
        "extracted_zones": [],
        "embedded_equipment_instances": [],
    }


def _source_frame(frame_id, asset_scope):
    return {
        "coordinate_frame_id": frame_id,
        "frame_type": "equipment_local" if asset_scope == "equipment" else "scene",
        "origin_description": "Source model origin; planning geometry is unresolved until a documented calibration is applied.",
        "axis_definition": {
            "x_axis": "source X axis",
            "y_axis": "source Y axis",
            "z_axis": "source Z axis",
        },
        "parent_frame_id": None,
        "transform_to_parent": None,
    }


def _ensure_source_frame(asset, frame_id):
    frames = asset.setdefault("coordinate_frames", [])
    if any(item.get("coordinate_frame_id") == frame_id for item in frames):
        return
    scope = asset.get("asset_scope")
    frames.append(_source_frame(frame_id, scope))


def _ensure_calibration_frame(asset, frame_id, source_frame_id, calibration):
    frames = asset.setdefault("coordinate_frames", [])
    frames[:] = [item for item in frames if item.get("coordinate_frame_id") != frame_id]
    frames.append(
        {
            "coordinate_frame_id": frame_id,
            "frame_type": "layout",
            "origin_description": "User-confirmed case-calibration frame for effective planning geometry; not a manufacturer reference frame.",
            "axis_definition": {
                "x_axis": "calibrated layout X axis",
                "y_axis": "calibrated layout Y axis",
                "z_axis": "calibrated layout Z axis",
            },
            "parent_frame_id": source_frame_id,
            "transform_to_parent": {
                "translation_m": _vector([0.0, 0.0, 0.0]),
                "rotation_deg": {
                    "roll_deg": float(calibration.get("rotation_x_deg") or 0.0),
                    "pitch_deg": 0.0,
                    "yaw_deg": 0.0,
                },
                "scale": float(calibration.get("uniform_scale_from_loaded_scene") or 1.0),
            },
        }
    )


def _source_frame_id(asset, source_format):
    frames = asset.get("coordinate_frames") or []
    if frames:
        return frames[0].get("coordinate_frame_id") or f"{asset['model_asset_id']}/{source_format}_frame"
    return f"{asset['model_asset_id']}/{source_format}_frame"


def _native_bounds(row):
    values = [
        _optional_float(row.get("loaded_X_native")),
        _optional_float(row.get("loaded_Y_native")),
        _optional_float(row.get("loaded_Z_native")),
    ]
    return {
        "axes": {"x": values[0], "y": values[1], "z": values[2]},
        "note": "Loaded native scene extents from geometry_inspection.csv; they are not asserted to be meters.",
    }


def _inventory_reference(row):
    return {
        "audit_id": row["id"],
        "relative_path": row["relative_path"],
        "source_name": row["name"],
        "format": row["format"],
        "file_size_bytes": row.get("bytes"),
    }


def _description(function_summary):
    return (
        f"{function_summary} This is an identity-level source-name and visual-audit description only; "
        "the source does not verify capacity, tolerance, throughput, payload, or safety performance."
    )


def _display_name(english_name, native_name):
    return english_name


def _library_uri(row):
    relative_path = str(row["relative_path"]).replace("\\", "/")
    return f"{LIBRARY_PREFIX}/{relative_path}"


def _ordered_assets(assets):
    # Existing IDs keep their source order. New records were appended in audit order.
    return list(assets)


def _is_managed_unresolved(item):
    item_id = str(item.get("item_id") or "")
    field_path = item.get("field_path")
    return (
        item.get("item_type") == "missing_field"
        and field_path in PLANNING_FIELDS
        and item_id.startswith("unresolved/")
    ) or item_id.endswith("/geometry_scale") and item_id.startswith("unresolved/")


def _is_missing(value):
    if value in (None, [], {}):
        return True
    if isinstance(value, dict):
        return not any(
            item is not None
            for key, item in value.items()
            if key not in {"source", "confidence"}
        )
    return False


def _get_path(value, path):
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _replace_provenance(asset, item):
    provenance = asset.setdefault("provenance", [])
    calibration_source_ids = {CALIBRATION_ID, MACHINE_CALIBRATION_ID}

    def replaces(current):
        if current.get("field_path") != item["field_path"]:
            return False
        if current.get("source_id") == item["source_id"]:
            return True
        return (
            item.get("field_path") == "geometry"
            and item.get("source_type") == "manual_override"
            and current.get("source_type") == "manual_override"
            and item.get("source_id") in calibration_source_ids
            and current.get("source_id") in calibration_source_ids
        )

    provenance[:] = [
        current
        for current in provenance
        if not replaces(current)
    ]
    provenance.append(item)


def _provenance(field_path, source_type, source_id, source_field, confidence, note):
    return {
        "field_path": field_path,
        "source_type": source_type,
        "source_id": source_id,
        "source_field": source_field,
        "confidence": confidence,
        "note": note,
    }


def _calibration_confidence(calibration):
    if calibration.get("geometry_status") == "case_calibration_using_approved_batch_scale":
        return 0.55
    grade = calibration.get("basis_grade")
    return {"A": 0.9, "B": 0.75, "P": 0.6}.get(str(grade or "").upper(), 0.6)


def _calibration_note(calibration):
    if calibration.get("geometry_status") == "case_calibration_using_approved_batch_scale":
        return (
            "Case calibration uses an approved batch-scale interpretation; it is "
            "not an individual user confirmation, manufacturer dimension, or safety specification."
        )
    return (
        "User-confirmed case calibration supplies effective planning dimensions; it "
        "is not a manufacturer dimension or safety specification."
    )


def _chunk_number(chunk_id):
    try:
        return int(str(chunk_id).rsplit("/", 1)[-1])
    except (TypeError, ValueError):
        return 0


def _vector(values):
    return {"x_m": values[0], "y_m": values[1], "z_m": values[2]}


def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _unique(values):
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _schema_registry(schemas_dir):
    resources = []
    for schema_path in Path(schemas_dir).rglob("*.json"):
        schema = _read_json(schema_path)
        resource = Resource.from_contents(schema)
        resources.append((schema_path.relative_to(schemas_dir).as_posix(), resource))
        if "$id" in schema:
            resources.append((schema["$id"], resource))
    return Registry().with_resources(resources)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv_by_id(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["id"]: row for row in rows}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_report(path, result):
    lines = [
        "# Model-library import",
        "",
        f"- Assets: {result['asset_count']}",
        f"- RAG chunks: {result['rag_chunk_count']}",
        f"- Unresolved items: {result['unresolved_item_count']}",
        f"- Catalog additions relative to the preserved 72-asset baseline: {result['added_asset_count']} ({result['added_glb_count']} GLB and {result['added_stl_count']} STL)",
        f"- Newly inserted on the latest idempotent importer invocation: {result['newly_inserted_asset_count']}",
        f"- Relocated FBX source paths: {result['moved_fbx_path_count']}",
        f"- Legacy source-name identities recovered for English retrieval: {result['legacy_identity_repaired_asset_count']}",
        f"- Calibrated assets applied: {result['calibrated_asset_count']}",
        f"- Individually user-confirmed calibrations: {result['user_confirmed_calibrated_asset_count']}",
        f"- Approved-batch-scale machine alternatives: {result['batch_scale_calibrated_asset_count']}",
        f"- New assets with planning geometry withheld pending calibration: {result['uncalibrated_new_asset_count']}",
        "",
        "Effective dimensions are case assumptions, not manufacturer dimensions or capability claims. No new clearance annotations were invented.",
        "",
        "## Calibrated asset IDs",
        "",
        *[f"- `{asset_id}`" for asset_id in result["calibrated_asset_ids"]],
        "",
        "## Relocated source paths",
        "",
        *[
            f"- `{item['model_asset_id']}`: `{item['current_file_uri']}`"
            for item in result["moved_paths"]
        ],
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
