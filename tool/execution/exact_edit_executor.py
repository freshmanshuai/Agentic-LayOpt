"""Exact execution for user-authored existing-layout directives."""

from __future__ import annotations

import copy
import time
from pathlib import Path

from tool.validation.canonical_constraint_validator import CanonicalConstraintValidator
from tool.representations.layout_edit_ir import apply_layout_edit_ir, validate_layout_edit_ir
from tool.representations.task_contracts import validate_task_contract


class ExactEditExecutor:
    """Apply a DirectiveEdit verbatim and fail on hard-constraint conflict."""

    def __init__(self, adapter, schemas_dir):
        self.adapter = adapter
        self.schemas_dir = Path(schemas_dir)
        self.validator = CanonicalConstraintValidator()

    def execute(
        self,
        edit_ir,
        base_layout,
        base_flp_ir,
        canonical_spec,
        task_contract,
        process_relations=None,
        service_transfers=None,
    ):
        started = time.perf_counter()
        contract = copy.deepcopy(task_contract or {})
        edit_report = validate_layout_edit_ir(
            edit_ir,
            base_layout,
            self.adapter,
            self.schemas_dir,
        )
        if edit_report["status"] != "VALID_EDIT_IR" or not base_flp_ir:
            return self._result(
                status="INVALID_EDIT_CONTRACT",
                layout=None,
                flp_ir=None,
                layout_diff=None,
                edit_report=edit_report,
                contract=contract,
                changed_instance_ids=[],
                hard_validation=None,
                task_validation=None,
                started=started,
            )

        try:
            applied = apply_layout_edit_ir(
                edit_ir,
                base_layout,
                base_flp_ir,
                self.adapter,
            )
            if service_transfers:
                from tool.context.conversation_deltas import apply_service_duty_transfers
                apply_service_duty_transfers(base_flp_ir, applied["flp_ir"], service_transfers)
        except ValueError as exc:
            edit_report = copy.deepcopy(edit_report)
            edit_report["status"] = "INVALID_TASK"
            edit_report.setdefault("errors", []).append(
                {
                    "error_type": "edit_execution_conflict",
                    "message": str(exc),
                    "field_path": "operations",
                }
            )
            return self._result(
                status="INVALID_EDIT_CONTRACT",
                layout=None,
                flp_ir=None,
                layout_diff=None,
                edit_report=edit_report,
                contract=contract,
                changed_instance_ids=[],
                hard_validation=None,
                task_validation=None,
                started=started,
            )

        layout = applied["rough_layout"]
        if process_relations is not None:
            from tool.representations.flp_ir_layer import _compile_flow_relations
            unresolved=[]
            flows=_compile_flow_relations(applied["flp_ir"]["equipment_instances"],applied["flp_ir"].get("objectives") or [],
                applied["flp_ir"]["provenance"],process_relations=process_relations,
                site_features=((applied["flp_ir"].get("site_ref") or {}).get("site_snapshot") or {}).get("site_features") or [],
                unresolved_items=unresolved,allow_instance_id_endpoints=True)
            applied["flp_ir"]["flow_relations"]=flows
            applied["flp_ir"].setdefault("unresolved_items",[]).extend(unresolved)
        # Replacement/addition changes the geometry being checked; the old
        # instance map cannot validate the new inventory correctly.
        from tool.layout.site_constraint_compiler import CanonicalConstraintCompiler
        canonical_spec = CanonicalConstraintCompiler().compile(
            (applied["flp_ir"].get("site_ref") or {}).get("site_snapshot") or {},
            flp_ir=applied["flp_ir"], task_contract=contract, reference_layout=base_layout)
        changed = _changed_instance_ids(base_layout, layout)
        target_ids = set(contract.get("target_instance_ids") or [])
        unexpected_changes = sorted(set(changed) - target_ids)
        task_validation = validate_task_contract(layout, contract)
        if unexpected_changes:
            task_validation = copy.deepcopy(task_validation)
            task_validation["status"] = "INVALID"
            task_validation.setdefault("errors", []).append(
                {
                    "code": "EDIT_NON_TARGET_CHANGED",
                    "instance_ids": unexpected_changes,
                    "allowed_repair_operations": [],
                    "preserve_instance_ids": contract.get(
                        "preserve_instance_ids"
                    )
                    or [],
                }
            )
            task_validation["violation_count"] = len(
                task_validation["errors"]
            )
            task_validation["satisfaction"] = 0.0
        hard_validation = self.validator.validate(
            layout,
            canonical_spec,
            reference_layout=base_layout,
            expected_instance_ids=contract.get("expected_instance_ids"),
            preserve_instance_ids=contract.get("preserve_instance_ids"),
        )
        if task_validation["status"] != "VALID":
            status = "INVALID_EDIT_CONTRACT"
        elif applied["flp_ir"].get("unresolved_items"):
            status = "INVALID_EDIT_CONTRACT"
        elif hard_validation["status"] != "VALID":
            status = "EDIT_CONFLICT"
        else:
            status = "EDIT_SUCCESS"
        result = self._result(
            status=status,
            layout=layout,
            flp_ir=applied["flp_ir"],
            layout_diff=applied["layout_diff"],
            edit_report=edit_report,
            contract=contract,
            changed_instance_ids=changed,
            hard_validation=hard_validation,
            task_validation=task_validation,
            started=started,
        )
        result["canonical_constraint_spec"] = canonical_spec
        return result

    @staticmethod
    def _result(
        *,
        status,
        layout,
        flp_ir,
        layout_diff,
        edit_report,
        contract,
        changed_instance_ids,
        hard_validation,
        task_validation,
        started,
    ):
        return {
            "schema_version": "exact-edit-execution-v1",
            "status": status,
            "layout": layout,
            "flp_ir": flp_ir,
            "layout_diff": layout_diff,
            "edit_ir_report": edit_report,
            "hard_validation": hard_validation,
            "task_contract_validation": task_validation,
            "changed_instance_ids": list(changed_instance_ids),
            "deviated_from_user_directive": bool(
                task_validation
                and task_validation.get("status") != "VALID"
            ),
            "runtime_s": time.perf_counter() - started,
            "trace": {
                "execution_type": "exact_edit_execution",
                "trigger_source": "user_directive",
                "task_type": "DirectiveEdit",
                "movable_instance_ids": copy.deepcopy(
                    contract.get("movable_instance_ids") or []
                ),
                "preserve_instance_ids": copy.deepcopy(
                    contract.get("preserve_instance_ids") or []
                ),
                "changed_instance_ids": list(changed_instance_ids),
                "deviated_from_user_directive": bool(
                    task_validation
                    and task_validation.get("status") != "VALID"
                ),
                "final_status": status,
            },
        }


def _changed_instance_ids(before, after):
    before_by_id = {
        item.get("instance_id"): item
        for item in (before or {}).get("placements") or []
    }
    after_by_id = {
        item.get("instance_id"): item
        for item in (after or {}).get("placements") or []
    }
    return sorted(
        instance_id
        for instance_id in set(before_by_id) | set(after_by_id)
        if before_by_id.get(instance_id) != after_by_id.get(instance_id)
    )
