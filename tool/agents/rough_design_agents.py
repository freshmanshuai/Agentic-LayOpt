import copy
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from tool.agents.agent_contracts import CONTRACT_VERSION, contract_schema, enforce_agent_output
from tool.context.authoritative_context import AuthoritativeContextResolver
from tool.validation.canonical_constraint_validator import CanonicalConstraintValidator
from tool.execution.exact_edit_executor import ExactEditExecutor
from tool.representations.flp_ir_layer import (
    AssetReferenceAdapter,
    CONSTRAINT_SYNONYMS,
    OBJECTIVE_SYNONYMS,
    build_evidence_package,
    canonical_objective_type_from_literal,
    compile_flp_ir,
    normalize_flp_ir,
    repair_or_clarify,
    route_intent,
    validate_flp_ir,
)
from tool.layout.site_constraint_compiler import CanonicalConstraintCompiler
from tool.representations.task_contracts import TaskContractCompiler, validate_task_contract
from tool.execution.validation_driven_recovery import ValidationDrivenRecovery
from tool.execution.violation_resolution_executor import ViolationResolutionExecutor
from tool.validation.viewport_precheck import ViewportPrecheck
from tool.clients.llm_clients import IntentLLMClient, RAGBlockedError
from tool.representations.layout_edit_ir import (
    compile_layout_edit_ir,
    validate_layout_edit_ir,
)
from tool.layout.natural_site_requirements import (
    apply_central_aisle_width_update,
    canonical_natural_site_constraint_type,
    natural_site_literal_spans,
    quantity_insensitive_surface_equivalent,
)
from tool.layout.layout_solvers import (
    DEFAULT_SOLVER_NAME,
    GEOMETRY_EPSILON_M,
    annotate_solver_selection,
    high_flow_distance_excess,
    layout_violations,
    normalize_solver_name,
    objective_value,
    objective_vector,
    select_solver,
    solve_canonical_feasibility,
)
from tool.context.instance_naming import declaration_labels
from tool.logistics.facility_logistics import facility_role, declared_facility_roles, resolve_facility_ref
from tool.logistics.relocation_effort import RELOCATION_EFFORT_OBJECTIVE, relocation_effort
from tool.logistics.transfer_workcell import interface_questions
from tool.context.session_engineering_context import (
    engineering_context_source_history,
    engineering_inventory_declarations,
    engineering_inventory_items,
    new_model_operation_surfaces,
    new_equipment_retrieval_surfaces,
    persist_engineering_context,
    reusable_confirmed_asset_choices,
)
from tool.validation.viewport_receipts import (
    ViewportReceiptError,
    layout_sha256,
    verify_viewport_receipt,
)


SUPPORTED_DESIGN_INTENTS = {
    "generate_feasible_layout",
    "optimize_layout_from_scratch",
    "optimize_existing_layout",
}
EXISTING_LAYOUT_TRANSFORMATION_INTENTS = {
    "edit_layout",
    "constraint_repair",
}
INSPECTION_INTENTS = {"validate_layout", "evaluate_layout"}
MAX_AGENTIC_ITERATIONS = 3
MAX_SESSION_HISTORY_TURNS = 6
VALID_LAYOUT_STATUSES = {"VALID_SOLVER_READY", "VALID_WITH_WARNINGS"}


class IntentUnderstandingAgent:
    name = "IntentUnderstandingAgent"
    responsibility = (
        "Extract linguistic layout slots, bind asset identity through Engineering RAG, "
        "and revise the semantic contract from Validator feedback without generating geometry."
    )

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def run(self, user_message, active_layout_id=None, scene_context=None):
        scene_context = AuthoritativeContextResolver().apply(
            user_message,
            scene_context or {},
        )
        if self.llm_client is None:
            raise RAGBlockedError(
                "intent_llm_unavailable",
                "IntentUnderstandingAgent requires an LLM client; rule fallback is disabled.",
                details={"fallback_enabled": False},
            )
        try:
            engineering_context = str(scene_context.get("engineering_context") or "")
            engineering_context_kwargs = (
                {"engineering_context": engineering_context}
                if engineering_context
                else {}
            )
            source_history = engineering_context_source_history(
                scene_context.get("conversation_history"), engineering_context
            )
            instance_directory = scene_context.get("planned_edit_instance_directory") or _current_instance_directory(scene_context)
            linguistic_frame = scene_context.get("linguistic_frame")
            if linguistic_frame is None:
                slot_extractor = getattr(
                    self.llm_client,
                    "extract_layout_slots",
                    None,
                )
                if callable(slot_extractor):
                    linguistic_frame = slot_extractor(
                        user_message,
                        active_layout_id=active_layout_id,
                        conversation_history=scene_context.get(
                            "conversation_history"
                        )
                        or [],
                        instance_directory=instance_directory,
                        **engineering_context_kwargs,
                    )
                else:
                    linguistic_frame = self.llm_client.extract_layout_intent(
                        user_message,
                        active_layout_id=active_layout_id,
                    )
            linguistic_frame = _normalize_linguistic_frame(
                linguistic_frame,
                user_message,
                conversation_history=source_history,
                site_features=_known_site_features(scene_context),
            )
            rag_grounding = scene_context.get("engineering_rag_grounding")
            if rag_grounding and instance_directory and linguistic_frame.get("intent_type") in {
                "edit", "edit_layout", "optimize", "optimize_existing_layout",
                "repair", "constraint_repair",
            }:
                protected_surfaces = _unique_surface_texts(
                    [
                        *(scene_context.get("fresh_rag_asset_surfaces") or []),
                        *new_model_operation_surfaces(linguistic_frame, user_message),
                        *[
                            choice.get("surface_text")
                            for choice in scene_context.get("confirmed_asset_choices") or []
                            if isinstance(choice, dict)
                        ],
                    ]
                )
                rag_grounding = _bind_existing_role_grounding(
                    rag_grounding,
                    instance_directory,
                    protected_surfaces=protected_surfaces,
                )
                scene_context["engineering_rag_grounding"] = rag_grounding
            grounded_extractor = getattr(
                self.llm_client, "extract_layout_intent_with_rag", None
            )
            if rag_grounding and not callable(grounded_extractor):
                raise RAGBlockedError(
                    "rag_grounded_intent_client_unsupported",
                    "The configured intent client cannot consume Engineering RAG grounding.",
                )
            if rag_grounding:
                extractor_kwargs = {"active_layout_id": active_layout_id}
                if instance_directory:
                    extractor_kwargs["instance_directory"] = instance_directory
                extractor_kwargs["linguistic_frame"] = linguistic_frame
                extractor_kwargs.update(engineering_context_kwargs)
                llm_frame = grounded_extractor(
                    user_message,
                    rag_grounding,
                    **extractor_kwargs,
                )
                llm_frame, deterministic_repairs = _repair_grounded_intent_frame(
                    llm_frame,
                    user_message,
                    rag_grounding,
                    linguistic_frame,
                    instance_directory=instance_directory,
                    conversation_history=source_history,
                )
                intent_contract = _validate_grounded_intent_frame(
                    llm_frame,
                    user_message,
                    rag_grounding,
                    instance_directory=instance_directory,
                    conversation_history=source_history,
                )
            else:
                llm_frame = self.llm_client.extract_layout_intent(
                    user_message,
                    active_layout_id=active_layout_id,
                )
                deterministic_repairs = []
                intent_contract = _valid_intent_contract()
        except RAGBlockedError:
            raise
        except Exception as exc:
            raise RAGBlockedError(
                "intent_llm_api_error",
                "Intent LLM failed and rule fallback is disabled.",
                details={"fallback_enabled": False, "exception_type": type(exc).__name__},
            ) from exc
        payload = _semantic_intent_step(
            llm_frame,
            user_message,
            active_layout_id=active_layout_id,
            scene_context=scene_context,
            agent_name=self.name,
            responsibility=self.responsibility,
            output_artifact="intent_v1",
            linguistic_frame=linguistic_frame,
            intent_contract=intent_contract,
            deterministic_repairs=deterministic_repairs,
            revision=1,
        )
        return enforce_agent_output(self.name, payload)

    def revise(
        self,
        user_message,
        previous_intent_step,
        validator_feedback,
        *,
        active_layout_id=None,
        scene_context=None,
        revision,
    ):
        scene_context = AuthoritativeContextResolver().apply(user_message, scene_context or {})
        reviser = getattr(self.llm_client, "revise_layout_intent", None)
        if not callable(reviser):
            raise RAGBlockedError(
                "intent_revision_client_unsupported",
                "The configured intent client cannot execute Validator-guided revisions.",
            )
        rag_grounding = scene_context.get("engineering_rag_grounding") or {}
        linguistic_frame = previous_intent_step.get("linguistic_frame") or {}
        engineering_context = str(scene_context.get("engineering_context") or "")
        source_history = engineering_context_source_history(
            scene_context.get("conversation_history"), engineering_context
        )
        engineering_context_kwargs = (
            {"engineering_context": engineering_context}
            if engineering_context
            else {}
        )
        instance_directory = scene_context.get("planned_edit_instance_directory") or _current_instance_directory(scene_context)
        llm_frame = reviser(
            user_message,
            previous_intent_step.get("semantic_intent") or {},
            validator_feedback,
            rag_grounding,
            linguistic_frame=linguistic_frame,
            active_layout_id=active_layout_id,
            instance_directory=instance_directory,
            **engineering_context_kwargs,
        )
        llm_frame, deterministic_repairs = _repair_grounded_intent_frame(
            llm_frame,
            user_message,
            rag_grounding,
            linguistic_frame,
            instance_directory=instance_directory,
            conversation_history=source_history,
        )
        intent_contract = _validate_grounded_intent_frame(
            llm_frame,
            user_message,
            rag_grounding,
            instance_directory=instance_directory,
            conversation_history=source_history,
        )
        payload = _semantic_intent_step(
            llm_frame,
            user_message,
            active_layout_id=active_layout_id,
            scene_context=scene_context,
            agent_name=self.name,
            responsibility=self.responsibility,
            output_artifact=f"intent_v{revision}",
            linguistic_frame=linguistic_frame,
            intent_contract=intent_contract,
            deterministic_repairs=deterministic_repairs,
            revision=revision,
        )
        payload["parent_intent_revision_id"] = (
            previous_intent_step.get("intent_revision_id") or "intent_v1"
        )
        payload["intent_revision_id"] = f"intent_v{revision}"
        return enforce_agent_output(self.name, payload)


class DesignerAgent:
    name = "DesignerAgent"
    responsibility = "Compile FLP-IR, call the selected solver, and apply structured validation or evaluation feedback."

    def __init__(self, dataset_dir, rag_service=None):
        self.dataset_dir = Path(dataset_dir)
        self.adapter = AssetReferenceAdapter(self.dataset_dir, rag_service=rag_service)

    def run(
        self,
        user_message,
        intent,
        scene_context=None,
        semantic_intent=None,
        *,
        revision=1,
        feedback=None,
    ):
        user_input = scene_context.get("current_user_input") if scene_context else None
        scene_context = AuthoritativeContextResolver().apply(
            user_input or user_message,
            scene_context or {},
        )
        task_contract = TaskContractCompiler().compile(
            intent,
            semantic_intent or {},
            scene_context["authoritative_context"],
        ).to_dict()
        scene_context["task_contract"] = task_contract
        if task_contract["task_type"] == "ConstraintRepair":
            return self._run_constraint_repair(
                intent,
                scene_context,
                semantic_intent or {},
                revision=revision,
                feedback=feedback,
            )
        if task_contract["task_type"] == "DirectiveEdit":
            return self._run_edit(
                intent,
                scene_context,
                semantic_intent or {},
                revision=revision,
                feedback=feedback,
            )
        if task_contract["task_type"] not in {"Generate", "Optimize"}:
            payload = {
                "agent_name": self.name,
                "design_status": "unsupported_handoff",
                "handoff": None,
                "evidence_package": None,
                "flp_ir": None,
                "rough_layout": None,
                "solver_report": None,
                "solver_used": None,
                "design_basis": {"intent_type": intent["intent_type"], "protected_user_constraints": []},
                "solver_evidence": None,
                "applied_feedback": [],
                "revision": _design_revision(revision, None, None, feedback),
                "trace": _trace(
                    self.name,
                    self.responsibility,
                    "intent_v1",
                    "design_handoff",
                    iteration=revision,
                    input_artifacts=["intent_v1"],
                    output_artifacts=["design_handoff"],
                    decision="handoff",
                ),
            }
            return enforce_agent_output(self.name, payload)

        evidence = build_evidence_package(
            intent,
            self.adapter,
            semantic_frame=semantic_intent,
            context=scene_context,
        )
        flp_ir = normalize_flp_ir(
            compile_flp_ir(
                scene_context.get("current_user_input") or user_message,
                intent,
                evidence,
                context=scene_context,
            ),
            preserve_instance_ids=bool(scene_context.get("instance_directory")),
        )
        flp_ir = _apply_validator_flare_repairs(
            flp_ir,
            semantic_intent or {},
            feedback,
        )
        if task_contract["task_type"] == "Optimize":
            flp_ir = _compile_existing_layout_flp_ir(
                flp_ir,
                scene_context.get("last_flp_ir")
                or scene_context.get("base_flp_ir"),
                reference_layout=scene_context.get("last_layout"),
                user_message=scene_context.get("current_user_input") or user_message,
            )
        elif task_contract["task_type"] == "Generate":
            generated_context = copy.deepcopy(
                scene_context.get("authoritative_context") or {}
            )
            if not generated_context.get("site_snapshot"):
                generated_context["site_snapshot"] = copy.deepcopy(
                    (flp_ir.get("site_ref") or {}).get("site_snapshot") or {}
                )
            generated_context["authoritative_flp_ir"] = flp_ir
            scene_context["task_contract"] = TaskContractCompiler().compile(
                intent,
                semantic_intent or {},
                generated_context,
            ).to_dict()
        return self._solve_compiled_ir(
            flp_ir,
            intent,
            evidence,
            scene_context,
            revision=revision,
            feedback=feedback,
            parent_design=None,
        )

    def revise(self, previous_design, intent, feedback, scene_context=None, *, revision):
        scene_context = AuthoritativeContextResolver().apply(
            (scene_context or {}).get("current_user_input")
            or intent.get("user_message")
            or "",
            scene_context or {},
        )
        if not _feedback_actions(feedback):
            raise RAGBlockedError(
                "empty_structured_feedback",
                "A feedback revision requires at least one structured action.",
                details={"revision": int(revision)},
            )
        flp_ir = copy.deepcopy(previous_design.get("flp_ir"))
        if flp_ir is not None:
            _append_revision_provenance(flp_ir, revision, feedback)
        task_contract = copy.deepcopy(
            previous_design.get("task_contract")
            or scene_context.get("task_contract")
            or {}
        )
        canonical_spec = copy.deepcopy(
            previous_design.get("canonical_constraint_spec") or {}
        )
        if feedback.get("source") == "validator":
            if (
                feedback.get("feedback_type")
                != "validation_driven_recovery"
            ):
                raise RAGBlockedError(
                    "invalid_validator_recovery_feedback",
                    "Designer recovery requires Validator TypedViolation feedback.",
                    details={"feedback_type": feedback.get("feedback_type")},
                )
            if feedback.get("flare_repair_required"):
                semantic_intent = copy.deepcopy(
                    scene_context.get("frozen_semantic_intent") or {}
                )
                if not semantic_intent:
                    raise RAGBlockedError(
                        "missing_frozen_semantic_intent",
                        "Validator-directed FLARE repair requires the frozen Semantic Intent.",
                        details={"revision": int(revision)},
                    )
                return self.run(
                    scene_context.get("current_user_input")
                    or intent.get("user_message")
                    or "",
                    intent,
                    scene_context=scene_context,
                    semantic_intent=semantic_intent,
                    revision=revision,
                    feedback=feedback,
                )
            if not previous_design.get("rough_layout"):
                raise RAGBlockedError(
                    "missing_layout_for_validation_recovery",
                    "Geometric ValidationDrivenRecovery requires a candidate layout.",
                    details={"revision": int(revision)},
                )
            typed = (
                feedback.get("typed_violations")
                or feedback.get("issues")
                or feedback.get("actions")
                or []
            )
            recovery = ValidationDrivenRecovery().recover(
                previous_design["rough_layout"],
                flp_ir,
                canonical_spec,
                typed,
                task_contract,
                reference_layout=task_contract.get("base_layout"),
                solver_options=scene_context.get("solver_options"),
            )
            return self._revised_design_payload(
                previous_design,
                recovery["layout"],
                flp_ir,
                feedback,
                revision,
                recovery_result=recovery,
                decision="validation_driven_recovery",
            )
        if feedback.get("source") == "evaluator" and previous_design.get(
            "rough_layout"
        ):
            if (
                feedback.get("feedback_type")
                == "evaluator_assisted_recovery"
            ):
                requested_layout = previous_design["rough_layout"]
                recovery_movable, recovery_preserve = (
                    _evaluator_assisted_recovery_scope(task_contract, feedback)
                )
                projected = _project_clearance_recovery_seed(
                    requested_layout,
                    feedback.get("typed_violations")
                    or feedback.get("issues")
                    or [],
                    recovery_movable,
                )
                recovery_spec = copy.deepcopy(canonical_spec)
                recovery_flp_ir = copy.deepcopy(flp_ir)
                _preserve_requested_edit_rotations(
                    requested_layout,
                    recovery_flp_ir,
                    recovery_spec,
                    task_contract,
                    recovery_movable,
                )
                projected_validation = CanonicalConstraintValidator().validate(
                    projected,
                    recovery_spec,
                    reference_layout=requested_layout,
                    expected_instance_ids=task_contract.get("expected_instance_ids"),
                    preserve_instance_ids=recovery_preserve,
                ) if projected is not None else None
                if projected_validation and projected_validation.get("status") == "VALID":
                    recovered = projected
                elif recovery_movable:
                    recovered = solve_canonical_feasibility(
                        recovery_flp_ir,
                        recovery_spec,
                        seed_layout=projected or requested_layout,
                        reference_layout=requested_layout,
                        movable_instance_ids=recovery_movable,
                        preserve_instance_ids=recovery_preserve,
                        scope="local_feasible",
                        optimize_objective=False,
                        time_limit_s=scene_context.get("recovery_time_limit_s") or 5.0,
                        solver_options=scene_context.get("solver_options"),
                    )
                else:
                    recovered = None
                recovery_result = _evaluator_assisted_recovery_result(
                    requested_layout,
                    recovered,
                    task_contract,
                    feedback,
                    recovery_spec,
                    recovery_movable_instance_ids=recovery_movable,
                    recovery_preserve_instance_ids=recovery_preserve,
                )
                task_contract.setdefault("metadata", {})[
                    "evaluator_assisted_recovery"
                ] = copy.deepcopy(recovery_result["disclosure"])
                previous_design = copy.deepcopy(previous_design)
                previous_design["task_contract"] = task_contract
                return self._revised_design_payload(
                    previous_design,
                    recovery_result["layout"],
                    flp_ir,
                    feedback,
                    revision,
                    recovery_result=recovery_result,
                    decision="evaluator_assisted_recovery",
                )
            optimized = solve_canonical_feasibility(
                flp_ir,
                canonical_spec,
                seed_layout=previous_design["rough_layout"],
                reference_layout=task_contract.get("base_layout"),
                movable_instance_ids=task_contract.get("movable_instance_ids"),
                preserve_instance_ids=task_contract.get("preserve_instance_ids"),
                scope="quality",
                optimize_objective=True,
                solver_options=scene_context.get("solver_options"),
            )
            if optimized is None:
                optimized = copy.deepcopy(previous_design["rough_layout"])
            return self._revised_design_payload(
                previous_design,
                optimized,
                flp_ir,
                feedback,
                revision,
                recovery_result=None,
                decision="canonical_quality_optimize",
            )
        if _feedback_requires_fresh_start(feedback):
            scene_context["seed_layout"] = None
            scene_context["last_layout"] = None
            scene_context["_discard_seed_layout"] = True
            scene_context["_warm_start_policy"] = "discard_invalid_validator_candidate"
        else:
            scene_context["seed_layout"] = previous_design.get("rough_layout")
            scene_context["_warm_start_policy"] = "evaluator_feedback_warm_start"
        return self._solve_compiled_ir(
            flp_ir,
            intent,
            previous_design.get("evidence_package"),
            scene_context,
            revision=revision,
            feedback=feedback,
            parent_design=previous_design,
        )

    def _revised_design_payload(
        self,
        previous_design,
        rough_layout,
        flp_ir,
        feedback,
        revision,
        *,
        recovery_result,
        decision,
    ):
        result = copy.deepcopy(previous_design)
        parent_layout = previous_design.get("rough_layout") or {}
        rough_layout = copy.deepcopy(rough_layout)
        rough_layout["layout_id"] = f"layout_v{revision}"
        rough_layout["revision_id"] = f"layout_v{revision}"
        rough_layout["parent_layout_id"] = parent_layout.get("layout_id")
        recovery_validation = (recovery_result or {}).get("validation") or {}
        hard = recovery_validation.get("hard_validation") or {}
        task = recovery_validation.get("task_contract_validation") or {}
        recovery_valid = bool(
            hard.get("status") == "VALID" and task.get("status") == "VALID"
        )
        report = copy.deepcopy(result.get("solver_report") or {})
        report.update(
            {
                "solver": "genetic_algorithm",
                "solver_used": "genetic_algorithm",
                "recovery_operator": (
                    "validation_driven_recovery"
                    if decision == "validation_driven_recovery"
                    else (
                        "evaluator_assisted_recovery"
                        if decision == "evaluator_assisted_recovery"
                        else None
                    )
                ),
                "status": (
                    "FEASIBLE_ALTERNATIVE"
                    if (
                        decision == "evaluator_assisted_recovery"
                        and (recovery_result or {}).get("status")
                        == "EVALUATOR_RECOVERY_SUCCESS"
                    )
                    else (
                        "FEASIBLE"
                        if recovery_result is not None and recovery_valid
                        else (
                            "INFEASIBLE"
                            if recovery_result is not None
                            else "CANDIDATE_GENERATED"
                        )
                    )
                ),
                "objective_value": objective_value(rough_layout, flp_ir),
                "violations": copy.deepcopy(hard.get("violations") or []),
                "violation_count": hard.get("violation_count"),
            }
        )
        result.update(
            {
                "agent_name": self.name,
                "design_status": (
                    "layout_recovered"
                    if decision == "validation_driven_recovery"
                    else (
                        "layout_recovered_with_disclosed_deviation"
                        if decision == "evaluator_assisted_recovery"
                        else "quality_candidate_generated"
                    )
                ),
                "flp_ir": flp_ir,
                "rough_layout": rough_layout,
                "solver_report": report,
                "solver_used": report["solver_used"],
                "viewport_precheck": None,
                "applied_feedback": copy.deepcopy(_feedback_actions(feedback)),
                "revision": _design_revision(
                    revision,
                    flp_ir,
                    rough_layout,
                    feedback,
                    parent_layout_id=parent_layout.get("layout_id"),
                ),
                "trace": _trace(
                    self.name,
                    self.responsibility,
                    f"layout_v{revision - 1} + executable_feedback",
                    f"layout_v{revision}",
                    iteration=revision,
                    input_artifacts=[
                        f"layout_v{revision - 1}",
                        f"feedback_v{revision - 1}",
                    ],
                    output_artifacts=[f"layout_v{revision}"],
                    decision=decision,
                ),
            }
        )
        result.setdefault("design_basis", {})["recovery_strategy"] = decision
        result["design_basis"]["mutation_summary"] = {
            "strategy": decision,
            "candidate_changed": rough_layout != parent_layout,
            "changed_instance_ids": (
                (recovery_result or {}).get("changed_instance_ids") or []
            ),
            "recovery_actions": copy.deepcopy(
                (recovery_result or {}).get("actions") or []
            ),
        }
        result["validation_driven_recovery"] = copy.deepcopy(
            recovery_result
            if decision == "validation_driven_recovery"
            else None
        )
        result["evaluator_assisted_recovery"] = copy.deepcopy(
            recovery_result
            if decision == "evaluator_assisted_recovery"
            else None
        )
        if recovery_result:
            result["trace"].update(
                copy.deepcopy(recovery_result.get("trace") or {})
            )
        return enforce_agent_output(self.name, result)

    def retry(self, previous_design, intent, scene_context=None, *, revision):
        """Run one compute-matched retry without fabricating structured feedback."""

        scene_context = dict(scene_context or {})
        scene_context["task_contract"] = copy.deepcopy(
            previous_design.get("task_contract") or {}
        )
        scene_context["seed_layout"] = previous_design.get("rough_layout")
        scene_context["_warm_start_policy"] = "bounded_retry_previous_layout"
        return self._solve_compiled_ir(
            copy.deepcopy(previous_design.get("flp_ir")),
            intent,
            previous_design.get("evidence_package"),
            scene_context,
            revision=revision,
            feedback=None,
            parent_design=previous_design,
        )

    def _solve_compiled_ir(
        self,
        flp_ir,
        intent,
        evidence,
        scene_context,
        *,
        revision,
        feedback,
        parent_design,
    ):
        solver_report = None
        rough_layout = None
        solver_used = None
        task_contract = copy.deepcopy(scene_context.get("task_contract") or {})
        if (
            task_contract.get("task_type") == "Optimize"
            and flp_ir
            and task_contract.get("base_layout")
        ):
            task_contract["baseline_objective_value"] = objective_value(
                task_contract["base_layout"],
                flp_ir,
            )
            task_contract["objective_flow_relations"] = copy.deepcopy(
                flp_ir.get("flow_relations") or []
            )
            task_contract.setdefault("metadata", {})["objective_problem"] = copy.deepcopy({
                "objectives": flp_ir.get("objectives") or [], "flow_relations": flp_ir.get("flow_relations") or [],
                "site_ref": flp_ir.get("site_ref") or {}})
            scene_context["task_contract"] = copy.deepcopy(task_contract)
        canonical_spec = (
            CanonicalConstraintCompiler().compile(
                _compiled_site_snapshot(flp_ir, scene_context),
                flp_ir=flp_ir,
                task_contract=task_contract,
                reference_layout=task_contract.get("base_layout")
                or scene_context.get("last_layout"),
            )
            if flp_ir
            else {}
        )
        feasibility_preflight = _feasibility_preflight(
            flp_ir,
            canonical_spec,
            task_contract,
        )
        validation_driven_recovery = None
        viewport_precheck = None
        baseline_solver_options = {}
        solver_seed_layout = None
        requested_solver = _requested_solver_from_context(scene_context)
        selection_source = scene_context.get("solver_selection_source") or (
            "user" if requested_solver else "default"
        )
        if flp_ir and flp_ir.get("equipment_instances") and not flp_ir.get("unresolved_items"):
            adaptive_budget = scene_context.get("adaptive_solver_budget", True) is not False
            baseline_solver_options = _solver_options_for_revision(
                requested_solver,
                revision,
                None,
                base_options=scene_context.get("solver_options"),
                adaptive_budget=adaptive_budget,
            )
            solver_options = _solver_options_for_revision(
                requested_solver,
                revision,
                feedback,
                base_options=scene_context.get("solver_options"),
                adaptive_budget=adaptive_budget,
            )
            solver = select_solver(
                flp_ir,
                requested_solver=requested_solver,
                solver_options=solver_options,
            )
            try:
                solver_seed_layout = (
                    None
                    if scene_context.get("_discard_seed_layout")
                    else scene_context.get("seed_layout") or scene_context.get("last_layout")
                )
                solver_result = solver.solve(
                    flp_ir,
                    seed_layout=solver_seed_layout,
                    time_limit_s=_revision_time_limit(
                        scene_context.get("time_limit_s"),
                        revision,
                        adaptive_budget=scene_context.get("adaptive_solver_budget", True) is not False,
                    ),
                )
                rough_layout = copy.deepcopy(solver_result["layout"])
                solver_report = annotate_solver_selection(
                    solver_result["solver_report"],
                    requested_solver=requested_solver,
                    selection_source=selection_source,
                )
            except Exception as exc:  # The Validator converts this evidence into a fatal no-layout result.
                selected_name = normalize_solver_name(requested_solver) or DEFAULT_SOLVER_NAME
                solver_report = annotate_solver_selection(
                    {
                        "solver": selected_name,
                        "solver_used": selected_name,
                        "status": "ERROR",
                        "objective_value": None,
                        "bound": None,
                        "gap": None,
                        "runtime_s": 0.0,
                        "constraint_penalty": None,
                        "violation_count": 1,
                        "violations": [
                            {
                                "violation_type": "solver_failure",
                                "message": str(exc),
                            }
                        ],
                        "failure_reason": str(exc),
                    },
                    requested_solver=requested_solver,
                    selection_source=selection_source,
                )
            solver_report["solver_options"] = solver_options
            solver_report["design_revision"] = revision
            solver_used = solver_report["solver_used"]
            _log_solver_execution(solver_report)

        parent_layout_id = (
            (parent_design or {}).get("rough_layout", {}).get("layout_id")
            if isinstance((parent_design or {}).get("rough_layout"), dict)
            else None
        )
        if rough_layout is not None:
            rough_layout["layout_id"] = f"layout_v{revision}"
            rough_layout["revision_id"] = f"layout_v{revision}"
            rough_layout["parent_layout_id"] = parent_layout_id
        revision_record = _design_revision(revision, flp_ir, rough_layout, feedback, parent_layout_id=parent_layout_id)
        applied_feedback = _feedback_actions(feedback)
        mutation_summary = _design_mutation_summary(
            revision=revision,
            parent_design=parent_design,
            flp_ir=flp_ir,
            solver_options=(solver_report or {}).get("solver_options") or {},
            baseline_solver_options=baseline_solver_options,
            seed_layout=solver_seed_layout,
            warm_start_policy=scene_context.get("_warm_start_policy") or "initial_context",
            feedback=feedback,
        )
        revision_record["mutation_summary"] = copy.deepcopy(mutation_summary)
        design_basis = {
            "intent_id": intent.get("intent_id"),
            "intent_type": intent.get("intent_type"),
            "evidence_fact_count": len((evidence or {}).get("facts") or []),
            "protected_user_constraints": [
                item.get("constraint_type")
                for item in (flp_ir or {}).get("constraints", {}).get("hard", [])
                if item.get("source") == "user_input"
            ],
            "feedback_source": (feedback or {}).get("source"),
            "feedback_strategy": _feedback_strategy(feedback),
            "user_constraints_preserved": _user_constraints_preserved(flp_ir, parent_design),
            "mutation_summary": mutation_summary,
        }
        solver_evidence = None
        if solver_report:
            solver_evidence = {
                "solver_used": solver_report.get("solver_used"),
                "status": solver_report.get("status"),
                "objective_value": solver_report.get("objective_value"),
                "constraint_penalty": solver_report.get("constraint_penalty"),
                "violation_count": solver_report.get("violation_count"),
                "runtime_s": solver_report.get("runtime_s"),
                "solver_options": solver_report.get("solver_options"),
            }
        payload = {
            "agent_name": self.name,
            "design_status": "rough_layout_generated" if rough_layout else "no_layout_generated",
            "handoff": None,
            "evidence_package": evidence,
            "flp_ir": flp_ir,
            "rough_layout": rough_layout,
            "task_contract": task_contract,
            "canonical_constraint_spec": canonical_spec,
            "feasibility_preflight": feasibility_preflight,
            "validation_driven_recovery": validation_driven_recovery,
            "viewport_precheck": viewport_precheck,
            "solver_report": solver_report,
            "solver_used": solver_used,
            "design_basis": design_basis,
            "solver_evidence": solver_evidence,
            "applied_feedback": applied_feedback,
            "revision": revision_record,
            "trace": _trace(
                self.name,
                self.responsibility,
                "intent + evidence + feedback",
                f"flp_ir_v{revision} + layout_v{revision}",
                iteration=revision,
                input_artifacts=["intent_v1"] + ([f"feedback_v{revision - 1}"] if feedback else []),
                output_artifacts=[f"flp_ir_v{revision}"] + ([f"layout_v{revision}"] if rough_layout else []),
                decision="revise_and_resolve" if feedback else "initial_design",
                evidence=[item.get("evidence_id") for item in applied_feedback if item.get("evidence_id")],
            ),
        }
        return enforce_agent_output(self.name, payload)

    def _run_edit(self, intent, scene_context, semantic_intent, *, revision, feedback):
        task_contract = copy.deepcopy(scene_context.get("task_contract") or {})
        edit_ir = compile_layout_edit_ir(intent, semantic_intent, scene_context, self.adapter)
        base_layout = scene_context.get("last_layout") or scene_context.get("seed_layout")
        base_flp_ir = copy.deepcopy(scene_context.get("last_flp_ir") or scene_context.get("base_flp_ir"))
        from tool.context.conversation_deltas import service_duty_transfers
        try:
            service_transfers = service_duty_transfers(
                str(scene_context.get("current_user_input") or intent.get("user_message") or "").rsplit("Current user request:\n",1)[-1],
                base_flp_ir, edit_ir)
        except ValueError as error:
            service_transfers = []
            edit_ir["unresolved_items"].append(dict(item_id="service_duty_transfer",item_type="missing_field",field_path="operations",
                description=str(error),severity="blocking",candidates=[],resolution_policy="ask_user"))
        hard=base_flp_ir.setdefault("constraints",{}).setdefault("hard",[])
        declared_types={_canonical_flare_constraint_type(c.get("constraint_type")) for c in semantic_intent.get("constraints") or [] if str(c.get("severity") or "hard").lower()=="hard"}
        present={c.get("constraint_type") for c in hard}
        if "fixed_pose" in declared_types-present:
            hard.append(dict(constraint_id=f"constraint_{len(hard)+1:03d}",constraint_type="fixed_pose",
                target_refs=[dict(target_type="equipment_instance",target_id=i) for i in task_contract.get("preserve_instance_ids") or []],
                parameters=dict(scope="current_edit_only",implementation="Task-contract preservation against the accepted parent layout"),source="user_input",evidence_refs=["source_request.raw_user_message"]))
        if "near" in declared_types-present:
            for op in edit_ir.get("operations") or []:
                ref=(op.get("parameters") or {}).get("reference_instance_id")
                if op.get("op_type")=="add" and ref:
                    hard.append(dict(constraint_id=f"constraint_{len(hard)+1:03d}",constraint_type="near",
                        target_refs=[dict(target_type="equipment_instance",target_id=i) for i in (op["instance_id"],ref)],
                        parameters=dict(placement_policy="nearest_feasible_candidate",basis="Qualitative near request; no manufacturer distance limit asserted"),source="user_input",evidence_refs=["source_request.raw_user_message"]))
        rotated_ids = {op.get("instance_id") for op in edit_ir.get("operations") or [] if op.get("op_type") == "rotate"}
        for item in (base_flp_ir or {}).get("equipment_instances") or []:
            if item["instance_id"] in rotated_ids:
                asset = self.adapter.get_asset(item["model_asset_id"])
                item["allowed_orientations_deg"] = copy.deepcopy((asset.get("planning_fields") or {}).get("allowed_orientations_deg") or [])
        canonical_spec = CanonicalConstraintCompiler().compile(
            _compiled_site_snapshot(base_flp_ir, scene_context),
            flp_ir=base_flp_ir,
            task_contract=task_contract,
            reference_layout=base_layout,
        )
        execution = ExactEditExecutor(
            self.adapter,
            self.dataset_dir / "schemas",
        ).execute(
            edit_ir,
            base_layout,
            base_flp_ir,
            canonical_spec,
            task_contract,
            process_relations=(semantic_intent.get("process_relations") or [])
                if re.search(r"(?:->|→)",str(scene_context.get("current_user_input") or "").rsplit("Current user request:\n",1)[-1]) else None,
            service_transfers=service_transfers,
        )
        rough_layout = execution.get("layout")
        flp_ir = execution.get("flp_ir")
        canonical_spec = execution.get("canonical_constraint_spec") or canonical_spec
        edit_report = execution["edit_ir_report"]
        payload = {
            "agent_name": self.name,
            "design_status": execution["status"],
            "handoff": None,
            "evidence_package": None,
            "layout_edit_ir": edit_ir,
            "layout_diff": execution.get("layout_diff"),
            "edit_ir_report": edit_report,
            "flp_ir": flp_ir,
            "rough_layout": rough_layout,
            "task_contract": task_contract,
            "canonical_constraint_spec": canonical_spec,
            "feasibility_preflight": _feasibility_preflight(
                flp_ir or base_flp_ir,
                canonical_spec,
                task_contract,
            ),
            "exact_edit_execution": execution,
            "validation_driven_recovery": None,
            "viewport_precheck": None,
            "solver_report": None,
            "solver_used": None,
            "design_basis": {
                "intent_type": "edit_layout",
                "base_layout_id": edit_ir.get("base_layout_id"),
                "operation_count": len(edit_ir.get("operations") or []),
                "protected_user_constraints": [
                    item.get("constraint_id")
                    for item in (flp_ir or base_flp_ir or {}).get("constraints", {}).get("hard", [])
                ],
            },
            "solver_evidence": None,
            "applied_feedback": copy.deepcopy(_feedback_actions(feedback)),
            "revision": _design_revision(
                revision,
                flp_ir,
                rough_layout,
                feedback,
                parent_layout_id=edit_ir.get("base_layout_id"),
            ),
            "trace": _trace(
                self.name,
                self.responsibility,
                "intent_v1 + active_layout",
                f"layout_edit_ir_v{revision} + layout_v{revision}",
                iteration=revision,
                input_artifacts=["intent_v1", edit_ir.get("base_layout_id")],
                output_artifacts=[f"layout_edit_ir_v{revision}"] + ([f"layout_v{revision}"] if rough_layout else []),
                decision="exact_edit_execution",
                trigger_source="user_directive",
                task_type="DirectiveEdit",
                movable_instance_ids=task_contract.get(
                    "movable_instance_ids"
                ),
                preserve_instance_ids=task_contract.get(
                    "preserve_instance_ids"
                ),
                changed_instance_ids=execution.get(
                    "changed_instance_ids"
                ),
                deviated_from_user_directive=execution.get(
                    "deviated_from_user_directive"
                ),
                final_status=execution["status"],
            ),
        }
        return enforce_agent_output(self.name, payload)

    def _run_constraint_repair(
        self,
        intent,
        scene_context,
        semantic_intent,
        *,
        revision,
        feedback,
    ):
        task_contract = copy.deepcopy(scene_context.get("task_contract") or {})
        base_layout = copy.deepcopy(
            task_contract.get("base_layout")
            or scene_context.get("last_layout")
            or scene_context.get("seed_layout")
            or {}
        )
        flp_ir = copy.deepcopy(
            scene_context.get("last_flp_ir")
            or scene_context.get("base_flp_ir")
            or {}
        )
        flp_ir = _apply_validator_flare_repairs(flp_ir,semantic_intent or {},feedback)
        current_user_input = scene_context.get("current_user_input") or ""
        inherited_objectives = copy.deepcopy(flp_ir.get("objectives") or [])
        if any(o.get("objective_type")=="minimize_agv_transport_cost" for o in inherited_objectives):
            hard=flp_ir.setdefault("constraints",{}).setdefault("hard",[])
            if not any((c.get("parameters") or {}).get("planning_kind")=="agv_external_access" for c in hard):
                hard.append(dict(constraint_id=f"constraint_{len(hard)+1:03d}",constraint_type="corridor_preservation",
                    target_refs=[dict(target_type="all_equipment",target_id="all")],
                    parameters=dict(planning_kind="agv_external_access",basis="Feasibility of the inherited AGV transport tasks must survive local Repair"),
                    source="user_input",evidence_refs=["source_request.raw_user_message"]))
        active_repair_objective = _materialize_constraint_repair_objective(
            flp_ir,
            semantic_intent,
            current_user_input,
        )
        inherited_site_snapshot = copy.deepcopy(
            (flp_ir.get("site_ref") or {}).get("site_snapshot")
            or scene_context.get("site_snapshot")
            or {}
        )
        site_update = apply_central_aisle_width_update(
            current_user_input,
            inherited_site_snapshot,
            organization_text=scene_context.get("engineering_context"),
            semantic_intent=(
                scene_context.get("current_semantic_intent")
                or scene_context.get("frozen_semantic_intent")
                or semantic_intent
            ),
            equipment_instances=flp_ir.get("equipment_instances") or [],
            intent_id=intent.get("intent_id") or "INTENT_001",
        )
        if site_update.get("site_snapshot"):
            flp_ir.setdefault("site_ref", {})["site_snapshot"] = copy.deepcopy(
                site_update["site_snapshot"]
            )
            scene_context["site_snapshot"] = copy.deepcopy(
                site_update["site_snapshot"]
            )
        flp_ir.setdefault("unresolved_items", []).extend(
            copy.deepcopy(site_update.get("unresolved_items") or [])
        )
        flp_ir.setdefault("provenance", []).extend(
            copy.deepcopy(site_update.get("provenance") or [])
        )
        canonical_spec = CanonicalConstraintCompiler().compile(
            scene_context.get("site_snapshot")
            or ((flp_ir.get("site_ref") or {}).get("site_snapshot"))
            or {},
            flp_ir=flp_ir,
            task_contract=task_contract,
            reference_layout=base_layout,
        )
        execution = ViolationResolutionExecutor().execute(
            base_layout,
            flp_ir,
            canonical_spec,
            task_contract,
            solver_options=scene_context.get("solver_options"),
            time_limit_s=scene_context.get("time_limit_s", 5.0),
        )
        rough_layout = copy.deepcopy(execution["layout"])
        rough_layout["layout_id"] = f"layout_v{revision}"
        rough_layout["revision_id"] = f"layout_v{revision}"
        rough_layout["parent_layout_id"] = base_layout.get("layout_id")
        changed_instance_ids = execution["changed_instance_ids"]
        candidate_changed = bool(changed_instance_ids)
        movement_score = _repair_movement_score(
            base_layout,
            rough_layout,
            task_contract.get("movable_instance_ids") or [],
        )
        inherited_flow_metrics = _historical_repair_flow_metrics(
            base_layout,
            rough_layout,
            flp_ir,
            inherited_objectives,
        )
        solver_report = {
            "solver": "genetic_algorithm",
            "solver_used": "genetic_algorithm",
            "method": "user_constraint_repair",
            "status": execution["status"],
            "objective_value": movement_score,
            "objective_values": (
                {"minimize_layout_change": movement_score}
                if active_repair_objective
                else {}
            ),
            "active_objective": copy.deepcopy(active_repair_objective),
            "historical_flow_metrics": inherited_flow_metrics,
            "constraint_penalty": None,
            "violation_count": None,
            "violations": [],
            "runtime_s": round(execution["runtime_s"], 6),
            "canonical_constraint_spec_sha256": canonical_spec.get("spec_sha256"),
            "task_contract_status": None,
            "solver_options": copy.deepcopy(
                scene_context.get("solver_options") or {}
            ),
        }
        payload = {
            "agent_name": self.name,
            "design_status": execution["status"],
            "handoff": None,
            "evidence_package": None,
            "layout_edit_ir": None,
            "layout_diff": None,
            "edit_ir_report": None,
            "flp_ir": flp_ir,
            "rough_layout": rough_layout,
            "task_contract": task_contract,
            "canonical_constraint_spec": canonical_spec,
            "feasibility_preflight": _feasibility_preflight(
                flp_ir,
                canonical_spec,
                task_contract,
            ),
            "user_constraint_repair": execution,
            "validation_driven_recovery": None,
            "viewport_precheck": None,
            "solver_report": solver_report,
            "solver_used": solver_report["solver_used"],
            "design_basis": {
                "intent_type": intent.get("intent_type"),
                "task_contract_type": "ConstraintRepair",
                "controlled_violation": copy.deepcopy(
                    task_contract.get("controlled_violation")
                ),
                "movable_instance_ids": copy.deepcopy(
                    task_contract.get("movable_instance_ids") or []
                ),
                "preserve_instance_ids": copy.deepcopy(
                    task_contract.get("preserve_instance_ids") or []
                ),
                "protected_user_constraints": [],
                "execution_strategy": "user_constraint_repair",
                "mutation_summary": {
                    "candidate_changed": candidate_changed,
                    "changed_instance_ids": changed_instance_ids,
                    "allowed_repair_operations": copy.deepcopy(
                        task_contract.get("allowed_repair_operations") or []
                    ),
                },
            },
            "solver_evidence": {
                "solver_used": solver_report["solver_used"],
                "status": solver_report["status"],
                "objective_value": solver_report["objective_value"],
                "violation_count": solver_report["violation_count"],
                "runtime_s": solver_report["runtime_s"],
            },
            "applied_feedback": copy.deepcopy(_feedback_actions(feedback)),
            "revision": _design_revision(
                revision,
                flp_ir,
                rough_layout,
                feedback,
                parent_layout_id=base_layout.get("layout_id"),
            ),
            "trace": _trace(
                self.name,
                self.responsibility,
                "controlled_invalid_layout + constraint_repair_contract",
                f"layout_v{revision}",
                iteration=revision,
                input_artifacts=["active_layout", "task_contract"],
                output_artifacts=[f"layout_v{revision}"],
                decision="user_constraint_repair",
                trigger_source="user_constraint_repair_request",
                task_type="ConstraintRepair",
                movable_instance_ids=task_contract.get(
                    "movable_instance_ids"
                ),
                preserve_instance_ids=task_contract.get(
                    "preserve_instance_ids"
                ),
                changed_instance_ids=changed_instance_ids,
                deviated_from_user_directive=False,
                final_status=execution["status"],
            ),
        }
        return enforce_agent_output(self.name, payload)


def _materialize_constraint_repair_objective(
    flp_ir,
    semantic_intent,
    user_message,
):
    """Make a literal Repair movement goal the active FLP objective.

    Existing flow objectives remain in the solver receipt as historical metrics.
    They are not silently treated as the current Repair objective when the user
    explicitly asks to minimize the required layout movement.
    """

    literal_sources = _linguistic_literal_sources(user_message)
    for item in (semantic_intent or {}).get("objectives") or []:
        if not isinstance(item, dict):
            continue
        objective_type = OBJECTIVE_SYNONYMS.get(
            str(item.get("objective_type") or "").strip(),
            str(item.get("objective_type") or "").strip(),
        )
        source_text = str(item.get("source_text") or "").strip()
        if (
            objective_type != "minimize_layout_change"
            or not _literal_span_in_sources(source_text, literal_sources)
        ):
            continue
        active_objective = {
            "objective_id": "objective_001",
            "objective_type": "minimize_layout_change",
            "sense": "minimize",
            "weight": 1.0,
            "parameters": {},
            "source": "user_input",
            "evidence_refs": ["source_request.raw_user_message"],
        }
        flp_ir["objectives"] = [copy.deepcopy(active_objective)]
        flp_ir.setdefault("provenance", []).append(
            {
                "field_path": "objectives[0]",
                "source_type": "user_input",
                "source_id": "INTENT_001",
                "source_field": source_text,
                "confidence": 1.0,
                "note": "The active constraint-repair objective is the user's literal minimum-movement request.",
            }
        )
        return {
            "objective_type": active_objective["objective_type"],
            "source_text": source_text,
            "source": "user_input",
        }
    return None


def _repair_movement_score(reference_layout, candidate_layout, movable_ids):
    reference = {
        str(item.get("instance_id")): item
        for item in (reference_layout or {}).get("placements") or []
        if isinstance(item, dict) and item.get("instance_id")
    }
    candidate = {
        str(item.get("instance_id")): item
        for item in (candidate_layout or {}).get("placements") or []
        if isinstance(item, dict) and item.get("instance_id")
    }
    total = 0.0
    for instance_id in movable_ids or []:
        before = reference.get(str(instance_id))
        after = candidate.get(str(instance_id))
        if before is None or after is None:
            continue
        before_position = before.get("position") or before.get("pose") or {}
        after_position = after.get("position") or after.get("pose") or {}
        try:
            total += abs(float(after_position["x_m"]) - float(before_position["x_m"]))
            total += abs(float(after_position["z_m"]) - float(before_position["z_m"]))
        except (KeyError, TypeError, ValueError):
            continue
    return total


def _historical_repair_flow_metrics(
    reference_layout,
    candidate_layout,
    flp_ir,
    inherited_objectives,
):
    """Retain prior flow results as diagnostics after a Repair changes objective."""

    objectives = [
        item for item in inherited_objectives or [] if isinstance(item, dict)
    ]
    types = {
        OBJECTIVE_SYNONYMS.get(
            str(item.get("objective_type") or "").strip(),
            str(item.get("objective_type") or "").strip(),
        )
        for item in objectives
    }
    metrics = {}
    if "minimize_agv_transport_cost" in types:
        from tool.logistics.agv_transport_cost import agv_transport_cost
        transport_problem=dict(flp_ir,objectives=objectives)
        metrics["minimize_agv_transport_cost"]={"parent":agv_transport_cost(reference_layout,transport_problem),"candidate":agv_transport_cost(candidate_layout,transport_problem)}
    if "minimize_weighted_flow_distance" in types:
        metrics["minimize_weighted_flow_distance"] = {
            "parent": objective_value(reference_layout, flp_ir),
            "candidate": objective_value(candidate_layout, flp_ir),
        }
    relocation = next((item for item in objectives if item.get("objective_type") == RELOCATION_EFFORT_OBJECTIVE), None)
    if relocation:
        metrics[RELOCATION_EFFORT_OBJECTIVE] = {
            "parent": relocation_effort(reference_layout, relocation["parameters"]),
            "candidate": relocation_effort(candidate_layout, relocation["parameters"]),
            "reference_layout_id": relocation["parameters"].get("reference_layout_id"),
        }
    high_flow = next(
        (
            item
            for item in objectives
            if OBJECTIVE_SYNONYMS.get(
                str(item.get("objective_type") or "").strip(),
                str(item.get("objective_type") or "").strip(),
            )
            == "minimize_high_flow_distance_excess"
        ),
        None,
    )
    parameters = (high_flow or {}).get("parameters") or {}
    try:
        flow_threshold = float(parameters["flow_threshold_batches_per_day"])
        distance_threshold = float(parameters["distance_threshold_m"])
    except (KeyError, TypeError, ValueError):
        return metrics
    metrics["minimize_high_flow_distance_excess"] = {
        "parent": high_flow_distance_excess(
            reference_layout,
            flp_ir,
            flow_threshold_batches_per_day=flow_threshold,
            distance_threshold_m=distance_threshold,
        ),
        "candidate": high_flow_distance_excess(
            candidate_layout,
            flp_ir,
            flow_threshold_batches_per_day=flow_threshold,
            distance_threshold_m=distance_threshold,
        ),
        "parameters": {
            "flow_threshold_batches_per_day": flow_threshold,
            "distance_threshold_m": distance_threshold,
        },
    }
    return metrics


class MonolithicPlanningAgent(DesignerAgent):
    """Single-call RAG-grounded planner whose geometry is produced only by the solver."""

    name = "MonolithicPlanningAgent"
    responsibility = (
        "Use one LLM call with Engineering RAG evidence to create a grounded design brief, "
        "then compile FLP-IR and invoke the fixed layout solver without generating coordinates."
    )

    def __init__(self, dataset_dir, rag_service=None, llm_client=None):
        super().__init__(dataset_dir, rag_service=rag_service)
        self.llm_client = llm_client

    def plan(self, user_message, *, active_layout_id=None, scene_context=None):
        scene_context = dict(scene_context or {})
        if self.llm_client is None:
            raise RAGBlockedError(
                "monolithic_llm_unavailable",
                "MonolithicPlanningAgent requires one live LLM planning call.",
                details={"fallback_enabled": False},
            )
        if not hasattr(self.adapter.rag_service, "query"):
            raise RAGBlockedError(
                "monolithic_rag_unavailable",
                "MonolithicPlanningAgent requires an injected Engineering RAG query service.",
            )
        rag_result = self.adapter.rag_service.query(user_message, top_k=8)
        rag_evidence = _monolithic_rag_snapshot(rag_result, self.adapter.rag_service)
        instance_directory = _current_instance_directory(scene_context)
        try:
            plan_kwargs = {
                "active_layout_id": active_layout_id,
                "initial_layout_context": _monolithic_initial_layout_context(scene_context),
            }
            if instance_directory:
                plan_kwargs["instance_directory"] = instance_directory
            llm_frame = self.llm_client.plan_layout_design(
                user_message,
                rag_evidence,
                **plan_kwargs,
            )
        except RAGBlockedError:
            raise
        except Exception as exc:
            raise RAGBlockedError(
                "monolithic_llm_api_error",
                "Monolithic planning failed; no fallback or second LLM call is allowed.",
                details={"exception_type": type(exc).__name__},
            ) from exc
        _validate_monolithic_brief(
            llm_frame,
            user_message,
            rag_evidence,
            self.adapter,
            instance_directory=instance_directory,
        )
        intent_step = _semantic_intent_step(
            llm_frame,
            user_message,
            active_layout_id=active_layout_id,
            scene_context=scene_context,
            agent_name=self.name,
            responsibility=self.responsibility,
            output_artifact="monolithic_design_brief_v1",
        )
        evidence_package = None
        flp_ir = None
        if intent_step["can_enter_designer"]:
            evidence_package = build_evidence_package(
                intent_step["intent"],
                self.adapter,
                semantic_frame=intent_step["semantic_intent"],
                context=scene_context,
            )
            flp_ir = normalize_flp_ir(
                compile_flp_ir(
                    intent_step["compiler_message"],
                    intent_step["intent"],
                    evidence_package,
                    context=scene_context,
                ),
                preserve_instance_ids=bool(
                    scene_context.get("instance_directory")
                ),
            )
        return {
            "plan_id": "monolithic_plan_v1",
            "intent_step": intent_step,
            "planning_brief": copy.deepcopy(intent_step["semantic_intent"]),
            "rag_evidence": rag_evidence,
            "rag_evidence_ids": _monolithic_evidence_ids(rag_evidence),
            "evidence_package": evidence_package,
            "flp_ir": flp_ir,
            "llm_call_count": 1,
        }

    def run(
        self,
        user_message,
        *,
        scene_context=None,
        active_layout_id=None,
        revision=1,
        planning_step=None,
        planning_cache_reused=False,
        parent_design=None,
    ):
        scene_context = dict(scene_context or {})
        if parent_design is not None:
            scene_context["seed_layout"] = parent_design.get("rough_layout")
            scene_context["_warm_start_policy"] = "bounded_retry_previous_layout"
        plan = copy.deepcopy(planning_step) if planning_step is not None else self.plan(
            user_message,
            active_layout_id=active_layout_id,
            scene_context=scene_context,
        )
        intent_step = plan["intent_step"]
        if not intent_step["can_enter_designer"]:
            return self._blocked_design(plan, revision, planning_cache_reused)
        design = self._solve_compiled_ir(
            copy.deepcopy(plan["flp_ir"]),
            intent_step["intent"],
            copy.deepcopy(plan["evidence_package"]),
            scene_context,
            revision=revision,
            feedback=None,
            parent_design=parent_design,
        )
        design["planning_brief"] = copy.deepcopy(plan["planning_brief"])
        design["rag_evidence"] = copy.deepcopy(plan["rag_evidence"])
        design["design_basis"].update(
            {
                "planning_mode": "single_llm_rag_grounded_brief",
                "llm_call_count": int(plan["llm_call_count"]),
                "planning_cache_reused": bool(planning_cache_reused),
                "coordinate_generator": "genetic_algorithm",
                "rag_backend": copy.deepcopy(plan["rag_evidence"]["backend"]),
            }
        )
        design["trace"] = _trace(
            self.name,
            self.responsibility,
            "user_message + engineering_rag_evidence",
            f"flp_ir_v{revision} + layout_v{revision}",
            iteration=revision,
            input_artifacts=["user_message", "engineering_rag_evidence", "monolithic_design_brief_v1"],
            output_artifacts=[f"flp_ir_v{revision}"]
            + ([f"layout_v{revision}"] if design.get("rough_layout") else []),
            decision="compile_brief_and_solve_once",
            evidence=plan["rag_evidence_ids"],
        )
        design["trace"]["rag_backend"] = copy.deepcopy(plan["rag_evidence"]["backend"])
        design["trace"]["llm_call_count"] = int(plan["llm_call_count"])
        design["trace"]["planning_cache_reused"] = bool(planning_cache_reused)
        return design

    def _blocked_design(self, plan, revision, planning_cache_reused):
        intent_step = plan["intent_step"]
        payload = {
            "agent_name": self.name,
            "design_status": "skipped_by_monolithic_plan_contract",
            "handoff": None,
            "evidence_package": copy.deepcopy(plan.get("evidence_package")),
            "flp_ir": None,
            "rough_layout": None,
            "solver_report": None,
            "solver_used": None,
            "design_basis": {
                "intent_type": intent_step["intent"].get("intent_type"),
                "protected_user_constraints": [],
                "skip_reason": "The grounded design brief has blocking requirements.",
                "planning_mode": "single_llm_rag_grounded_brief",
                "llm_call_count": int(plan["llm_call_count"]),
                "planning_cache_reused": bool(planning_cache_reused),
                "coordinate_generator": "genetic_algorithm",
                "rag_backend": copy.deepcopy(plan["rag_evidence"]["backend"]),
            },
            "solver_evidence": None,
            "applied_feedback": [],
            "revision": _design_revision(revision, None, None, None),
            "trace": _trace(
                self.name,
                self.responsibility,
                "user_message + engineering_rag_evidence",
                "monolithic_design_blocked",
                iteration=revision,
                input_artifacts=["user_message", "engineering_rag_evidence"],
                output_artifacts=["monolithic_design_blocked"],
                decision="request_clarification",
                evidence=plan["rag_evidence_ids"],
            ),
            "planning_brief": copy.deepcopy(plan["planning_brief"]),
            "rag_evidence": copy.deepcopy(plan["rag_evidence"]),
        }
        payload["trace"]["rag_backend"] = copy.deepcopy(plan["rag_evidence"]["backend"])
        payload["trace"]["llm_call_count"] = int(plan["llm_call_count"])
        payload["trace"]["planning_cache_reused"] = bool(planning_cache_reused)
        return enforce_agent_output(self.name, payload)


class ValidatorAgent:
    name = "ValidatorAgent"
    responsibility = (
        "Validate semantic intent separately from FLP-IR/layout state, then emit "
        "executable feedback for the responsible upstream agent."
    )

    def __init__(self, dataset_dir, rag_service=None):
        self.dataset_dir = Path(dataset_dir)
        self.adapter = AssetReferenceAdapter(self.dataset_dir, rag_service=rag_service)

    def validate_intent(
        self,
        intent_step,
        user_message,
        *,
        scene_context=None,
        revision=1,
    ):
        scene_context = scene_context or {}
        contract = copy.deepcopy(
            intent_step.get("intent_contract") or _valid_intent_contract()
        )
        issues = []
        for item in contract.get("issues") or []:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "issue_id": item.get("issue_id")
                    or f"intent_issue_{len(issues) + 1:03d}",
                    "issue_type": item.get("issue_type")
                    or "intent_contract_issue",
                    "field_path": item.get("field_path"),
                    "message": item.get("message"),
                    "severity": item.get("severity") or "blocking",
                    "resolution_policy": item.get("resolution_policy")
                    or "revise_intent",
                    "details": copy.deepcopy(item.get("details") or {}),
                }
            )
        for item in intent_step.get("unresolved_requirements") or []:
            if not isinstance(item, dict) or item.get("severity") != "blocking":
                continue
            requirement_id = item.get("requirement_id")
            if any(
                issue.get("issue_id") == requirement_id
                or (
                    issue.get("field_path") == item.get("field_path")
                    and issue.get("message") == item.get("description")
                )
                for issue in issues
            ):
                continue
            issues.append(
                {
                    "issue_id": requirement_id
                    or f"unresolved_intent_{len(issues) + 1:03d}",
                    "issue_type": requirement_id or "missing_required_field",
                    "field_path": item.get("field_path"),
                    "message": item.get("description"),
                    "severity": "blocking",
                    "resolution_policy": item.get("resolution_policy")
                    or "ask_user",
                    "details": {},
                }
            )

        revise_issues = [
            item
            for item in issues
            if item.get("resolution_policy") == "revise_intent"
        ]
        clarification_issues = [
            item
            for item in issues
            if item.get("resolution_policy") != "revise_intent"
        ]
        if not issues:
            status = "VALID"
        elif clarification_issues:
            status = "NEED_CLARIFICATION"
        else:
            status = "REPAIRABLE"
        actions = [
            {
                "action_id": f"intent_repair_{index:03d}",
                "action_type": "revise_intent_field",
                "target_agent": "IntentUnderstandingAgent",
                "field_path": item.get("field_path"),
                "issue_type": item.get("issue_type"),
                "instruction": item.get("message"),
                "allowed_sources": [
                    "user_message",
                    "conversation_history",
                    "linguistic_frame",
                    "engineering_rag_grounding",
                    "active_layout",
                    "instance_directory",
                ],
            }
            for index, item in enumerate(revise_issues, start=1)
        ]
        clarification_option_questions = _public_clarification_asset_questions(
            (scene_context or {}).get("clarification_asset_options") or []
        )
        clarification_requirements = [
            {
                "requirement_id": item.get("issue_type"),
                "description": item.get("message"),
                "severity": "blocking",
                "resolution_policy": "ask_user",
            }
            for item in clarification_issues
            if (
                not clarification_option_questions
                or item.get("issue_type")
                not in {
                    "intent_asset_not_grounded",
                    "intent_process_relation_not_grounded",
                    "intent_requirement_not_grounded",
                }
            )
        ]
        clarification_questions = clarification_option_questions + _intent_clarification_questions(
            clarification_requirements,
            (
                []
                if clarification_option_questions
                else (intent_step.get("semantic_intent") or {}).get(
                    "clarification_questions"
                )
                or []
            ),
        )
        repair = {
            "source": "validator",
            "target_agent": "IntentUnderstandingAgent",
            "source_revision_id": f"intent_validation_v{revision}",
            "disposition": (
                "none"
                if status == "VALID"
                else (
                    "auto_repair"
                    if status == "REPAIRABLE"
                    else "clarification_required"
                )
            ),
            "should_revise": status == "REPAIRABLE",
            "actions": actions,
            "repair_actions": copy.deepcopy(actions),
            "issues": copy.deepcopy(issues),
            "clarification_questions": clarification_questions,
            "requires_user_clarification": status == "NEED_CLARIFICATION",
            "can_auto_repair": status == "REPAIRABLE",
            "fatal": False,
        }
        report = {
            "schema_version": "layout-agentic-intent-validation-v1",
            "status": status,
            "scope": "intent",
            "errors": copy.deepcopy(issues),
            "warnings": [],
            "contract_status": contract.get("status"),
            "can_enter_designer": status == "VALID"
            and bool(intent_step.get("can_enter_designer")),
        }
        payload = {
            "agent_name": self.name,
            "validation_report": report,
            "intent_validation": copy.deepcopy(report),
            "layout_validation": {
                "status": "NOT_RUN",
                "issues": [],
                "violation_count": 0,
            },
            "validation_scope": "intent",
            "repair": repair,
            "revision": {
                "revision_index": int(revision),
                "validation_revision_id": f"intent_validation_v{revision}",
                "intent_revision_id": intent_step.get("intent_revision_id")
                or f"intent_v{revision}",
                "flp_ir_revision_id": None,
                "layout_revision_id": None,
            },
            "trace": _trace(
                self.name,
                self.responsibility,
                intent_step.get("intent_revision_id") or f"intent_v{revision}",
                f"intent_validation_v{revision}",
                iteration=revision,
                input_artifacts=[
                    intent_step.get("intent_revision_id")
                    or f"intent_v{revision}"
                ],
                output_artifacts=[
                    f"intent_validation_v{revision}",
                    f"intent_feedback_v{revision}",
                ],
                decision=repair["disposition"],
                evidence=[
                    f"{item.get('issue_type')}:{item.get('field_path')}"
                    for item in issues
                ],
            ),
        }
        return enforce_agent_output(self.name, payload)

    def run(
        self,
        flp_ir,
        intent,
        rough_layout=None,
        *,
        revision=1,
        layout_edit_ir=None,
        edit_ir_report=None,
        scene_context=None,
        task_contract=None,
        canonical_constraint_spec=None,
        viewport_precheck=None,
        semantic_intent=None,
    ):
        scene_context = scene_context or {}
        task_contract = copy.deepcopy(
            task_contract
            or scene_context.get("task_contract")
            or {}
        )
        if flp_ir is None:
            unresolved = intent.get("unresolved_requirements") or []
            if intent.get("intent_type") == "edit_layout":
                if edit_ir_report is not None:
                    report = copy.deepcopy(edit_ir_report)
                elif layout_edit_ir is not None:
                    report = validate_layout_edit_ir(
                        layout_edit_ir,
                        None,
                        self.adapter,
                        self.dataset_dir / "schemas",
                    )
                else:
                    report = {
                        "status": "NEED_CLARIFICATION",
                        "errors": [
                            {
                                "error_type": "missing_required_field",
                                "message": "Please specify an executable layout edit operation and target.",
                                "field_path": "semantic_intent.edit_operations",
                            }
                        ],
                        "warnings": [],
                    }
            elif unresolved:
                error_type_by_requirement = {
                    "missing_site_dimensions": "missing_site",
                    "missing_equipment_requirements": "missing_equipment_instances",
                    "missing_active_layout": "missing_active_layout_ref",
                    "unsupported_design_intent": "unsupported_task",
                }
                report = {
                    "status": "NEED_CLARIFICATION",
                    "errors": [
                        {
                            "error_type": error_type_by_requirement.get(
                                item.get("requirement_id"),
                                "missing_required_field",
                            ),
                            "message": item.get("description"),
                            "field_path": item.get("field_path"),
                        }
                        for item in unresolved
                        if item.get("severity") == "blocking"
                    ],
                    "warnings": [],
                }
            else:
                error_type = "edit_operation_in_flp_ir" if intent["intent_type"] == "edit_layout" else "unsupported_task"
                report = {
                    "status": "UNSUPPORTED_TASK",
                    "errors": [
                        {
                            "error_type": error_type,
                            "message": f"{intent['intent_type']} is not represented as FLP-IR in this phase.",
                            "field_path": None,
                        }
                    ],
                    "warnings": [],
                }
        else:
            report = validate_flp_ir(flp_ir, self.adapter, self.dataset_dir / "schemas")

        awaiting_parameters = rough_layout is None and report.get("status") == "NEED_CLARIFICATION"
        if flp_ir is not None and not awaiting_parameters:
            canonical_constraint_spec = copy.deepcopy(
                canonical_constraint_spec
                or CanonicalConstraintCompiler().compile(
                    _compiled_site_snapshot(flp_ir, scene_context),
                    flp_ir=flp_ir,
                    task_contract=task_contract,
                    reference_layout=task_contract.get("base_layout"),
                )
            )
            canonical_validation = CanonicalConstraintValidator().validate(
                rough_layout,
                canonical_constraint_spec,
                reference_layout=task_contract.get("base_layout"),
                expected_instance_ids=task_contract.get(
                    "expected_instance_ids"
                ),
                preserve_instance_ids=task_contract.get(
                    "preserve_instance_ids"
                ),
            )
            task_validation = validate_task_contract(
                rough_layout,
                task_contract,
            )
            flare_fidelity = _validate_flare_fidelity(
                flp_ir,
                task_contract,
                semantic_intent or {},
                scene_context,
            )
            viewport_precheck = copy.deepcopy(
                viewport_precheck
                or _viewport_precheck(
                    rough_layout,
                    flp_ir,
                    task_contract,
                    self.adapter,
                    canonical_spec=canonical_constraint_spec,
                )
            )
            evaluator_recovery = (
                (task_contract.get("metadata") or {}).get(
                    "evaluator_assisted_recovery"
                )
                or {}
            )
            disclosed_task_deviations = (
                copy.deepcopy(task_validation.get("errors") or [])
                if (
                    evaluator_recovery
                    and task_contract.get("task_type") == "DirectiveEdit"
                )
                else []
            )
            typed_violations = [
                *(flare_fidelity.get("violations") or []),
                *(canonical_validation.get("violations") or []),
                *(
                    []
                    if disclosed_task_deviations
                    else (task_validation.get("errors") or [])
                ),
                *_viewport_precheck_violations(viewport_precheck),
            ]
            layout_errors = [
                _typed_violation_as_validation_error(item)
                for item in typed_violations
            ]
            layout_validation = {
                "schema_version": "canonical-layout-validation-v1",
                "status": (
                    "VALID"
                    if not typed_violations
                    else "REPAIR_REQUIRED"
                ),
                "issues": layout_errors,
                "typed_violations": copy.deepcopy(typed_violations),
                "violation_count": len(typed_violations),
                "canonical_hard_validation": canonical_validation,
                "task_contract_validation": task_validation,
                "flare_fidelity_validation": flare_fidelity,
                "disclosed_task_contract_deviations": copy.deepcopy(
                    disclosed_task_deviations
                ),
                "evaluator_assisted_recovery": copy.deepcopy(
                    evaluator_recovery
                ),
                "viewport_precheck": viewport_precheck,
                "canonical_constraint_spec_sha256": canonical_constraint_spec.get(
                    "spec_sha256"
                ),
            }
            if not task_contract.get("complete", True):
                for field_path in task_contract.get("missing_fields") or []:
                    layout_errors.append(
                        {
                            "error_type": "missing_required_field",
                            "code": "TASK_CONTRACT_INCOMPLETE",
                            "message": (
                                "TaskContract remains incomplete after authoritative "
                                f"context resolution: {field_path}."
                            ),
                            "field_path": field_path,
                            "instance_ids": [],
                        }
                    )
            if layout_errors:
                report = {
                    **report,
                    "errors": list(report.get("errors") or []) + layout_errors,
                }
            repair = _validation_recovery_feedback(
                report,
                typed_violations,
                task_contract,
                revision,
            )
            if repair.get("disposition") == "edit_conflict":
                layout_validation["status"] = "EDIT_CONFLICT"
            elif disclosed_task_deviations and not typed_violations:
                layout_validation["status"] = (
                    "VALID_WITH_DISCLOSED_DEVIATION"
                )
                report = {
                    **report,
                    "warnings": list(report.get("warnings") or [])
                    + [
                        {
                            **_typed_violation_as_validation_error(item),
                            "severity": "warning",
                            "disclosed_by": "EvaluatorAgent",
                        }
                        for item in disclosed_task_deviations
                    ],
                }
        else:
            layout_validation = _validate_layout_candidate(rough_layout, flp_ir)
            repair = repair_or_clarify(flp_ir, report)
            repair["source_revision_id"] = f"validation_v{revision}"

        _finalize_validation_status(
            report,
            layout_validation,
            repair,
            flp_present=flp_ir is not None,
        )
        if report.get("status") == "NEED_CLARIFICATION":
            clarification_option_questions = _public_clarification_asset_questions(
                scene_context.get("clarification_asset_options") or []
            )
            if clarification_option_questions:
                repair["clarification_questions"] = clarification_option_questions
                repair["requires_user_clarification"] = True
        revision_record = {
            "revision_index": int(revision),
            "validation_revision_id": f"validation_v{revision}",
            "flp_ir_revision_id": f"flp_ir_v{revision}" if flp_ir is not None else None,
            "layout_revision_id": f"layout_v{revision}" if rough_layout is not None else None,
        }
        payload = {
            "agent_name": self.name,
            "validation_report": report,
            "intent_validation": None,
            "layout_validation": layout_validation,
            "validation_scope": "layout",
            "repair": repair,
            "revision": revision_record,
            "trace": _trace(
                self.name,
                self.responsibility,
                f"flp_ir_v{revision} + layout_v{revision}",
                f"validation_v{revision}",
                iteration=revision,
                input_artifacts=[f"flp_ir_v{revision}"] + ([f"layout_v{revision}"] if rough_layout else []),
                output_artifacts=[
                    f"validation_v{revision}",
                    f"validation_recovery_feedback_v{revision}",
                ],
                decision=repair.get("disposition"),
                evidence=[
                    f"{item.get('error_type')}:{','.join(item.get('instance_ids') or [])}"
                    for item in report.get("errors") or []
                ],
            ),
        }
        return enforce_agent_output(self.name, payload)


def _finalize_validation_status(report, layout_validation, repair, *, flp_present):
    """Keep validation failures visible when deriving the final stage status."""

    current_status = str(report.get("status") or "")
    layout_status = str((layout_validation or {}).get("status") or "")
    blocked_statuses = {
        "UNSUPPORTED_TASK",
        "REPAIR_REQUIRED",
        "NEED_CLARIFICATION",
        "EDIT_CONFLICT",
        "FAILED",
    }
    has_errors = bool(report.get("errors")) or bool(
        (layout_validation or {}).get("issues")
    )
    has_blocked_layout_status = layout_status in blocked_statuses

    if current_status in blocked_statuses or has_errors or has_blocked_layout_status:
        disposition = (repair or {}).get("disposition")
        if disposition == "edit_conflict":
            report["status"] = "EDIT_CONFLICT"
        elif disposition == "clarification_required":
            report["status"] = "NEED_CLARIFICATION"
        elif disposition == "fatal":
            report["status"] = "FAILED"
        elif current_status == "UNSUPPORTED_TASK":
            report["status"] = "UNSUPPORTED_TASK"
        else:
            report["status"] = "REPAIR_REQUIRED"
        return report["status"]

    if flp_present:
        report["status"] = (
            "VALID_WITH_WARNINGS"
            if layout_status == "VALID_WITH_DISCLOSED_DEVIATION"
            else "VALID_SOLVER_READY"
        )
    return report.get("status")


class EvaluatorAgent:
    name = "EvaluatorAgent"
    responsibility = "Evaluate legal layout quality, compare revisions, and emit structured optimization feedback."

    def run(
        self,
        rough_layout,
        validation_report,
        solver_report=None,
        flp_ir=None,
        *,
        previous_evaluation=None,
        intent=None,
        revision=1,
        task_contract=None,
        validator_feedback=None,
    ):
        can_evaluate = rough_layout is not None and validation_report["status"] in VALID_LAYOUT_STATUSES
        if not can_evaluate:
            feedback = _evaluator_blocked_candidate_feedback(
                validation_report,
                validator_feedback or {},
                task_contract or {},
                revision,
                has_layout=rough_layout is not None,
            )
            recovery_proposed = feedback.get("should_revise") is True
            payload = {
                "agent_name": self.name,
                "evaluation_status": (
                    "recovery_proposed"
                    if recovery_proposed
                    else (
                        "clarification_required"
                        if feedback.get("requires_user_clarification")
                        else "not_evaluated"
                    )
                ),
                "metrics": {},
                "comparison": _compare_evaluations({}, previous_evaluation),
                "optimization_feedback": feedback,
                "recovery_notice": copy.deepcopy(
                    feedback.get("user_notice") or {}
                ),
                "notes": [
                    (
                        "Evaluator proposed a minimum-deviation feasible alternative; "
                        "the exact DirectiveEdit remains recorded as unsatisfied."
                        if recovery_proposed
                        else "Evaluation requires a layout that passed hard-constraint validation."
                    )
                ],
                "revision": {
                    "revision_index": int(revision),
                    "evaluation_revision_id": f"evaluation_v{revision}",
                    "layout_revision_id": f"layout_v{revision}" if rough_layout else None,
                },
                "trace": _trace(
                    self.name,
                    self.responsibility,
                    f"layout_v{revision} + validation_v{revision}",
                    f"evaluation_v{revision}",
                    iteration=revision,
                    input_artifacts=[f"validation_v{revision}"],
                    output_artifacts=[
                        f"evaluation_v{revision}",
                        *(
                            [f"evaluator_recovery_feedback_v{revision}"]
                            if recovery_proposed
                            else []
                        ),
                    ],
                    decision=(
                        "propose_disclosed_recovery"
                        if recovery_proposed
                        else "not_evaluated"
                    ),
                ),
            }
            return enforce_agent_output(self.name, payload)

        metrics = _evaluate_layout(rough_layout, solver_report=solver_report, flp_ir=flp_ir)
        comparison = _compare_evaluations(metrics, previous_evaluation)
        feedback = _build_optimization_feedback(
            metrics,
            comparison,
            intent or {},
            solver_report or {},
            revision,
        )
        payload = {
            "agent_name": self.name,
            "evaluation_status": "evaluated",
            "metrics": metrics,
            "comparison": comparison,
            "optimization_feedback": feedback,
            "recovery_notice": {},
            "notes": _evaluation_notes(metrics, comparison, feedback),
            "revision": {
                "revision_index": int(revision),
                "evaluation_revision_id": f"evaluation_v{revision}",
                "layout_revision_id": f"layout_v{revision}",
            },
            "trace": _trace(
                self.name,
                self.responsibility,
                f"layout_v{revision} + validation_v{revision}",
                f"evaluation_v{revision} + optimization_feedback_v{revision}",
                iteration=revision,
                input_artifacts=[f"layout_v{revision}", f"validation_v{revision}"],
                output_artifacts=[f"evaluation_v{revision}", f"optimization_feedback_v{revision}"],
                decision="request_revision" if feedback["should_revise"] else "accept_or_stop",
                evidence=[
                    f"objective={metrics.get('objective_value')}",
                    f"violations={metrics.get('violation_count')}",
                    f"runtime_s={metrics.get('solver_runtime_s')}",
                ],
            ),
        }
        return enforce_agent_output(self.name, payload)


def _execute_intent_validation_loop(
    dataset_dir,
    rag_service,
    llm_client,
    user_message,
    initial_intent_step,
    scene_context,
    *,
    max_iterations=MAX_AGENTIC_ITERATIONS,
    enable_feedback=True,
):
    """Validate a shared initial intent and optionally revise it before design."""

    max_iterations = max(1, int(max_iterations))
    current = copy.deepcopy(initial_intent_step)
    current.setdefault("intent_revision_id", "intent_v1")
    validator = ValidatorAgent(dataset_dir, rag_service=rag_service)
    intent_agent = IntentUnderstandingAgent(llm_client)
    active_layout = scene_context.get("last_layout") or scene_context.get(
        "seed_layout"
    )
    active_layout_id = (active_layout or {}).get("layout_id")
    history = []
    transitions = []
    terminal_validation = None
    termination_reason = None

    for revision in range(1, max_iterations + 1):
        terminal_validation = validator.validate_intent(
            current,
            user_message,
            scene_context=scene_context,
            revision=revision,
        )
        transitions.append(
            {
                "kind": "intent_validation",
                "revision": revision,
                "step": terminal_validation,
                "skipped": False,
            }
        )
        status = terminal_validation["validation_report"]["status"]
        history_entry = {
            "revision_index": revision,
            "intent_revision_id": current.get("intent_revision_id")
            or f"intent_v{revision}",
            "validation_revision_id": (
                terminal_validation.get("revision") or {}
            ).get("validation_revision_id"),
            "status": status,
            "blocking_issue_count": len(
                terminal_validation["validation_report"].get("errors") or []
            ),
            "feedback_executed": False,
        }
        history.append(history_entry)
        if status == "VALID":
            termination_reason = "intent_validated"
            break

        feedback = terminal_validation.get("repair") or {}
        if (
            status == "REPAIRABLE"
            and enable_feedback
            and feedback.get("should_revise")
            and revision < max_iterations
        ):
            history_entry["feedback_executed"] = True
            transitions.append(
                {
                    "kind": "intent_feedback",
                    "revision": revision,
                    "source": "validator",
                    "target": "IntentUnderstandingAgent",
                    "feedback": copy.deepcopy(feedback),
                }
            )
            current = intent_agent.revise(
                user_message,
                current,
                feedback,
                active_layout_id=active_layout_id,
                scene_context=scene_context,
                revision=revision + 1,
            )
            transitions.append(
                {
                    "kind": "intent_revision",
                    "revision": revision + 1,
                    "step": copy.deepcopy(current),
                    "skipped": False,
                }
            )
            continue

        if status == "REPAIRABLE" and not enable_feedback:
            termination_reason = "intent_validator_observation_only"
        elif status == "REPAIRABLE":
            termination_reason = "maximum_intent_revisions_reached"
        else:
            termination_reason = "user_clarification_required"
        break

    return {
        "initial_intent_step": copy.deepcopy(initial_intent_step),
        "intent_step": current,
        "validation": terminal_validation,
        "history": history,
        "transitions": transitions,
        "feedback_rounds": sum(
            item.get("feedback_executed") is True for item in history
        ),
        "validated": bool(
            terminal_validation
            and terminal_validation["validation_report"]["status"] == "VALID"
        ),
        "termination_reason": termination_reason
        or "intent_validation_schedule_complete",
    }


def _execute_design_loop(
    dataset_dir,
    rag_service,
    user_message,
    intent_step,
    scene_context,
    *,
    max_iterations=MAX_AGENTIC_ITERATIONS,
    enable_validator_feedback=True,
    enable_evaluator=True,
    enable_evaluator_feedback=True,
):
    final = None
    transitions = []
    for transition in _iterate_design_loop(
        dataset_dir,
        rag_service,
        user_message,
        intent_step,
        scene_context,
        max_iterations=max_iterations,
        enable_validator_feedback=enable_validator_feedback,
        enable_evaluator=enable_evaluator,
        enable_evaluator_feedback=enable_evaluator_feedback,
    ):
        if transition["kind"] == "complete":
            final = transition["result"]
        else:
            transitions.append(transition)
    if final is None:  # pragma: no cover - protects the state-machine contract.
        raise RuntimeError("Agentic design loop ended without a completion transition.")
    final["transitions"] = transitions
    return final


def _iterate_design_loop(
    dataset_dir,
    rag_service,
    user_message,
    intent_step,
    scene_context,
    *,
    max_iterations=MAX_AGENTIC_ITERATIONS,
    enable_validator_feedback=True,
    enable_evaluator=True,
    enable_evaluator_feedback=True,
):
    max_iterations = max(1, int(max_iterations))
    scene_context = AuthoritativeContextResolver().apply(
        scene_context.get("current_user_input")
        if isinstance(scene_context, dict)
        else user_message,
        scene_context or {},
    )
    intent = intent_step["intent"]
    frozen_semantic_intent = copy.deepcopy(
        intent_step.get("semantic_intent") or {}
    )
    scene_context["frozen_semantic_intent"] = copy.deepcopy(
        frozen_semantic_intent
    )
    scene_context["current_semantic_intent"] = copy.deepcopy(
        frozen_semantic_intent
    )
    task_contract = TaskContractCompiler().compile(
        intent,
        frozen_semantic_intent,
        scene_context["authoritative_context"],
    ).to_dict()
    scene_context["task_contract"] = task_contract
    compiler_message = intent_step.get("compiler_message") or user_message
    designer = DesignerAgent(dataset_dir, rag_service=rag_service)
    validator = ValidatorAgent(dataset_dir, rag_service=rag_service)
    evaluator = EvaluatorAgent() if enable_evaluator else None
    revision_history = []
    best_candidate = None
    best_rank = None
    best_history_index = None
    previous_evaluation = None
    repair_rounds = 0
    optimization_rounds = 0
    converged = False
    quality_converged = False
    termination_reason = None

    intent_type = intent.get("intent_type")
    existing_layout = scene_context.get("last_layout") or scene_context.get("seed_layout")
    existing_flp_ir = scene_context.get("last_flp_ir")
    if intent_type in INSPECTION_INTENTS and existing_layout and existing_flp_ir:
        design = _reused_context_design_step(intent, scene_context, revision=1)
        yield {"kind": "design", "revision": 1, "step": design, "skipped": True}
        yield {"kind": "validation_started", "revision": 1}
        validation = validator.run(
            existing_flp_ir,
            intent,
            existing_layout,
            revision=1,
            scene_context=scene_context,
            semantic_intent=frozen_semantic_intent,
        )
        yield {"kind": "validation", "revision": 1, "step": validation, "skipped": False}
        if intent_type == "evaluate_layout" and enable_evaluator:
            yield {"kind": "evaluation_started", "revision": 1}
            evaluation = evaluator.run(
                existing_layout,
                validation["validation_report"],
                solver_report=scene_context.get("last_solver_report"),
                flp_ir=existing_flp_ir,
                intent=intent,
                revision=1,
            )
            evaluation_skipped = evaluation["evaluation_status"] != "evaluated"
        else:
            evaluation = _skipped_evaluation_step(intent, revision=1, reason="evaluation_not_requested")
            evaluation_skipped = True
        yield {
            "kind": "evaluation",
            "revision": 1,
            "step": evaluation,
            "skipped": evaluation_skipped,
        }
        revision_history.append(_revision_history_entry(design, validation, evaluation))
        status = validation["validation_report"]["status"]
        result = _completed_loop_result(
            design,
            validation,
            evaluation,
            revision_history,
            max_iterations=max_iterations,
            converged=status in VALID_LAYOUT_STATUSES,
            quality_converged=(
                intent_type == "evaluate_layout" and enable_evaluator and not evaluation_skipped
            ),
            termination_reason=(
                "evaluation_complete"
                if intent_type == "evaluate_layout" and enable_evaluator
                else "validation_complete"
            ),
            repair_rounds=0,
            optimization_rounds=0,
            status=status,
        )
        yield {"kind": "complete", "result": result}
        return

    if not intent_step.get("can_enter_designer", False):
        design = _blocked_design_step(intent, intent_step, revision=1)
        yield {"kind": "design", "revision": 1, "step": design, "skipped": True}
        validation = validator.run(
            None,
            intent,
            None,
            revision=1,
            scene_context=scene_context,
            task_contract=task_contract,
            semantic_intent=frozen_semantic_intent,
        )
        yield {"kind": "validation", "revision": 1, "step": validation}
        evaluation = _skipped_evaluation_step(
            intent,
            revision=1,
            reason="no_validated_layout",
        )
        yield {
            "kind": "evaluation",
            "revision": 1,
            "step": evaluation,
            "skipped": True,
        }
        revision_history.append(_revision_history_entry(design, validation, evaluation))
        termination_reason = (
            "unsupported_intent"
            if validation["validation_report"]["status"] == "UNSUPPORTED_TASK"
            else "intent_requires_clarification"
        )
        result = _completed_loop_result(
            design,
            validation,
            evaluation,
            revision_history,
            max_iterations=max_iterations,
            converged=False,
            quality_converged=False,
            termination_reason=termination_reason,
            repair_rounds=0,
            optimization_rounds=0,
            status=validation["validation_report"]["status"],
        )
        yield {"kind": "complete", "result": result}
        return

    yield {"kind": "design_started", "revision": 1, "reason": "initial_design"}
    current_design = designer.run(
        compiler_message,
        intent,
        scene_context=scene_context,
        semantic_intent=intent_step["semantic_intent"],
        revision=1,
    )
    terminal_validation = None
    terminal_evaluation = None

    for revision in range(1, max_iterations + 1):
        yield {"kind": "design", "revision": revision, "step": current_design, "skipped": False}
        yield {"kind": "validation_started", "revision": revision}
        terminal_validation = validator.run(
            current_design.get("flp_ir"),
            intent,
            current_design.get("rough_layout"),
            revision=revision,
            layout_edit_ir=current_design.get("layout_edit_ir"),
            edit_ir_report=current_design.get("edit_ir_report"),
            scene_context=scene_context,
            task_contract=current_design.get("task_contract")
            or task_contract,
            canonical_constraint_spec=current_design.get(
                "canonical_constraint_spec"
            ),
            viewport_precheck=current_design.get("viewport_precheck"),
            semantic_intent=frozen_semantic_intent,
        )
        yield {"kind": "validation", "revision": revision, "step": terminal_validation}
        legal = terminal_validation["validation_report"]["status"] in VALID_LAYOUT_STATUSES
        validator_feedback = terminal_validation.get("repair") or {}
        evaluator_can_recover = (
            enable_evaluator
            and validator_feedback.get("disposition") == "edit_conflict"
        )
        can_call_evaluator = (
            enable_evaluator
            and (
                (
                    current_design.get("rough_layout") is not None
                    and legal
                )
                or evaluator_can_recover
            )
        )
        if can_call_evaluator:
            yield {"kind": "evaluation_started", "revision": revision}
            terminal_evaluation = evaluator.run(
                current_design.get("rough_layout"),
                terminal_validation["validation_report"],
                solver_report=current_design.get("solver_report"),
                flp_ir=current_design.get("flp_ir"),
                previous_evaluation=previous_evaluation,
                intent=intent,
                revision=revision,
                task_contract=current_design.get("task_contract")
                or task_contract,
                validator_feedback=validator_feedback,
            )
        else:
            terminal_evaluation = _skipped_evaluation_step(
                intent,
                revision=revision,
                reason=(
                    "feasible_layout_generated"
                    if intent_type == "generate_feasible_layout"
                    else ("evaluator_disabled" if legal else "no_validated_layout")
                ),
            )
        yield {
            "kind": "evaluation",
            "revision": revision,
            "step": terminal_evaluation,
            "skipped": not can_call_evaluator,
        }

        history_entry = _revision_history_entry(current_design, terminal_validation, terminal_evaluation)
        revision_history.append(history_entry)
        history_index = len(revision_history) - 1
        if legal:
            candidate_rank = _candidate_rank(
                terminal_evaluation,
                current_design,
                terminal_validation,
                intent_step=intent_step,
            )
            can_replace = best_candidate is None or candidate_rank < best_rank
            if (
                can_replace
                and best_candidate is not None
                and enable_evaluator_feedback
            ):
                can_replace = _a3_candidate_can_replace(
                    current_design,
                    best_candidate[0],
                    terminal_evaluation,
                    best_candidate[2],
                    terminal_validation,
                    best_candidate[1],
                )
            if can_replace:
                if best_history_index is not None:
                    revision_history[best_history_index]["retained_as_best"] = False
                best_candidate = (current_design, terminal_validation, terminal_evaluation)
                best_rank = candidate_rank
                best_history_index = history_index
                history_entry["retained_as_best"] = True
                history_entry["candidate_decision"] = "retain_as_best"
            else:
                history_entry["candidate_decision"] = "reject_worse_than_best"

            if not enable_evaluator:
                converged = True
                quality_converged = False
                termination_reason = "a2_feasibility_closed"
                break

            feedback = terminal_evaluation["optimization_feedback"]
            previous_evaluation = terminal_evaluation
            quality_allowed = bool(
                (current_design.get("flp_ir") or {}).get("objectives")
            ) and task_contract.get("task_type") in {"Generate", "Optimize"}
            if feedback.get("should_revise") and quality_allowed:
                if not enable_evaluator_feedback:
                    termination_reason = "a2_quality_feedback_disabled"
                    converged = True
                    break
                if revision >= max_iterations:
                    termination_reason = "maximum_quality_iterations_reached"
                    break
                optimization_rounds += 1
                history_entry["feedback_executed"] = True
                yield {
                    "kind": "feedback",
                    "revision": revision,
                    "source": "evaluator",
                    "feedback": feedback,
                }
                yield {"kind": "design_started", "revision": revision + 1, "reason": "evaluator_feedback"}
                current_design = designer.revise(
                    current_design,
                    intent,
                    feedback,
                    scene_context=scene_context,
                    revision=revision + 1,
                )
                continue
            converged = True
            quality_converged = terminal_evaluation.get("evaluation_status") == "evaluated"
            termination_reason = (
                "a3_quality_no_strict_improvement_required"
                if not quality_allowed
                else "a3_quality_converged"
            )
            break

        recovery = terminal_validation["repair"]
        disposition = recovery.get("disposition")
        evaluator_recovery = (
            terminal_evaluation.get("optimization_feedback") or {}
        )
        if (
            enable_evaluator_feedback
            and evaluator_recovery.get("feedback_type")
            == "evaluator_assisted_recovery"
            and evaluator_recovery.get("should_revise")
            and revision < max_iterations
        ):
            optimization_rounds += 1
            history_entry["feedback_executed"] = True
            history_entry["candidate_decision"] = (
                "evaluator_assisted_recovery_requested"
            )
            yield {
                "kind": "feedback",
                "revision": revision,
                "source": "evaluator",
                "feedback": evaluator_recovery,
            }
            yield {
                "kind": "design_started",
                "revision": revision + 1,
                "reason": "evaluator_assisted_recovery",
            }
            current_design = designer.revise(
                current_design,
                intent,
                evaluator_recovery,
                scene_context=scene_context,
                revision=revision + 1,
            )
            continue
        if disposition == "auto_recovery" and revision < max_iterations:
            if enable_validator_feedback:
                repair_rounds += 1
                history_entry["feedback_executed"] = True
                yield {
                    "kind": "feedback",
                    "revision": revision,
                    "source": "validator",
                    "feedback": recovery,
                }
                yield {
                    "kind": "design_started",
                    "revision": revision + 1,
                    "reason": "validation_driven_recovery",
                }
                current_design = designer.revise(
                    current_design,
                    intent,
                    recovery,
                    scene_context=scene_context,
                    revision=revision + 1,
                )
            else:
                history_entry["bounded_retry_executed"] = True
                history_entry["candidate_decision"] = "validator_observed_no_feedback"
                yield {
                    "kind": "design_started",
                    "revision": revision + 1,
                    "reason": "validator_observe_bounded_retry",
                }
                current_design = designer.retry(
                    current_design,
                    intent,
                    scene_context=scene_context,
                    revision=revision + 1,
                )
            continue
        if disposition == "auto_recovery":
            termination_reason = "maximum_repair_iterations_reached"
        elif disposition == "edit_conflict":
            termination_reason = "edit_conflict"
        elif disposition == "clarification_required":
            termination_reason = "user_clarification_required"
        elif disposition == "fatal":
            termination_reason = "unrecoverable_validation_failure"
        else:
            termination_reason = "validation_blocked"
        break

    if best_candidate is not None:
        final_design, final_validation, final_evaluation = best_candidate
        status = final_validation["validation_report"]["status"]
        if not converged and termination_reason is None:
            termination_reason = "best_legal_candidate_retained"
    else:
        final_design = current_design
        final_validation = terminal_validation
        final_evaluation = terminal_evaluation
        status = (final_validation or {}).get("validation_report", {}).get("status") or "FAILED"
        if termination_reason == "maximum_repair_iterations_reached":
            status = "FAILED_MAX_ITERATIONS"

    if best_history_index is not None:
        revision_history[best_history_index]["selected_for_output"] = True
    result = _completed_loop_result(
        final_design,
        final_validation,
        final_evaluation,
        revision_history,
        max_iterations=max_iterations,
        converged=converged,
        quality_converged=quality_converged,
        termination_reason=termination_reason or "state_machine_complete",
        repair_rounds=repair_rounds,
        optimization_rounds=optimization_rounds,
        status=status,
    )
    yield {"kind": "complete", "result": result}


def _reused_context_design_step(intent, scene_context, *, revision):
    flp_ir = copy.deepcopy(scene_context.get("last_flp_ir"))
    rough_layout = copy.deepcopy(scene_context.get("last_layout") or scene_context.get("seed_layout"))
    solver_report = copy.deepcopy(scene_context.get("last_solver_report"))
    payload = {
        "agent_name": "DesignerAgent",
        "design_status": "reused_existing_layout",
        "handoff": None,
        "evidence_package": None,
        "flp_ir": flp_ir,
        "rough_layout": rough_layout,
        "solver_report": solver_report,
        "solver_used": (solver_report or {}).get("solver_used") or (rough_layout or {}).get("solver_used"),
        "design_basis": {
            "intent_type": intent.get("intent_type"),
            "protected_user_constraints": [],
            "skip_reason": "The current task consumes the active layout without redesigning it.",
        },
        "solver_evidence": None,
        "applied_feedback": [],
        "revision": _design_revision(
            revision,
            flp_ir,
            rough_layout,
            None,
            parent_layout_id=(rough_layout or {}).get("layout_id"),
        ),
        "trace": _trace(
            "DesignerAgent",
            DesignerAgent.responsibility,
            "active_layout",
            "existing_layout_reused",
            iteration=revision,
            input_artifacts=["active_layout"],
            output_artifacts=["existing_layout_reused"],
            decision="not_invoked_for_inspection_task",
        ),
    }
    return enforce_agent_output("DesignerAgent", payload)


def _skipped_evaluation_step(intent, *, revision, reason):
    payload = {
        "agent_name": "EvaluatorAgent",
        "evaluation_status": "not_evaluated",
        "metrics": {},
        "comparison": _compare_evaluations({}, None),
        "optimization_feedback": {
            "source": "evaluator",
            "source_revision_id": f"evaluation_v{revision}",
            "should_revise": False,
            "issue_types": [],
            "actions": [],
            "stop_reason": reason,
        },
        "notes": ["Evaluation was not invoked because it is not required for this task route."],
        "revision": {
            "revision_index": int(revision),
            "evaluation_revision_id": f"evaluation_v{revision}",
            "layout_revision_id": None,
        },
        "trace": _trace(
            "EvaluatorAgent",
            EvaluatorAgent.responsibility,
            "intent_v1",
            "evaluation_skipped",
            iteration=revision,
            input_artifacts=["intent_v1"],
            output_artifacts=["evaluation_skipped"],
            decision=f"not_invoked:{reason}",
        ),
    }
    return enforce_agent_output("EvaluatorAgent", payload)


def _blocked_design_step(intent, intent_step, *, revision):
    handoff = "layout_edit_ir" if intent.get("intent_type") == "edit_layout" else None
    payload = {
        "agent_name": "DesignerAgent",
        "design_status": "skipped_by_intent_contract",
        "handoff": handoff,
        "evidence_package": None,
        "flp_ir": None,
        "rough_layout": None,
        "solver_report": None,
        "solver_used": None,
        "design_basis": {
            "intent_type": intent.get("intent_type"),
            "protected_user_constraints": [],
            "skip_reason": "Intent contract forbids design while blocking requirements remain.",
        },
        "solver_evidence": None,
        "applied_feedback": [],
        "revision": _design_revision(revision, None, None, None),
        "trace": _trace(
            "DesignerAgent",
            DesignerAgent.responsibility,
            "intent_v1",
            "design_skipped",
            iteration=revision,
            input_artifacts=["intent_v1"],
            output_artifacts=["design_skipped"],
            decision="not_invoked_due_to_intent_contract",
            evidence=[item.get("requirement_id") for item in intent_step.get("unresolved_requirements") or []],
        ),
    }
    return enforce_agent_output("DesignerAgent", payload)


def _revision_history_entry(design, validation, evaluation):
    revision = int((design.get("revision") or {}).get("revision_index") or 1)
    metrics = evaluation.get("metrics") or {}
    report = design.get("solver_report") or {}
    validation_report = validation.get("validation_report") or {}
    layout_validation = validation.get("layout_validation") or {}
    hard_validation = layout_validation.get("canonical_hard_validation") or {}
    task_validation = layout_validation.get("task_contract_validation") or {}
    viewport_precheck = (
        layout_validation.get("viewport_precheck")
        or design.get("viewport_precheck")
        or {}
    )
    return {
        "revision_index": revision,
        "parent_layout_id": (design.get("revision") or {}).get("parent_layout_id"),
        "artifacts": {
            "layout_edit_ir": (
                f"layout_edit_ir_v{revision}" if design.get("layout_edit_ir") is not None else None
            ),
            "flp_ir": (design.get("revision") or {}).get("flp_ir_revision_id"),
            "layout": (design.get("revision") or {}).get("layout_revision_id"),
            "validation": (validation.get("revision") or {}).get("validation_revision_id"),
            "evaluation": (evaluation.get("revision") or {}).get("evaluation_revision_id"),
        },
        "design_status": design.get("design_status"),
        "validation_status": validation_report.get("status"),
        "evaluation_status": evaluation.get("evaluation_status"),
        "recovery_disposition": (
            validation.get("repair") or {}
        ).get("disposition"),
        "solver_used": design.get("solver_used"),
        "solver_budget": {
            "solver_options": copy.deepcopy(report.get("solver_options") or {}),
            "planned_generations": (report.get("solver_options") or {}).get("generations"),
            "generations_completed": report.get("generations_completed"),
            "evaluated_individuals": report.get("evaluated_individuals"),
            "runtime_s": report.get("runtime_s"),
        },
        "mutation_summary": copy.deepcopy(
            (design.get("design_basis") or {}).get("mutation_summary") or {}
        ),
        "objective_value": metrics.get("objective_value", report.get("objective_value")),
        "violation_count": len(validation_report.get("errors") or []),
        "hard_violation_count": hard_validation.get("violation_count"),
        "task_contract_satisfaction": task_validation.get("satisfaction"),
        "viewport_precheck_valid": viewport_precheck.get("valid"),
        "applied_feedback": copy.deepcopy(design.get("applied_feedback") or []),
        "feedback_executed": False,
        "bounded_retry_executed": False,
        "candidate_decision": "not_legal",
        "retained_as_best": False,
        "selected_for_output": False,
    }


def _candidate_rank(
    evaluation,
    design=None,
    validation=None,
    *,
    intent_step=None,
):
    design = design or {}
    validation = validation or {}
    contract = design.get("task_contract") or {}
    contract_complete = bool(
        contract.get(
            "complete",
            (intent_step or {}).get("can_enter_designer", False),
        )
    )
    layout_generated = design.get("rough_layout") is not None
    layout_validation = validation.get("layout_validation") or {}
    hard_validation = layout_validation.get("canonical_hard_validation") or {}
    hard_violations = int(
        hard_validation.get(
            "violation_count",
            layout_validation.get("violation_count", 0),
        )
        or 0
    )
    task_validation = layout_validation.get("task_contract_validation") or {}
    task_satisfaction = float(task_validation.get("satisfaction") or 0.0)
    precheck = (
        layout_validation.get("viewport_precheck")
        or design.get("viewport_precheck")
        or {}
    )
    viewport_valid = bool(precheck.get("valid") or precheck.get("status") == "VALID")
    metrics = evaluation.get("metrics") or {}
    objective = _optional_float(metrics.get("objective_value"))
    if objective is None:
        report = design.get("solver_report") or {}
        objective = _optional_float(report.get("objective_value"))
    runtime = float((design.get("solver_report") or {}).get("runtime_s") or 0.0)
    return (
        0 if contract_complete else 1,
        0 if layout_generated else 1,
        hard_violations,
        -task_satisfaction,
        0 if viewport_valid else 1,
        float("inf") if objective is None else objective,
        runtime,
    )


def _compile_existing_layout_flp_ir(compiled, authoritative, *, reference_layout=None, user_message=None):
    """Preserve the existing IR contract while adding requested objectives/flows."""

    if not isinstance(authoritative, dict) or not authoritative:
        return compiled
    result = copy.deepcopy(authoritative)
    # New task metadata and unresolved parameter questions belong to the
    # current request. Copying only objectives silently discarded both.
    for field in ("task_meta", "optimization_scope", "solver_request", "unresolved_items"):
        if field in (compiled or {}):
            result[field] = copy.deepcopy(compiled[field])
    result["flow_relations"] = copy.deepcopy(
        (compiled or {}).get("flow_relations") or []
    )
    result["objectives"] = copy.deepcopy(
        (compiled or {}).get("objectives") or []
    )
    for severity in ("hard","soft"):
        retained=result.setdefault("constraints",{}).setdefault(severity,[])
        for constraint in ((compiled or {}).get("constraints") or {}).get(severity,[]):
            identity=lambda c:(c.get("constraint_type"),c.get("target_refs"),c.get("parameters"))
            if not any(identity(c)==identity(constraint) for c in retained):
                added=copy.deepcopy(constraint)
                added["constraint_id"]=f"constraint_{severity}_{len(retained)+1:03d}"
                retained.append(added)
    if (compiled or {}).get("source_request"):
        result["source_request"] = copy.deepcopy(compiled["source_request"])
    source = str(user_message or (result.get("source_request") or {}).get("raw_user_message") or "")
    if reference_layout and re.search(r"\bkeep orientations unchanged\b", source, re.I):
        poses = {p["instance_id"]: p for p in reference_layout.get("placements") or []}
        for item in result.get("equipment_instances") or []:
            if item["instance_id"] in poses:
                item["allowed_orientations_deg"] = [float((poses[item["instance_id"]].get("rotation") or {}).get("yaw_deg", 0))]
    return result


def _strict_objective_improvement(
    candidate_design,
    incumbent_design,
    candidate_evaluation,
    incumbent_evaluation,
):
    candidate = _optional_float(
        (candidate_evaluation.get("metrics") or {}).get("objective_value")
    )
    incumbent = _optional_float(
        (incumbent_evaluation.get("metrics") or {}).get("objective_value")
    )
    if candidate is None:
        candidate = _optional_float(
            (candidate_design.get("solver_report") or {}).get(
                "objective_value"
            )
        )
    if incumbent is None:
        incumbent = _optional_float(
            (incumbent_design.get("solver_report") or {}).get(
                "objective_value"
            )
        )
    return (
        candidate is not None
        and incumbent is not None
        and candidate < incumbent - GEOMETRY_EPSILON_M
    )


def _a3_candidate_can_replace(
    candidate_design,
    incumbent_design,
    candidate_evaluation,
    incumbent_evaluation,
    candidate_validation,
    incumbent_validation,
):
    """Fail closed unless quality improves and all A2 validity axes do not regress."""

    candidate_axes = _validated_candidate_axes(candidate_validation)
    incumbent_axes = _validated_candidate_axes(incumbent_validation)
    if candidate_axes is None or incumbent_axes is None:
        return False
    hard_non_regression = (
        candidate_axes["hard_violation_count"]
        <= incumbent_axes["hard_violation_count"]
    )
    task_non_regression = (
        candidate_axes["task_contract_satisfaction"]
        + GEOMETRY_EPSILON_M
        >= incumbent_axes["task_contract_satisfaction"]
    )
    viewport_non_regression = (
        candidate_axes["viewport_valid"] or not incumbent_axes["viewport_valid"]
    )
    flare_contract_unchanged = (
        _a3_flare_contract_identity(candidate_design)
        == _a3_flare_contract_identity(incumbent_design)
    )
    return bool(
        hard_non_regression
        and task_non_regression
        and viewport_non_regression
        and flare_contract_unchanged
        and _strict_objective_improvement(
            candidate_design,
            incumbent_design,
            candidate_evaluation,
            incumbent_evaluation,
        )
    )


def _a3_flare_contract_identity(design):
    design = design or {}
    task_contract = design.get("task_contract") or {}
    flp_ir = design.get("flp_ir") or {}
    return {
        "task_type": task_contract.get("task_type"),
        "expected_instance_ids": sorted(
            task_contract.get("expected_instance_ids") or []
        ),
        "movable_instance_ids": sorted(
            task_contract.get("movable_instance_ids") or []
        ),
        "preserve_instance_ids": sorted(
            task_contract.get("preserve_instance_ids") or []
        ),
        "instance_bindings": sorted(
            (
                str(item.get("instance_id")),
                str(item.get("model_asset_id")),
            )
            for item in flp_ir.get("equipment_instances") or []
            if isinstance(item, dict)
        ),
        "hard_constraints": sorted(
            str(item.get("constraint_type"))
            for item in (flp_ir.get("constraints") or {}).get("hard", [])
            if isinstance(item, dict)
        ),
        "objectives": sorted(
            str(item.get("objective_type"))
            for item in flp_ir.get("objectives") or []
            if isinstance(item, dict)
        ),
    }


def _validated_candidate_axes(validation):
    layout_validation = (validation or {}).get("layout_validation") or {}
    hard = layout_validation.get("canonical_hard_validation") or {}
    task = layout_validation.get("task_contract_validation") or {}
    viewport = layout_validation.get("viewport_precheck") or {}
    hard_count = hard.get("violation_count")
    task_satisfaction = task.get("satisfaction")
    viewport_valid = viewport.get("valid")
    if viewport_valid is None:
        viewport_valid = (
            True
            if viewport.get("status") == "VALID"
            else False if viewport.get("status") == "INVALID" else None
        )
    try:
        hard_count = int(hard_count)
        task_satisfaction = float(task_satisfaction)
    except (TypeError, ValueError):
        return None
    if not isinstance(viewport_valid, bool):
        return None
    return {
        "hard_violation_count": hard_count,
        "task_contract_satisfaction": task_satisfaction,
        "viewport_valid": viewport_valid,
    }


def _feasibility_preflight(flp_ir, canonical_spec, task_contract):
    instances = list((flp_ir or {}).get("equipment_instances") or [])
    regions = list((canonical_spec or {}).get("allowed_placement_regions") or [])
    constraints = (canonical_spec or {}).get("instance_constraints") or {}
    infeasible = []
    for instance in instances:
        instance_id = instance.get("instance_id")
        dimensions = (instance.get("geometry") or {}).get("dimensions_m") or {}
        try:
            width = float(dimensions["width_m"])
            depth = float(dimensions["depth_m"])
        except (KeyError, TypeError, ValueError):
            infeasible.append(
                {
                    "instance_id": instance_id,
                    "reason": "invalid_geometry",
                }
            )
            continue
        allowed_ids = set(
            (constraints.get(instance_id) or {}).get("allowed_region_ids") or []
        )
        allowed = [
            item
            for item in regions
            if not allowed_ids or item.get("region_id") in allowed_ids
        ]
        if not any(
            width <= float(item.get("width_m") or 0.0) + GEOMETRY_EPSILON_M
            and depth <= float(item.get("depth_m") or 0.0) + GEOMETRY_EPSILON_M
            for item in allowed
        ):
            infeasible.append(
                {
                    "instance_id": instance_id,
                    "reason": "no_allowed_region_can_contain_instance",
                }
            )
    contract_missing = list((task_contract or {}).get("missing_fields") or [])
    return {
        "schema_version": "feasibility-preflight-v1",
        "status": (
            "FEASIBLE"
            if not infeasible and not contract_missing and instances
            else "INFEASIBLE_OR_INCOMPLETE"
        ),
        "task_contract_complete": not contract_missing,
        "task_contract_missing_fields": contract_missing,
        "infeasible_instances": infeasible,
        "instance_count": len(instances),
        "canonical_constraint_spec_sha256": (
            canonical_spec or {}
        ).get("spec_sha256"),
    }


def _viewport_precheck(
    layout,
    flp_ir,
    task_contract,
    adapter,
    *,
    canonical_spec=None,
):
    expected_models = {
        item.get("instance_id"): item.get("model_asset_id")
        for item in (flp_ir or {}).get("equipment_instances") or []
        if item.get("instance_id")
    }
    fixed_equipment = (canonical_spec or {}).get(
        "fixed_equipment_instances"
    ) or []
    expected_models.update(
        {
            item.get("instance_id"): (
                item.get("model_asset_id")
                or item.get("source_model_asset_id")
            )
            for item in fixed_equipment
            if item.get("instance_id")
        }
    )
    expected_instance_ids = list(
        (task_contract or {}).get("expected_instance_ids")
        or sorted(
            item.get("instance_id")
            for item in (flp_ir or {}).get("equipment_instances") or []
            if item.get("instance_id")
        )
    )
    expected_instance_ids.extend(
        item.get("instance_id")
        for item in fixed_equipment
        if item.get("instance_id")
        and item.get("instance_id") not in expected_instance_ids
    )
    return ViewportPrecheck().run(
        layout,
        expected_instance_ids=expected_instance_ids,
        expected_model_by_instance=expected_models,
        available_model_asset_ids=set(adapter.assets_by_id),
    )


def _final_violation_count(validation, evaluation):
    counts = []
    metrics = evaluation.get("metrics") or {}
    if metrics.get("violation_count") is not None:
        counts.append(int(metrics["violation_count"]))
    layout_validation = validation.get("layout_validation") or {}
    if layout_validation.get("violation_count") is not None:
        counts.append(int(layout_validation["violation_count"]))
    counts.append(
        len((validation.get("validation_report") or {}).get("errors") or [])
    )
    return max(counts) if counts else 0


def _completed_loop_result(
    design,
    validation,
    evaluation,
    revision_history,
    *,
    max_iterations,
    converged,
    quality_converged,
    termination_reason,
    repair_rounds,
    optimization_rounds,
    status,
):
    selected_revision = next(
        (item["revision_index"] for item in revision_history if item.get("selected_for_output")),
        (design.get("revision") or {}).get("revision_index"),
    )
    artifact_chain = ["intent_v1"]
    for item in revision_history:
        for artifact in item["artifacts"].values():
            if artifact and artifact not in artifact_chain:
                artifact_chain.append(artifact)
    solver_budgets = [
        item["solver_budget"]
        for item in revision_history
        if item.get("solver_used") and isinstance(item.get("solver_budget"), dict)
    ]
    round_schedule = [
        {
            "revision_index": item["revision_index"],
            "planned_generations": (item.get("solver_budget") or {}).get(
                "planned_generations"
            ),
            "actual_generations": (item.get("solver_budget") or {}).get(
                "generations_completed"
            ),
            "rng_seed": ((item.get("solver_budget") or {}).get("solver_options") or {}).get(
                "seed"
            ),
            "mutation_summary": copy.deepcopy(item.get("mutation_summary") or {}),
        }
        for item in revision_history
    ]
    return {
        "status": status,
        "design": design,
        "validation": validation,
        "evaluation": evaluation,
        "iterations": revision_history,
        "revision_history": revision_history,
        "loop_report": {
            "contract_version": CONTRACT_VERSION,
            "state_machine": (
                "ResolveAuthoritativeContext->CompileTaskContract->"
                "CompileCanonicalConstraints->FeasibilityPreflight->"
                "ProduceCandidate->CanonicalHardValidate->"
                "TaskContractValidate->ViewportPrecheck->TypedRepair->"
                "A2BrowserReceipt->A3BlockedCandidateReview->"
                "A3QualityOptimize->NonRegressionCheck->BrowserReceipt->Accept"
            ),
            "candidate_selection": [
                "clarification_or_task_contract_complete",
                "layout_generated",
                "hard_constraint_violation_count",
                "task_contract_satisfaction",
                "viewport_validity",
                "objective_value",
                "runtime",
            ],
            "a2_responsibility": "complete_feasibility_loop",
            "a3_replacement_policy": (
                "replace_only_if_hard_task_viewport_non_regression_and_"
                "strict_objective_improvement"
            ),
            "a3_blocked_candidate_policy": (
                "remove_false_clarification_from_complete_authoritative_contract;"
                "otherwise_keep_clarification;"
                "for_edit_conflict_produce_nearest_feasible_ga_alternative_"
                "with_mandatory_deviation_disclosure"
            ),
            "max_iterations": int(max_iterations),
            "iterations": len(revision_history),
            "converged": bool(converged),
            "legal_solution_found": any(
                item.get("validation_status") in VALID_LAYOUT_STATUSES for item in revision_history
            ),
            "quality_converged": bool(quality_converged),
            "termination_reason": termination_reason,
            "repair_rounds": int(repair_rounds),
            "optimization_rounds": int(optimization_rounds),
            "selected_revision": selected_revision,
            "attempted_revisions": [item["revision_index"] for item in revision_history],
            "solver_execution": {
                "calls": len(solver_budgets),
                "configured_generations": sum(
                    int((item.get("solver_options") or {}).get("generations") or 0)
                    for item in solver_budgets
                ),
                "completed_generations": sum(
                    int(item.get("generations_completed") or 0) for item in solver_budgets
                ),
                "evaluated_individuals": sum(
                    int(item.get("evaluated_individuals") or 0) for item in solver_budgets
                ),
                "runtime_s": sum(float(item.get("runtime_s") or 0.0) for item in solver_budgets),
            },
            "round_schedule": round_schedule,
            "artifact_chain": artifact_chain,
        },
    }


class RoughDesignOrchestrator:
    def __init__(self, dataset_dir, rag_service=None, intent_llm_client=None):
        self.dataset_dir = Path(dataset_dir)
        self.rag_service = rag_service
        self.intent_llm_client = (
            intent_llm_client if intent_llm_client is not None else IntentLLMClient.from_env()
        )

    def run(self, user_message, active_layout_id=None, scene_context=None):
        scene_context = dict(scene_context or {})
        scene_context["fresh_rag_asset_surfaces"] = []
        engineering_context = str(scene_context.get("engineering_context") or "")
        engineering_context_kwargs = (
            {"engineering_context": engineering_context}
            if engineering_context
            else {}
        )
        source_history = engineering_context_source_history(
            scene_context.get("conversation_history"), engineering_context
        )
        if not active_layout_id and scene_context.get("last_layout"):
            active_layout_id = scene_context["last_layout"].get("layout_id")
        raw_linguistic_frame = self.intent_llm_client.extract_layout_slots(
            user_message,
            active_layout_id=active_layout_id,
            conversation_history=scene_context.get("conversation_history")
            or [],
            instance_directory=_current_instance_directory(scene_context),
            **engineering_context_kwargs,
        )
        linguistic_frame = _normalize_linguistic_frame(
            raw_linguistic_frame,
            user_message,
            conversation_history=source_history,
            site_features=_known_site_features(scene_context),
        )
        scene_context["linguistic_frame"] = linguistic_frame
        explicit_site_repair = _explicit_central_aisle_width_repair(
            user_message
        )
        asset_surfaces = new_equipment_retrieval_surfaces(
            linguistic_frame,
            user_message=user_message,
            engineering_context=engineering_context,
            has_accepted_layout=bool(
                scene_context.get("last_layout") or scene_context.get("seed_layout")
            ),
            instance_directory=_current_instance_directory(scene_context),
            cached_grounding=scene_context.get("engineering_rag_grounding"),
        )
        existing_grounding = scene_context.get("engineering_rag_grounding")
        should_retrieve = existing_grounding is None or bool(asset_surfaces)
        if (
            should_retrieve
            and hasattr(self.rag_service, "query")
            and not (
                explicit_site_repair
                and not _linguistic_asset_surfaces(linguistic_frame)
            )
        ):
            rag_result = self.rag_service.query(
                _engineering_rag_query_text(engineering_context, user_message),
                top_k=8,
                asset_mentions=asset_surfaces or None,
            )
            retrieved_grounding = _monolithic_rag_snapshot(
                rag_result, self.rag_service
            )
            scene_context["engineering_rag_grounding"] = _merge_monolithic_rag_snapshots(
                existing_grounding, retrieved_grounding
            )
            scene_context["fresh_rag_asset_surfaces"] = copy.deepcopy(asset_surfaces)
        initial_intent_step = IntentUnderstandingAgent(
            self.intent_llm_client
        ).run(
            user_message,
            active_layout_id=active_layout_id,
            scene_context=scene_context,
        )
        intent_validation_loop = _execute_intent_validation_loop(
            self.dataset_dir,
            self.rag_service,
            self.intent_llm_client,
            user_message,
            initial_intent_step,
            scene_context,
            max_iterations=MAX_AGENTIC_ITERATIONS,
            enable_feedback=True,
        )
        intent_step = intent_validation_loop["intent_step"]
        intent = intent_step["intent"]
        loop = _execute_design_loop(
            self.dataset_dir,
            self.rag_service,
            user_message,
            intent_step,
            scene_context,
        )
        designer_step = loop["design"]
        validator_step = loop["validation"]
        evaluator_step = loop["evaluation"]
        status = loop["status"]
        agent_trace = [initial_intent_step["trace"]] + [
            transition["step"]["trace"]
            for transition in intent_validation_loop["transitions"]
            if transition["kind"] in {"intent_revision", "intent_validation"}
        ] + [
            transition["step"]["trace"]
            for transition in loop["transitions"]
            if transition["kind"] in {"design", "validation", "evaluation"}
        ]
        agent_outputs = {
            "IntentUnderstandingAgent": [initial_intent_step],
            "DesignerAgent": [],
            "ValidatorAgent": [
                transition["step"]
                for transition in intent_validation_loop["transitions"]
                if transition["kind"] == "intent_validation"
            ],
            "EvaluatorAgent": [],
        }
        agent_outputs["IntentUnderstandingAgent"].extend(
            transition["step"]
            for transition in intent_validation_loop["transitions"]
            if transition["kind"] == "intent_revision"
        )
        output_agent_by_kind = {
            "design": "DesignerAgent",
            "validation": "ValidatorAgent",
            "evaluation": "EvaluatorAgent",
        }
        for transition in loop["transitions"]:
            agent_name = output_agent_by_kind.get(transition["kind"])
            if agent_name:
                agent_outputs[agent_name].append(transition["step"])
        layout_response = {
            "status": status,
            "intent": intent,
            "semantic_intent": intent_step["semantic_intent"],
            "intent_validation": intent_validation_loop["validation"],
            "intent_revision_history": intent_validation_loop["history"],
            "layout_edit_ir": designer_step.get("layout_edit_ir"),
            "layout_diff": designer_step.get("layout_diff"),
            "flp_ir": designer_step["flp_ir"],
            "rough_layout": designer_step["rough_layout"],
            "validation_report": validator_step["validation_report"],
            "layout_validation": copy.deepcopy(
                validator_step.get("layout_validation") or {}
            ),
            "clarification_questions": validator_step["repair"]["clarification_questions"],
            "solver_report": designer_step.get("solver_report"),
            "solver_used": designer_step.get("solver_used"),
            "evaluation": {
                "evaluation_status": evaluator_step["evaluation_status"],
                "metrics": evaluator_step["metrics"],
                "comparison": evaluator_step["comparison"],
                "optimization_feedback": evaluator_step["optimization_feedback"],
                "recovery_notice": copy.deepcopy(
                    evaluator_step.get("recovery_notice") or {}
                ),
            },
            "loop_report": loop["loop_report"],
            "revision_history": loop["revision_history"],
        }
        result = {
            "status": status,
            "solver_used": designer_step.get("solver_used"),
            "intent": intent,
            "semantic_intent": intent_step["semantic_intent"],
            "initial_intent": initial_intent_step["intent"],
            "initial_semantic_intent": initial_intent_step["semantic_intent"],
            "intent_validation": intent_validation_loop["validation"],
            "intent_revision_history": intent_validation_loop["history"],
            "design": {
                "design_status": designer_step["design_status"],
                "handoff": designer_step["handoff"],
                "evidence_package": designer_step["evidence_package"],
                "layout_edit_ir": designer_step.get("layout_edit_ir"),
                "layout_diff": designer_step.get("layout_diff"),
                "flp_ir": designer_step["flp_ir"],
                "rough_layout": designer_step["rough_layout"],
                "solver_report": designer_step.get("solver_report"),
                "solver_used": designer_step.get("solver_used"),
                "design_basis": designer_step.get("design_basis"),
                "solver_evidence": designer_step.get("solver_evidence"),
                "applied_feedback": designer_step.get("applied_feedback"),
                "task_contract": copy.deepcopy(
                    designer_step.get("task_contract") or {}
                ),
                "revision": designer_step.get("revision"),
            },
            "validation": {
                "validation_report": validator_step["validation_report"],
                "layout_validation": copy.deepcopy(
                    validator_step.get("layout_validation") or {}
                ),
                "repair": validator_step["repair"],
            },
            "evaluation": {
                "evaluation_status": evaluator_step["evaluation_status"],
                "metrics": evaluator_step["metrics"],
                "notes": evaluator_step["notes"],
                "comparison": evaluator_step["comparison"],
                "optimization_feedback": evaluator_step["optimization_feedback"],
                "recovery_notice": copy.deepcopy(
                    evaluator_step.get("recovery_notice") or {}
                ),
                "revision": evaluator_step["revision"],
            },
            "agent_trace": agent_trace,
            "agent_outputs": agent_outputs,
            "loop_report": loop["loop_report"],
            "revision_history": loop["revision_history"],
            "paper_agent_cards": build_paper_agent_cards(),
            "layout_response": layout_response,
        }
        _attach_assistant_message(result, self.intent_llm_client, user_message)
        return result


class AgenticLayOptOrchestrator:
    def __init__(self, dataset_dir, rag_service=None, intent_llm_client=None):
        self.dataset_dir = Path(dataset_dir)
        self.rag_service = rag_service
        self.intent_llm_client = (
            intent_llm_client if intent_llm_client is not None else IntentLLMClient.from_env()
        )
        self.sessions = {}

    def stream(
        self,
        user_message,
        *,
        session_id,
        scene_context=None,
        active_layout_id=None,
        engineering_context=None,
    ):
        session = self.sessions.setdefault(
            session_id,
            {
                "turn_id": 0,
                "active_layout_id": active_layout_id,
                "selected_instance_ids": [],
                "last_domain_asset_candidates": [],
                "last_layout": None,
                "last_flp_ir": None,
                "last_solver_report": None,
                "pending_request_context": None,
                "pending_intent_state": None,
                "engineering_context": "",
                "engineering_rag_grounding": None,
                "retrieved_asset_surfaces": [],
                "retrieved_inventory_context": None,
                "engineering_rag_last_retrieved_turn_id": None,
                "engineering_rag_retrieval_history": [],
                "turn_history": [],
                "latest_viewport_receipt": None,
                "pending_viewport_receipts": {},
            },
        )
        session["turn_id"] += 1
        turn_id = session["turn_id"]
        scene_context = dict(scene_context or {})
        incoming_context = (
            engineering_context
            if engineering_context is not None
            else scene_context.get("engineering_context")
        )
        scene_context["engineering_context"] = persist_engineering_context(
            session, incoming_context
        )
        if session.get("engineering_rag_grounding"):
            scene_context["engineering_rag_grounding"] = copy.deepcopy(
                session["engineering_rag_grounding"]
            )
        elif scene_context.get("engineering_rag_grounding"):
            session["engineering_rag_grounding"] = copy.deepcopy(
                scene_context["engineering_rag_grounding"]
            )
        if active_layout_id:
            session["active_layout_id"] = active_layout_id
        if scene_context.get("selected_instance_ids"):
            session["selected_instance_ids"] = list(scene_context["selected_instance_ids"])
        if session.get("last_layout"):
            scene_context.setdefault("last_layout", session["last_layout"])
            scene_context.setdefault("seed_layout", session["last_layout"])
            session["active_layout_id"] = session["last_layout"].get("layout_id")
        if session.get("last_flp_ir"):
            scene_context.setdefault("last_flp_ir", session["last_flp_ir"])
        if session.get("initial_layout"):
            scene_context.setdefault("initial_layout", copy.deepcopy(session["initial_layout"]))
        if session.get("last_solver_report"):
            scene_context.setdefault("last_solver_report", session["last_solver_report"])

        effective_active_layout_id = active_layout_id or session.get("active_layout_id")
        pending_context = session.get("pending_request_context")
        pending_intent_state = session.get("pending_intent_state")
        scene_context["current_user_input"] = (
            pending_context if pending_context else str(user_message or "")
        )
        pending_asset_options = (
            pending_intent_state.get("clarification_asset_options") or []
            if pending_intent_state
            else []
        )
        confirmed_asset_choices = _merge_confirmed_asset_choices(
            (
                pending_intent_state.get("confirmed_asset_choices") or []
                if pending_intent_state
                else []
            ),
            _confirmed_clarification_asset_choices(
                user_message,
                pending_asset_options,
            ),
        )
        history_context = _compact_conversation_history(session.get("turn_history"))
        scene_context["conversation_history"] = copy.deepcopy(history_context)
        if pending_intent_state:
            scene_context["pending_intent_state"] = copy.deepcopy(
                pending_intent_state
            )
        if confirmed_asset_choices:
            scene_context["confirmed_asset_choices"] = copy.deepcopy(
                confirmed_asset_choices
            )
        intent_message = _conversation_intent_message(
            user_message,
            pending_context=pending_context,
        )
        if pending_context:
            # Newly confirmed engineering requirements must reach compilation.
            scene_context["current_user_input"] = intent_message
        turn_context = _turn_context(history_context, pending_context=pending_context)

        yield _event("session_started", session_id, turn_id, "AgenticLayOpt", "running", "Session turn started.")

        yield _event(
            "intent_slots_started",
            session_id,
            turn_id,
            "IntentUnderstandingAgent",
            "running",
            "Separating equipment, constraints, objectives, process relations and edits.",
        )
        slot_history = copy.deepcopy(history_context)
        if pending_intent_state:
            slot_history.append(
                {
                    "pending_request": pending_context,
                    "unresolved_requirements": copy.deepcopy(
                        pending_intent_state.get("unresolved_requirements")
                        or []
                    ),
                    "clarification_questions": copy.deepcopy(
                        pending_intent_state.get("clarification_questions")
                        or []
                    ),
                    "clarification_choice_labels": _public_clarification_asset_questions(
                        pending_asset_options
                    ),
                }
            )
        raw_linguistic_frame = self.intent_llm_client.extract_layout_slots(
            intent_message,
            active_layout_id=effective_active_layout_id,
            conversation_history=slot_history,
            instance_directory=_current_instance_directory(scene_context),
            **(
                {"engineering_context": scene_context["engineering_context"]}
                if scene_context["engineering_context"]
                else {}
            ),
        )
        source_history = engineering_context_source_history(
            slot_history, scene_context["engineering_context"]
        )
        linguistic_frame = _normalize_linguistic_frame(
            raw_linguistic_frame,
            intent_message,
            conversation_history=source_history,
            site_features=_known_site_features(scene_context),
        )
        scene_context["linguistic_frame"] = linguistic_frame
        yield _event(
            "intent_slots_done",
            session_id,
            turn_id,
            "IntentUnderstandingAgent",
            "done",
            "Linguistic intent slots extracted.",
            payload={
                "linguistic_frame": linguistic_frame,
                "asset_surfaces": _linguistic_asset_surfaces(
                    linguistic_frame,
                ),
            },
        )

        explicit_site_repair = _explicit_central_aisle_width_repair(intent_message)
        linguistic_surface_quantities = _linguistic_surface_quantities(
            linguistic_frame
        )
        grounding = scene_context.get("engineering_rag_grounding")
        instance_directory = _current_instance_directory(scene_context)
        inventory_context = engineering_inventory_declarations(
            scene_context["engineering_context"]
        )
        previous_inventory_context = session.get("retrieved_inventory_context")
        inventory_context_changed = bool(
            previous_inventory_context is not None
            and inventory_context
            and inventory_context != previous_inventory_context
        )
        pending_semantic_intent = (
            pending_intent_state.get("semantic_intent") or {}
            if pending_intent_state
            else {}
        )
        pending_intent_type = str(
            pending_semantic_intent.get("intent_type") or ""
        ).casefold()
        pending_generate_clarification = bool(
            pending_intent_state
            and not session.get("last_layout")
            and not scene_context.get("last_layout")
            and not scene_context.get("seed_layout")
            and pending_intent_type
            in {
                "create_initial_layout",
                "generate_feasible_layout",
                "optimize_layout_from_scratch",
            }
            and "engineering_context" in pending_intent_state
            and pending_intent_state.get("engineering_context")
            == scene_context["engineering_context"]
            and previous_inventory_context is not None
            and inventory_context == previous_inventory_context
        )
        confirmed_asset_choices = reusable_confirmed_asset_choices(
            confirmed_asset_choices,
            linguistic_frame,
            scene_context["engineering_context"],
            inventory_context_changed=inventory_context_changed,
            previous_inventory_context=previous_inventory_context,
        )
        if confirmed_asset_choices:
            scene_context["confirmed_asset_choices"] = copy.deepcopy(
                confirmed_asset_choices
            )
        else:
            scene_context.pop("confirmed_asset_choices", None)
        asset_surfaces = new_equipment_retrieval_surfaces(
            linguistic_frame,
            user_message=intent_message,
            engineering_context=scene_context["engineering_context"],
            has_accepted_layout=bool(
                scene_context.get("last_layout") or scene_context.get("seed_layout")
            ),
            instance_directory=instance_directory,
            cached_grounding=grounding,
            retrieved_surfaces=session.get("retrieved_asset_surfaces") or [],
            confirmed_asset_choices=confirmed_asset_choices,
            inventory_context_changed=inventory_context_changed,
            previous_inventory_context=previous_inventory_context,
        )
        scene_context["fresh_rag_asset_surfaces"] = copy.deepcopy(asset_surfaces)
        needs_initial_retrieval = grounding is None
        should_retrieve = bool(asset_surfaces) or (
            needs_initial_retrieval
            and not (explicit_site_repair and not _linguistic_asset_surfaces(linguistic_frame))
        )
        if should_retrieve and hasattr(self.rag_service, "query"):
            yield _event(
                "rag_started",
                session_id,
                turn_id,
                "EngineeringRAG",
                "running",
                "Retrieving engineering evidence.",
            )
            retrieval_query = _engineering_rag_query_text(
                scene_context["engineering_context"], intent_message
            )
            retrieval_started = time.perf_counter()
            rag_result = self.rag_service.query(
                retrieval_query,
                top_k=8,
                asset_mentions=asset_surfaces or None,
            )
            retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 3)
            retrieved_grounding = _monolithic_rag_snapshot(
                rag_result,
                self.rag_service,
            )
            grounding = _merge_monolithic_rag_snapshots(
                grounding, retrieved_grounding
            )
            session["engineering_rag_grounding"] = copy.deepcopy(grounding)
            session["retrieved_asset_surfaces"] = _unique_surface_texts(
                [*(session.get("retrieved_asset_surfaces") or []), *asset_surfaces]
            )
            if inventory_context:
                session["retrieved_inventory_context"] = inventory_context
            session["engineering_rag_last_retrieved_turn_id"] = turn_id
            retrieval_record = {
                "turn_id": turn_id,
                "asset_surfaces": copy.deepcopy(asset_surfaces),
                "retrieval_ms": retrieval_ms,
                "evidence_ids": _monolithic_evidence_ids(grounding),
            }
            session.setdefault("engineering_rag_retrieval_history", []).append(
                retrieval_record
            )
            yield _event(
                "rag_done",
                session_id,
                turn_id,
                "EngineeringRAG",
                "done",
                "Engineering evidence attached to the intent call.",
                payload={
                    "payload_sha256": grounding.get("payload_sha256"),
                    "evidence_ids": _monolithic_evidence_ids(grounding),
                    "backend": copy.deepcopy(grounding.get("backend") or {}),
                    "clarification_options": copy.deepcopy(
                        _public_clarification_asset_options(
                            _clarification_asset_options(grounding, self.dataset_dir)
                        )
                    ),
                    "retrieval_ms": retrieval_ms,
                    "retrieved_asset_surfaces": copy.deepcopy(asset_surfaces),
                },
            )
        elif grounding:
            yield _event(
                "rag_reused",
                session_id,
                turn_id,
                "EngineeringRAG",
                "done",
                "Reused accepted session evidence for unchanged equipment references.",
                payload={
                    "reuse_reason": "unchanged_session_evidence",
                    "source_turn_id": session.get(
                        "engineering_rag_last_retrieved_turn_id"
                    ),
                    "evidence_ids": _monolithic_evidence_ids(grounding),
                    "reused_asset_surfaces": copy.deepcopy(
                        session.get("retrieved_asset_surfaces") or []
                    ),
                    "retrieval_ms": None,
                },
            )
        elif explicit_site_repair and not _linguistic_asset_surfaces(
            linguistic_frame
        ):
            yield _event(
                "rag_skipped",
                session_id,
                turn_id,
                "EngineeringRAG",
                "done",
                "Reusing the active layout context for the explicit site repair.",
            )
        if grounding:
            grounding = _apply_confirmed_asset_choices(
                grounding,
                confirmed_asset_choices,
                surface_quantities=linguistic_surface_quantities,
            )
            session["engineering_rag_grounding"] = copy.deepcopy(grounding)
            scene_context["engineering_rag_grounding"] = grounding
        if linguistic_frame.get("intent_type") in {"edit","edit_layout","optimize","optimize_existing_layout","repair","constraint_repair"} and _current_instance_directory(scene_context):
            protected_surfaces = _unique_surface_texts(
                [
                    *asset_surfaces,
                    *new_model_operation_surfaces(linguistic_frame, intent_message),
                    *[
                        choice.get("surface_text")
                        for choice in confirmed_asset_choices
                        if isinstance(choice, dict)
                    ],
                ]
            )
            grounding = _bind_existing_role_grounding(
                grounding,
                _current_instance_directory(scene_context),
                protected_surfaces=protected_surfaces,
            )
            scene_context["engineering_rag_grounding"] = grounding
        if pending_intent_state and not session.get("last_layout") and not re.search(r"\b(?:rename|relabel)\b", str(user_message or ""), re.I):
            from tool.context.conversation_deltas import retain_confirmed_role_declarations
            linguistic_frame, grounding, retained_roles = retain_confirmed_role_declarations(
                linguistic_frame, grounding,
                pending_intent_state.get("asset_role_declarations") or
                (pending_intent_state.get("semantic_intent") or {}).get("asset_mentions"),
                confirmed_asset_choices=confirmed_asset_choices,
                allow_partial_inventory_merge=pending_generate_clarification,
            )
            if retained_roles:
                scene_context["linguistic_frame"] = linguistic_frame
                scene_context["engineering_rag_grounding"] = grounding
                yield _event("role_aliases_retained", session_id, turn_id, "IntentUnderstandingAgent", "done",
                    "Retained the original equipment roles while applying the confirmed catalog choices.",
                    payload={"asset_mentions":copy.deepcopy(linguistic_frame["asset_mentions"])})
        scene_context["clarification_asset_options"] = _clarification_asset_options(
            grounding,
            self.dataset_dir,
        )
        scene_context["public_clarification_asset_options"] = (
            _public_clarification_asset_options(
                scene_context["clarification_asset_options"]
            )
        )
        if linguistic_frame.get("intent_type") in {"edit","edit_layout"}:
            scene_context["planned_edit_instance_directory"] = _planned_edit_instance_directory(
                linguistic_frame, grounding or {}, _current_instance_directory(scene_context))
        yield _event("intent_started", session_id, turn_id, "IntentUnderstandingAgent", "running", "Parsing user intent.")
        intent_step = IntentUnderstandingAgent(self.intent_llm_client).run(
            intent_message,
            active_layout_id=effective_active_layout_id,
            scene_context=scene_context,
        )
        yield _event(
            "intent_done",
            session_id,
            turn_id,
            "IntentUnderstandingAgent",
            "done",
            intent_step["semantic_intent"].get("reasoning_summary", "Intent parsed."),
            payload={
                "intent": intent_step["intent"],
                "semantic_intent": intent_step["semantic_intent"],
                "completeness": intent_step["completeness"],
                "unresolved_requirements": intent_step["unresolved_requirements"],
                "can_enter_designer": intent_step["can_enter_designer"],
                "semantic_evidence": intent_step["semantic_evidence"],
                "intent_contract": intent_step["intent_contract"],
            },
        )

        intent_validation_loop = _execute_intent_validation_loop(
            self.dataset_dir,
            self.rag_service,
            self.intent_llm_client,
            intent_message,
            intent_step,
            scene_context,
            max_iterations=MAX_AGENTIC_ITERATIONS,
            enable_feedback=True,
        )
        for transition in intent_validation_loop["transitions"]:
            if transition["kind"] == "intent_validation":
                validation = transition["step"]
                yield _event(
                    "intent_validation_done",
                    session_id,
                    turn_id,
                    "ValidatorAgent",
                    "done",
                    validation["validation_report"]["status"],
                    payload={"intent_validation": validation},
                    iteration=transition["revision"],
                )
            elif transition["kind"] == "intent_feedback":
                yield _event(
                    "intent_feedback_generated",
                    session_id,
                    turn_id,
                    "ValidatorAgent",
                    "done",
                    "Structured semantic feedback sent to IntentUnderstandingAgent.",
                    payload={
                        "feedback": transition["feedback"],
                        "target_agent": "IntentUnderstandingAgent",
                    },
                    iteration=transition["revision"],
                )
            elif transition["kind"] == "intent_revision":
                yield _event(
                    "intent_revision_done",
                    session_id,
                    turn_id,
                    "IntentUnderstandingAgent",
                    "done",
                    "Semantic intent revised from Validator feedback.",
                    payload={"intent_step": transition["step"]},
                    iteration=transition["revision"],
                )
        intent_step = intent_validation_loop["intent_step"]

        loop_result = None
        loop_transitions = []
        for transition in _iterate_design_loop(
            self.dataset_dir,
            self.rag_service,
            intent_message,
            intent_step,
            scene_context,
        ):
            kind = transition["kind"]
            revision = transition.get("revision")
            if kind == "complete":
                loop_result = transition["result"]
                continue
            loop_transitions.append(transition)
            if kind == "design_started":
                yield _event(
                    "design_started",
                    session_id,
                    turn_id,
                    "DesignerAgent",
                    "running",
                    f"Building design revision {revision}.",
                    iteration=revision,
                )
            elif kind == "design":
                step = transition["step"]
                yield _event(
                    "design_done",
                    session_id,
                    turn_id,
                    "DesignerAgent",
                    "skipped" if transition.get("skipped") else "done",
                    _design_event_summary(step),
                    payload={"design": step},
                    iteration=revision,
                )
            elif kind == "validation_started":
                yield _event(
                    "validation_started",
                    session_id,
                    turn_id,
                    "ValidatorAgent",
                    "running",
                    f"Validating revision {revision}.",
                    iteration=revision,
                )
            elif kind == "validation":
                step = transition["step"]
                yield _event(
                    "validation_done",
                    session_id,
                    turn_id,
                    "ValidatorAgent",
                    "skipped" if transition.get("skipped") else "done",
                    step["validation_report"]["status"],
                    payload={
                        "validation": step,
                        "clarification_options": copy.deepcopy(
                            scene_context.get(
                                "public_clarification_asset_options"
                            )
                            or []
                        ),
                    },
                    iteration=revision,
                )
            elif kind == "evaluation_started":
                yield _event(
                    "evaluation_started",
                    session_id,
                    turn_id,
                    "EvaluatorAgent",
                    "running",
                    f"Evaluating revision {revision}.",
                    iteration=revision,
                )
            elif kind == "evaluation":
                step = transition["step"]
                yield _event(
                    "evaluation_done",
                    session_id,
                    turn_id,
                    "EvaluatorAgent",
                    "skipped" if transition.get("skipped") else "done",
                    step["evaluation_status"],
                    payload={"evaluation": step},
                    iteration=revision,
                )
            elif kind == "feedback":
                source_agent = "ValidatorAgent" if transition["source"] == "validator" else "EvaluatorAgent"
                feedback = transition["feedback"]
                yield _event(
                    "feedback_generated",
                    session_id,
                    turn_id,
                    source_agent,
                    "done",
                    f"{len(_feedback_actions(feedback))} structured action(s) sent to DesignerAgent.",
                    payload={"feedback": feedback, "target_agent": "DesignerAgent"},
                    iteration=revision,
                )

        if loop_result is None:  # pragma: no cover - protects the SSE protocol.
            raise RuntimeError("Agentic design loop produced no completion result.")
        loop_result["transitions"] = loop_transitions
        designer_step = loop_result["design"]
        validator_step = loop_result["validation"]
        evaluator_step = loop_result["evaluation"]
        status = loop_result["status"]

        domain_candidates = []
        evidence = designer_step.get("evidence_package") or {}
        for match in evidence.get("matched_assets") or []:
            if match.get("match_type") == "domain_intent" or "domain" in str(match.get("surface_text", "")).lower():
                domain_candidates.extend(match.get("candidate_model_asset_ids") or [])
        session["last_domain_asset_candidates"] = domain_candidates
        if status in VALID_LAYOUT_STATUSES and designer_step.get("rough_layout"):
            session["last_layout"] = designer_step["rough_layout"]
            session["last_flp_ir"] = designer_step.get("flp_ir")
            session["last_solver_report"] = designer_step.get("solver_report")
            session["active_layout_id"] = session["last_layout"].get("layout_id")
            active_instance_ids = {
                item.get("instance_id")
                for item in session["last_layout"].get("placements") or []
            }
            session["selected_instance_ids"] = [
                instance_id
                for instance_id in session["selected_instance_ids"]
                if instance_id in active_instance_ids
            ]
        if status == "NEED_CLARIFICATION":
            session["pending_request_context"] = intent_message
            session["pending_intent_state"] = {
                "asset_role_declarations": copy.deepcopy((intent_step.get("semantic_intent") or {}).get("asset_mentions") or []),
                "engineering_context": scene_context.get("engineering_context") or "",
                "intent_revision_id": intent_step.get("intent_revision_id"),
                "semantic_intent": copy.deepcopy(
                    intent_step.get("semantic_intent") or {}
                ),
                "unresolved_requirements": copy.deepcopy(
                    intent_step.get("unresolved_requirements") or []
                ),
                "clarification_questions": copy.deepcopy(
                    (intent_step.get("semantic_intent") or {}).get(
                        "clarification_questions"
                    )
                    or []
                ),
                "clarification_asset_options": copy.deepcopy(
                    scene_context.get("clarification_asset_options") or []
                ),
                "confirmed_asset_choices": copy.deepcopy(
                    confirmed_asset_choices
                ),
            }
        else:
            session["pending_request_context"] = None
            session["pending_intent_state"] = None
        session["turn_history"].append(
            {
                "turn_id": turn_id,
                "user_message": str(user_message or "").strip(),
                "intent_type": intent_step["intent"].get("intent_type"),
                "status": status,
                "active_layout_id": session.get("active_layout_id"),
                "selected_instance_ids": list(
                    session.get("selected_instance_ids") or []
                ),
                "unresolved_requirements": copy.deepcopy(
                    intent_step.get("unresolved_requirements") or []
                ),
                "clarification_questions": copy.deepcopy(
                    (intent_step.get("semantic_intent") or {}).get(
                        "clarification_questions"
                    )
                    or []
                ),
            }
        )
        session["turn_history"] = session["turn_history"][
            -MAX_SESSION_HISTORY_TURNS:
        ]

        yield _event(
            "loop_done",
            session_id,
            turn_id,
            "AgenticLayOpt",
            "done" if status in VALID_LAYOUT_STATUSES else "blocked",
            loop_result["loop_report"]["termination_reason"],
            payload={
                "status": status,
                "loop_report": loop_result["loop_report"],
                "revision_history": loop_result["revision_history"],
            },
            iteration=loop_result["loop_report"]["iterations"],
        )

        if status in VALID_LAYOUT_STATUSES and designer_step.get("rough_layout"):
            layout_snapshot = copy.deepcopy(designer_step["rough_layout"])
            layout_content_sha256 = layout_sha256(layout_snapshot)
            render_token = _viewport_render_token(
                session_id,
                turn_id,
                layout_content_sha256,
            )
            pending_receipts = session.setdefault("pending_viewport_receipts", {})
            pending_receipts[render_token] = {
                "turn_id": turn_id,
                "layout_sha256": layout_content_sha256,
                "layout_snapshot": layout_snapshot,
            }
            while len(pending_receipts) > MAX_SESSION_HISTORY_TURNS:
                pending_receipts.pop(next(iter(pending_receipts)))
            yield _event(
                "viewport_command",
                session_id,
                turn_id,
                "WebViewport",
                "done",
                "Render rough layout.",
                viewport_command={
                    "method": "RenderLayoutJson",
                    "payload": designer_step["rough_layout"],
                    "receipt_required": True,
                    "render_token": render_token,
                    "turn_id": turn_id,
                    "layout_sha256": layout_content_sha256,
                },
                iteration=loop_result["loop_report"]["selected_revision"],
            )

        final_result = {
            "status": status,
            "solver_used": designer_step.get("solver_used"),
            "intent": intent_step["intent"],
            "semantic_intent": intent_step["semantic_intent"],
            "intent_validation": intent_validation_loop["validation"],
            "intent_revision_history": intent_validation_loop["history"],
            "design": {
                "design_status": designer_step["design_status"],
                "layout_edit_ir": designer_step.get("layout_edit_ir"),
                "layout_diff": designer_step.get("layout_diff"),
                "flp_ir": designer_step["flp_ir"],
                "rough_layout": designer_step["rough_layout"],
                "solver_report": designer_step.get("solver_report"),
                "solver_used": designer_step.get("solver_used"),
                "design_basis": designer_step.get("design_basis"),
                "solver_evidence": designer_step.get("solver_evidence"),
                "applied_feedback": designer_step.get("applied_feedback"),
                "task_contract": copy.deepcopy(
                    designer_step.get("task_contract") or {}
                ),
                "revision": designer_step.get("revision"),
            },
            "validation": {
                "validation_report": validator_step["validation_report"],
                "layout_validation": copy.deepcopy(
                    validator_step.get("layout_validation") or {}
                ),
                "repair": validator_step["repair"],
            },
            "evaluation": {
                "evaluation_status": evaluator_step["evaluation_status"],
                "metrics": evaluator_step["metrics"],
                "notes": evaluator_step["notes"],
                "comparison": evaluator_step["comparison"],
                "optimization_feedback": evaluator_step["optimization_feedback"],
                "recovery_notice": copy.deepcopy(
                    evaluator_step.get("recovery_notice") or {}
                ),
                "revision": evaluator_step["revision"],
            },
            "loop_report": loop_result["loop_report"],
            "revision_history": loop_result["revision_history"],
        }
        _attach_assistant_message(final_result, self.intent_llm_client, user_message)
        yield _event(
            "turn_done" if status in VALID_LAYOUT_STATUSES else "turn_blocked",
            session_id,
            turn_id,
            "AgenticLayOpt",
            "done" if status in VALID_LAYOUT_STATUSES else "blocked",
            status,
            payload={
                "status": status,
                "active_layout_id": session.get("active_layout_id"),
                "selected_instance_ids": session.get("selected_instance_ids"),
                "solver_used": designer_step.get("solver_used"),
                "final_violation_count": _final_violation_count(validator_step, evaluator_step),
                "loop_report": loop_result["loop_report"],
                "revision_history": loop_result["revision_history"],
                "assistant_message": final_result.get("assistant_message"),
                "assistant_message_source": final_result.get("assistant_message_source"),
                "assistant_message_error": final_result.get("assistant_message_error"),
                "clarification_questions": copy.deepcopy(
                    (validator_step.get("repair") or {}).get(
                        "clarification_questions"
                    )
                    or []
                ),
                "clarification_options": copy.deepcopy(
                    scene_context.get("public_clarification_asset_options") or []
                ),
                "turn_context": turn_context,
            },
            iteration=loop_result["loop_report"]["iterations"],
        )


    def record_viewport_receipt(self, session_id, receipt):
        session = self.sessions.get(session_id)
        if session is None:
            raise ViewportReceiptError("Unknown Agentic LayOpt session.")
        received_token = receipt.get("render_token") if isinstance(receipt, dict) else None
        pending_receipts = session.get("pending_viewport_receipts") or {}
        pending = pending_receipts.get(received_token)
        if not received_token or pending is None:
            raise ViewportReceiptError("Viewport receipt render_token is missing or stale.")
        verified = verify_viewport_receipt(
            pending["layout_snapshot"],
            receipt,
            expected_turn_id=pending["turn_id"],
            expected_layout_sha256=pending["layout_sha256"],
        )
        if verified["status"] == "MISMATCH":
            raise ViewportReceiptError(
                "Viewport receipt did not verify: " + verified["status"]
            )
        pending_receipts.pop(received_token, None)
        verified["render_token"] = received_token
        verified["session_id"] = str(session_id)
        verified["turn_id"] = pending["turn_id"]
        session["latest_viewport_receipt"] = copy.deepcopy(verified)
        if not session.get("initial_layout"):
            session["initial_layout"] = copy.deepcopy(pending["layout_snapshot"])
        return verified


def _compact_conversation_history(history):
    compact = []
    for item in list(history or [])[-MAX_SESSION_HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "turn_id": int(item.get("turn_id") or 0),
                "user_message": str(item.get("user_message") or "").strip(),
                "intent_type": str(item.get("intent_type") or "unknown"),
                "status": str(item.get("status") or "unknown"),
                "active_layout_id": item.get("active_layout_id"),
                "selected_instance_ids": [
                    str(value)
                    for value in item.get("selected_instance_ids") or []
                    if value is not None
                ],
                "unresolved_requirements": copy.deepcopy(
                    item.get("unresolved_requirements") or []
                ),
                "clarification_questions": [
                    str(value)
                    for value in item.get("clarification_questions") or []
                    if value is not None
                ],
            }
        )
    return compact


def _conversation_intent_message(user_message, *, pending_context=None, turn_history=None):
    current = str(user_message or "").strip()
    if pending_context:
        return (
            "Previous request requiring clarification:\n"
            f"{str(pending_context).strip()}\n\n"
            f"User clarification:\n{current}"
        )
    # Completed turns stay available as structured history for reference
    # resolution. Only the current turn defines the operation being requested.
    return current


def _turn_context(history, *, pending_context=None):
    return {
        "prior_completed_turn_count": len(history),
        "prior_turn_ids": [item["turn_id"] for item in history],
        "clarification_context_applied": bool(pending_context),
    }


def _viewport_render_token(session_id, turn_id, layout_content_sha256):
    payload = {
        "session_id": str(session_id),
        "turn_id": int(turn_id),
        "layout_sha256": str(layout_content_sha256),
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def build_paper_agent_cards():
    cards = [
        {
            "agent_name": "IntentUnderstandingAgent",
            "responsibility": "extract linguistic slots, bind assets through Engineering RAG, and execute Validator-guided semantic revisions",
            "does_not": ["generate geometry", "run a solver", "declare layout legality"],
        },
        {
            "agent_name": "DesignerAgent",
            "responsibility": "compile and normalize FLP-IR, solve layouts, and execute structured feedback",
            "does_not": ["self-validate legality", "silently change user hard constraints"],
        },
        {
            "agent_name": "ValidatorAgent",
            "responsibility": "validate semantic intent and layout state as separate scopes, then route feedback to the responsible agent",
            "does_not": ["optimize soft objectives", "modify the candidate directly"],
        },
        {
            "agent_name": "EvaluatorAgent",
            "responsibility": "evaluate only legal layouts, compare revisions, and emit optimization feedback",
            "does_not": ["repair hard constraints", "modify FLP-IR or layout"],
        },
    ]
    for card in cards:
        schema = contract_schema(card["agent_name"])
        card["contract_version"] = schema["contract_version"]
        card["input_fields"] = schema["input_fields"]
        card["output_fields"] = schema["output_fields"]
        card["input"] = ", ".join(schema["input_fields"])
        card["output"] = ", ".join(schema["output_fields"])
    return cards


def _monolithic_rag_snapshot(rag_result, rag_service):
    if not isinstance(rag_result, dict):
        raise RAGBlockedError(
            "invalid_monolithic_rag_response",
            "Engineering RAG must return a JSON object for monolithic planning.",
        )
    matches = []
    for match in rag_result.get("asset_matches") or []:
        if not isinstance(match, dict):
            continue
        candidates = []
        for candidate in (match.get("candidate_model_asset_ids") or [])[:5]:
            if isinstance(candidate, dict):
                candidates.append(
                    {
                        key: copy.deepcopy(candidate.get(key))
                        for key in ("model_asset_id", "score", "match_type")
                        if candidate.get(key) is not None
                    }
                )
            elif candidate:
                candidates.append({"model_asset_id": str(candidate)})
        sources = []
        for source in (match.get("source_provenance") or [])[:5]:
            if not isinstance(source, dict):
                continue
            sources.append(
                {
                    "case_id": source.get("case_id"),
                    "experiment_run_id": source.get("experiment_run_id"),
                    "mention_id": source.get("mention_id"),
                    "evidence_ids": [
                        evidence.get("evidence_id")
                        for evidence in (source.get("evidence") or [])[:8]
                        if isinstance(evidence, dict) and evidence.get("evidence_id")
                    ],
                }
            )
        matches.append(
            {
                "mention_id": match.get("mention_id"),
                "surface_text": match.get("surface_text"),
                "start_char": match.get("start_char"),
                "end_char": match.get("end_char"),
                "decision": match.get("decision")
                or match.get("ambiguity_status"),
                "selected_model_asset_id": match.get("selected_model_asset_id"),
                "candidate_model_asset_ids": candidates,
                "source_case_id": match.get("source_case_id"),
                "source_run_id": match.get("source_run_id"),
                "sources": sources,
            }
        )
    result_provenance = copy.deepcopy(rag_result.get("provenance") or {})
    query_trace = rag_result.get("engineering_rag_trace") or {}
    service_backend = getattr(rag_service, "backend_identity", None)
    service_backend = service_backend() if callable(service_backend) else service_backend
    backend_identity = query_trace.get("backend_identity") or service_backend or {}
    backend = {
        "service_class": type(rag_service).__name__,
        "backend_id": backend_identity.get("backend_id"),
        "mode": backend_identity.get("mode") or result_provenance.get("mode"),
        "service": backend_identity.get("service"),
        "source_path": result_provenance.get("source_path"),
        "source_sha256": result_provenance.get("source_sha256"),
    }
    compact_results = []
    for result in (rag_result.get("results") or [])[:24]:
        if not isinstance(result, dict):
            continue
        compact_results.append(
            {
                key: copy.deepcopy(result.get(key))
                for key in ("evidence_id", "source_id", "field_path", "confidence")
                if result.get(key) is not None
            }
        )
    snapshot = {
        "backend": backend,
        "status": rag_result.get("status"),
        "asset_matches": matches,
        "results": compact_results,
    }
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    if len(encoded) > 65536:
        raise RAGBlockedError(
            "monolithic_rag_evidence_too_large",
            "Compact Engineering RAG evidence exceeds the 64 KiB planning limit.",
            details={"payload_bytes": len(encoded), "limit_bytes": 65536},
        )
    snapshot["payload_bytes"] = len(encoded)
    snapshot["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return snapshot


def _merge_monolithic_rag_snapshots(existing, retrieved):
    """Retain retrieved evidence while adding the actual evidence from a new query."""

    if not existing:
        return copy.deepcopy(retrieved or {})
    if not retrieved:
        return copy.deepcopy(existing)
    matches = []
    match_positions = {}
    for match in [
        *(existing.get("asset_matches") or []),
        *(retrieved.get("asset_matches") or []),
    ]:
        if not isinstance(match, dict):
            continue
        key = _folded_text(match.get("surface_text"))
        if not key:
            continue
        if key in match_positions:
            matches[match_positions[key]] = copy.deepcopy(match)
        else:
            match_positions[key] = len(matches)
            matches.append(copy.deepcopy(match))
    results = []
    seen_results = set()
    for result in [
        *(existing.get("results") or []),
        *(retrieved.get("results") or []),
    ]:
        if not isinstance(result, dict):
            continue
        evidence_id = str(result.get("evidence_id") or "")
        key = evidence_id or json.dumps(
            result, ensure_ascii=False, sort_keys=True, default=str
        )
        if key not in seen_results:
            seen_results.add(key)
            results.append(copy.deepcopy(result))
    merged = {
        "backend": copy.deepcopy(retrieved.get("backend") or existing.get("backend") or {}),
        "status": retrieved.get("status") or existing.get("status"),
        "asset_matches": matches,
        "results": results,
    }
    encoded = json.dumps(
        merged, ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    if len(encoded) > 65536:
        raise RAGBlockedError(
            "monolithic_rag_evidence_too_large",
            "Compact Engineering RAG evidence exceeds the 64 KiB planning limit.",
            details={"payload_bytes": len(encoded), "limit_bytes": 65536},
        )
    merged["payload_bytes"] = len(encoded)
    merged["payload_sha256"] = hashlib.sha256(encoded).hexdigest()
    return merged


def _engineering_rag_query_text(engineering_context, current_stage_request):
    context = str(engineering_context or "").strip()
    request = str(current_stage_request or "").strip()
    if not context:
        return request
    return (
        "Stable engineering context for retrieval only; it is not the current operation:\n"
        f"{context}\n\nCurrent stage request:\n{request}"
    )


def _unique_surface_texts(values):
    result = []
    seen = set()
    for value in values or []:
        surface = str(value or "").strip()
        key = _folded_text(surface)
        if surface and key not in seen:
            seen.add(key)
            result.append(surface)
    return result


def _compiled_site_snapshot(flp_ir, scene_context):
    """Use the site snapshot carried by compiled FLP-IR after compilation.

    The frontend scene snapshot can predate natural-site and organization
    compilation. Once FLP-IR exists, its site snapshot contains the current
    regions, bays, and functional subregions used by design and validation.
    """

    compiled_snapshot = ((flp_ir or {}).get("site_ref") or {}).get(
        "site_snapshot"
    )
    return copy.deepcopy(
        compiled_snapshot or (scene_context or {}).get("site_snapshot") or {}
    )


def _asset_option_catalog(dataset_dir):
    """Read catalog names only for user-facing clarification choices."""

    try:
        raw = json.loads(
            (Path(dataset_dir) / "model_assets.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError, TypeError):
        return {}
    entries = (
        raw
        if isinstance(raw, list)
        else raw.get("assets", raw.get("items", []))
        if isinstance(raw, dict)
        else []
    )
    return {
        str(item.get("model_asset_id")): item
        for item in entries
        if isinstance(item, dict) and item.get("model_asset_id")
    }


def _human_asset_option_label(asset, ordinal=None):
    """Describe a catalog choice without exposing its internal identifier.

    The wording is deliberately limited to catalog fields.  A structured
    capability is presented as a catalog-recorded function; an equipment type
    is presented as a catalog type.  It does not infer a function from mesh
    geometry or an asset ID.
    """

    planning = asset.get("planning_fields") or {}
    capabilities = _unique_readable_catalog_terms(
        str(item).replace("_", " ").replace("-", " ")
        for item in planning.get("process_capabilities") or []
        if str(item).strip()
    )
    equipment_type = _human_catalog_type(asset.get("equipment_type"))
    catalog_name = _readable_catalog_name(asset)
    kind = catalog_name or equipment_type or "Unlabeled catalog record"
    details = []
    if equipment_type and _folded_text(equipment_type) != _folded_text(kind):
        details.append(f"catalog type: {equipment_type}")
    if capabilities:
        details.append(
            "catalog-recorded function"
            + ("s" if len(capabilities) > 1 else "")
            + ": "
            + ", ".join(capabilities)
        )
    asset_scope = _human_catalog_scope(asset.get("asset_scope"))
    category = _human_catalog_scope(asset.get("category"))
    if asset_scope and asset_scope != "equipment":
        details.append(f"catalog scope: {asset_scope}")
    if not catalog_name and category:
        details.append(f"catalog category: {category}")
    footprint = (asset.get("geometry") or {}).get("footprint") or (
        asset.get("geometry") or {}
    ).get("dimensions_m") or {}
    width = _optional_float(footprint.get("width_m"))
    depth = _optional_float(footprint.get("depth_m"))
    if width and depth:
        width_text = f"{width:.2f}".rstrip("0").rstrip(".")
        depth_text = f"{depth:.2f}".rstrip("0").rstrip(".")
        details.append(
            f"catalog footprint: approximately {width_text} m by {depth_text} m"
        )
    return kind if not details else f"{kind} ({'; '.join(details)})"


def _human_catalog_type(value):
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    folded = _folded_text(text)
    if folded in {"", "fbx scene", "unknown", "equipment"}:
        return None
    return text


def _human_catalog_scope(value):
    """Render a catalog classification as words without asserting a function."""

    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def _readable_catalog_name(asset):
    values = [
        asset.get("catalog_name_en"),
        asset.get("english_name"),
        asset.get("display_name_en"),
        asset.get("canonical_name_en"),
        asset.get("asset_name"),
        *(asset.get("aliases") or []),
    ]
    for value in values:
        label = str(value or "").strip()
        if not label or _looks_like_internal_asset_identifier(label):
            continue
        public_label = _english_catalog_label(label)
        if public_label:
            return public_label
    return None


def _english_catalog_label(value):
    """Return a safe English catalog label while retaining the source record.

    Source-library names often store an English prefix followed by a native or
    damaged-encoding parenthetical.  The original catalog record remains
    untouched; this projection simply avoids showing unreadable text in an
    English clarification choice.
    """

    label = " ".join(str(value or "").split())
    if not label or _looks_like_internal_asset_identifier(label):
        return None
    match = re.match(r"^(?P<prefix>[^()]+?)\s*\((?P<suffix>.*)\)$", label)
    if match:
        prefix = " ".join(match.group("prefix").split())
        suffix = match.group("suffix")
        if re.search(r"[A-Za-z]", prefix) and _contains_non_english_source_text(suffix):
            label = prefix
    if not re.search(r"[A-Za-z]", label) or _contains_replacement_character(label):
        return None
    return label


def _contains_non_english_source_text(value):
    text = str(value or "")
    return _contains_replacement_character(text) or any(ord(character) > 127 for character in text)


def _contains_replacement_character(value):
    return "\ufffd" in str(value or "")


def _looks_like_internal_asset_identifier(value):
    text = str(value or "").strip().casefold()
    return text.startswith("asset/") or text.startswith("model_asset_id=")


def _unique_readable_catalog_terms(values):
    result = []
    for value in values or []:
        text = " ".join(str(value or "").split())
        folded = _folded_text(text)
        if text and folded and folded not in {_folded_text(item) for item in result}:
            result.append(text)
    return result


def _clarification_asset_options(rag_grounding, dataset_dir):
    """Project unresolved RAG matches into descriptions a person can choose."""

    catalog = _asset_option_catalog(dataset_dir)
    options = []
    for match in (rag_grounding or {}).get("asset_matches") or []:
        if not isinstance(match, dict):
            continue
        surface_text = str(match.get("surface_text") or "").strip()
        selected = str(match.get("selected_model_asset_id") or "").strip()
        dimensions = ((catalog.get(selected) or {}).get("geometry") or {}).get("dimensions_m") or {}
        missing_geometry = bool(dimensions) and any(dimensions.get(key) is None for key in ("width_m", "depth_m", "height_m"))
        if not surface_text or (selected and not missing_geometry):
            continue
        candidate_ids = []
        for candidate in match.get("candidate_model_asset_ids") or []:
            candidate_id = (
                candidate.get("model_asset_id")
                if isinstance(candidate, dict)
                else candidate
            )
            candidate_id = str(candidate_id or "").strip()
            if candidate_id and candidate_id not in candidate_ids:
                candidate_ids.append(candidate_id)
        choices = []
        for ordinal, candidate_id in enumerate(candidate_ids, start=1):
            asset = catalog.get(candidate_id) or {}
            label = _human_asset_option_label(asset, ordinal)
            choices.append(
                {
                    "display_label": label,
                    "model_asset_id": candidate_id,
                    "reply_phrases": _clarification_reply_phrases(asset, label),
                }
            )
        if choices:
            options.append(
                {
                    "surface_text": surface_text,
                    "choices": choices,
                }
            )
    return options


def _public_clarification_asset_options(options):
    """Return only the wording a frontend may render to a person."""

    projected = []
    for option in options or []:
        surface_text = str(option.get("surface_text") or "").strip()
        choices = [
            {"display_label": str(choice.get("display_label") or "").strip()}
            for choice in (option.get("choices") or [])[:3]
            if str(choice.get("display_label") or "").strip()
        ]
        if surface_text and choices:
            projected.append(
                {
                    "surface_text": surface_text,
                    "choices": choices,
                }
            )
    return projected


def _public_clarification_asset_questions(options):
    questions = []
    for option in options or []:
        surface_text = str(option.get("surface_text") or "").strip()
        labels = [
            str(choice.get("display_label") or "").strip()
            for choice in option.get("choices") or []
            if str(choice.get("display_label") or "").strip()
        ]
        if surface_text and labels:
            questions.append(
                f'For "{surface_text}", choose one: ' + "; ".join(labels) + "."
            )
    return questions


def _clarification_reply_phrases(asset, display_label):
    """Keep ordinary-language reply phrases private to the choice record."""

    base_label = str(display_label or "").split(" (", 1)[0].strip()
    candidates = [
        display_label,
        base_label,
        _readable_catalog_name(asset),
        _human_catalog_type(asset.get("equipment_type")),
    ]
    generic = {
        "",
        "catalog equipment",
        "unlabeled catalog record",
        "composite scene",
        "equipment",
    }
    phrases = []
    for candidate in candidates:
        text = " ".join(str(candidate or "").split())
        folded = _folded_text(text)
        if not folded or folded in generic or folded in {_folded_text(item) for item in phrases}:
            continue
        phrases.append(text)
    return phrases


def _confirmed_clarification_asset_choices(user_message, options):
    confirmed = []
    for option in options or []:
        # A combined reply can select several different catalog entries. Match
        # each role within its own sentence before comparing candidate names.
        # Otherwise a longer name selected for another role wins globally.
        target = re.split(r"\bfor\b", str(option.get("surface_text") or ""), maxsplit=1, flags=re.I)[0]
        stop = {"a","an","one","two","three","four","five","six","seven","eight","nine","ten","and","both","labeled","labelled"}
        tokens = {t for t in _process_surface_tokens(target) if len(t)>1 and t not in stop}
        clauses = re.split(r"[;\n]+|(?<=[.!?])\s+", str(user_message or ""))
        relevant = [c for c in clauses if tokens and tokens <= _process_surface_tokens(c)]
        reply = _folded_text(" ".join(relevant) if relevant else user_message)
        matches = []
        for choice in option.get("choices") or []:
            bare_label = _folded_text(choice.get("display_label"))
            base_label = str(choice.get("display_label") or "").split(" (", 1)[0].strip()
            base_label_folded = _folded_text(base_label)
            if bare_label in {"table","machine","equipment","unit","cabinet","rack","conveyor"} and not re.search(r"\b(?:use|choose|select)\s+(?:the\s+)?"+re.escape(bare_label)+r"(?=\s*(?:[.,;]|for\b|$))", reply):
                continue
            phrases = _unique_readable_catalog_terms(
                [base_label, choice.get("display_label"), *(choice.get("reply_phrases") or [])]
            )
            matched_lengths = [
                len(_folded_text(phrase))
                for phrase in phrases
                if _folded_text(phrase) and _folded_text(phrase) in reply
            ]
            if matched_lengths:
                identity = base_label_folded
                score = (max(matched_lengths), int(bool(identity and identity in reply)))
                matches.append((choice, score))
        longest = max((score for _, score in matches), default=(0,0))
        selected = [choice for choice, score in matches if score == longest]
        if len(selected) == 1:
            confirmed.append(
                {
                    "surface_text": option.get("surface_text"),
                    "model_asset_id": selected[0].get("model_asset_id"),
                    "display_label": selected[0].get("display_label"),
                }
            )
    return confirmed


def _bind_existing_role_grounding(grounding,directory,*,protected_surfaces=None):
    """Resolve a literal existing role only within its actually retrieved pool."""
    result=copy.deepcopy(grounding or {})
    def key(value):return re.sub(r"^(?:(?:a|an|the|existing)\s+)+","",_folded_text(value))
    protected={key(value) for value in protected_surfaces or [] if key(value)}
    def storage_role(value):
        words = _process_surface_tokens(key(value))
        if not words.intersection({"rack", "storage"}):
            return None
        qualifier = words - {"rack", "storage", "unit"}
        return qualifier or None
    for match in result.get("asset_matches") or []:
        selected=[item for item in directory if any(key(alias)==key(match.get("surface_text")) for alias in (item["stable_alias"].values() if isinstance(item["stable_alias"],dict) else [item["stable_alias"]]))]
        if not selected and storage_role(match.get("surface_text")):
            selected = [item for item in directory if any(storage_role(alias)==storage_role(match["surface_text"])
                for alias in (item["stable_alias"].values() if isinstance(item["stable_alias"],dict) else [item["stable_alias"]]))]
        pool={item.get("model_asset_id") if isinstance(item,dict) else item for item in match.get("candidate_model_asset_ids") or []}
        if len(selected)==1 and selected[0]["model_asset_id"] in pool and key(match.get("surface_text")) not in protected:
            match.update(selected_model_asset_id=selected[0]["model_asset_id"],quantity_index=selected[0]["quantity_index"],decision="resolved",ambiguity_status="resolved",decision_source="authoritative_existing_instance")
    present={key(match.get("surface_text")) for match in result.get("asset_matches") or [] if match.get("surface_text")}
    for instance in directory or []:
        if not isinstance(instance,dict):continue
        model_id=str(instance.get("model_asset_id") or "").strip()
        if not model_id:continue
        witness=None
        for match in result.get("asset_matches") or []:
            selected=str(match.get("selected_model_asset_id") or "").strip()
            pool={str(item.get("model_asset_id") if isinstance(item,dict) else item).strip()
                  for item in match.get("candidate_model_asset_ids") or []}
            if selected==model_id or model_id in pool:
                witness=match
                break
        if witness is None:continue
        aliases=[]
        value=instance.get("stable_alias")
        if isinstance(value,dict):aliases=[str(alias).strip() for alias in value.values() if str(alias or "").strip()]
        elif isinstance(value,(list,tuple,set)):aliases=[str(alias).strip() for alias in value if str(alias or "").strip()]
        elif str(value or "").strip():aliases=[str(value).strip()]
        for alias in aliases:
            alias_key=key(alias)
            if not alias_key or alias_key in present:continue
            bound=copy.deepcopy(witness)
            bound.update(surface_text=alias,selected_model_asset_id=model_id,
                decision="resolved",ambiguity_status="resolved",
                decision_source="authoritative_existing_instance",
                instance_id=instance.get("instance_id"),
                quantity_index=instance.get("quantity_index"),
                source_surface_text=witness.get("surface_text"))
            result.setdefault("asset_matches",[]).append(bound)
            present.add(alias_key)
    return result


def _merge_confirmed_asset_choices(*choice_groups):
    by_surface = {}
    for choices in choice_groups:
        for choice in choices or []:
            surface = _folded_text(choice.get("surface_text"))
            model_asset_id = str(choice.get("model_asset_id") or "").strip()
            if surface and model_asset_id:
                by_surface[surface] = copy.deepcopy(choice)
    return list(by_surface.values())


def _linguistic_surface_quantities(frame):
    return {
        _folded_text(item.get("surface_text")): max(
            1,
            int(_optional_float(item.get("quantity")) or 1),
        )
        for item in (frame or {}).get("asset_mentions") or []
        if isinstance(item, dict) and str(item.get("surface_text") or "").strip()
    }


def _apply_confirmed_asset_choices(
    rag_grounding,
    confirmed_choices,
    *,
    surface_quantities=None,
):
    grounding = copy.deepcopy(rag_grounding or {})
    surface_quantities = dict(surface_quantities or {})
    confirmed = [
        choice
        for choice in confirmed_choices or []
        if _folded_text(choice.get("surface_text"))
        and str(choice.get("model_asset_id") or "").strip()
    ]
    for match in grounding.get("asset_matches") or []:
        if not isinstance(match, dict):
            continue
        match_surface = str(match.get("surface_text") or "").strip()
        match_quantity = surface_quantities.get(_folded_text(match_surface))
        compatible = [
            choice
            for choice in confirmed
            if quantity_insensitive_surface_equivalent(
                choice.get("surface_text"),
                match_surface,
                left_quantity=_leading_quantity_from_surface(
                    choice.get("surface_text")
                ),
                right_quantity=match_quantity,
            )
        ]
        if not compatible:
            # The next linguistic pass may use the catalog title rather than
            # the original role name. A title with an appositive still denotes
            # the confirmed type; a generic new role does not.
            folded_match = _folded_text(match_surface)
            compatible = [choice for choice in confirmed
                if (title := _folded_text(str(choice.get("display_label") or "").split(" (")[0]))
                and (folded_match == _folded_text(choice.get("display_label"))
                     or folded_match == title or folded_match.startswith(title + ","))]
        selected_ids = {str(c.get("model_asset_id") or "").strip() for c in compatible}
        if len(selected_ids) != 1:
            continue
        selected = selected_ids.pop()
        candidates = {
            str(
                item.get("model_asset_id") if isinstance(item, dict) else item
            ).strip()
            for item in match.get("candidate_model_asset_ids") or []
        }
        if selected not in candidates:
            continue
        match["selected_model_asset_id"] = selected
        match["decision"] = "resolved"
        match["ambiguity_status"] = "resolved"
        match["decision_source"] = "user_confirmed_catalog_choice"
    return grounding


def _leading_quantity_from_surface(value):
    match = re.match(
        r"^\s*(?P<count>\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\b",
        str(value or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    token = match.group("count").casefold()
    words = {
        "a": 1, "an": 1, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10,
    }
    return int(token) if token.isdigit() else words.get(token)


def _monolithic_initial_layout_context(scene_context):
    layout = scene_context.get("last_layout") or scene_context.get("seed_layout")
    if not isinstance(layout, dict):
        return None
    site = layout.get("site") or {}
    dimensions = site.get("dimensions_m") or site
    return {
        "layout_id": layout.get("layout_id"),
        "site": {
            "width_m": dimensions.get("width_m"),
            "depth_m": dimensions.get("depth_m"),
        },
        "equipment_instances": [
            {
                "instance_id": placement.get("instance_id"),
                "model_asset_id": placement.get("model_asset_id"),
            }
            for placement in (layout.get("placements") or [])
            if isinstance(placement, dict)
        ],
    }


def _planned_edit_instance_directory(linguistic_frame, grounding, directory):
    """Bind declared additions for language/route resolution before placement.

    This is a prospective name table only. The authoritative physical inventory
    remains unchanged until ExactEditExecutor applies the typed operations.
    """
    result=copy.deepcopy(directory or [])
    rag_by_surface={_folded_text(m.get("surface_text")):m for m in grounding.get("asset_matches") or []}
    operations=_bind_linguistic_edit_operations(
        linguistic_frame.get("edit_operations") or [],
        rag_by_surface,
        instance_directory=directory,
    )
    next_id=max([int(r["instance_id"].rsplit("_",1)[-1]) for r in result if r["instance_id"].rsplit("_",1)[-1].isdigit()] or [0])+1
    for operation in operations:
        if operation.get("op_type")=="add" and not operation.get("new_model_asset_id"):
            operation["new_model_asset_id"]=operation.get("target_model_asset_id")
        if operation.get("op_type")!="add" or not operation.get("new_model_asset_id"):continue
        model=operation["new_model_asset_id"]
        # Prospective identities must come from a real retrieved selection.
        if model not in {m.get("selected_model_asset_id") for m in grounding.get("asset_matches") or []}:continue
        quantity=int(operation.get("quantity") or 1)
        alias=str(operation.get("surface_text") or "").strip()
        alias=re.sub(r"^(?:add\s+)?(?:one|a|an)\s+", "", alias, flags=re.I)
        alias=re.sub(r"^(?:new|additional)\s+", "", alias, flags=re.I)
        alias=re.sub(r"\s+(?:labeled|labelled)\s+([A-Za-z])$",r" \1",alias,flags=re.I)
        ordinal=max([r["quantity_index"] for r in result if r["model_asset_id"]==model] or [0])+1
        for index in range(quantity):
            label=alias if quantity==1 else alias+f" {index+1}"
            result.append(dict(instance_id=f"equipment_{next_id:03d}",model_asset_id=model,quantity_index=ordinal+index,stable_alias=label))
            next_id+=1
    return _normalize_instance_directory(result)


def _current_instance_directory(scene_context):
    scene_context = scene_context or {}
    explicit = scene_context.get("instance_directory")
    if explicit is not None:
        return _normalize_instance_directory(explicit)
    resolved_directory=AuthoritativeContextResolver().resolve("",scene_context).get("instance_directory") or []
    if resolved_directory:
        return _normalize_instance_directory(resolved_directory)

    flp_ir = scene_context.get("last_flp_ir") or scene_context.get("base_flp_ir") or {}
    equipment_instances = flp_ir.get("equipment_instances") or []
    if equipment_instances:
        rows = []
        model_counts = {}
        for item in equipment_instances:
            if not isinstance(item, dict):
                continue
            model_asset_id = str(item.get("model_asset_id") or "").strip()
            instance_id = str(item.get("instance_id") or "").strip()
            if not model_asset_id or not instance_id:
                continue
            model_counts[model_asset_id] = model_counts.get(model_asset_id, 0) + 1
            quantity_index = item.get("quantity_index") or model_counts[model_asset_id]
            rows.append(
                {
                    "instance_id": instance_id,
                    "model_asset_id": model_asset_id,
                    "quantity_index": quantity_index,
                    "stable_alias": str(item.get("stable_alias") or "").strip()
                    or f"{model_asset_id}#{quantity_index}",
                }
            )
        if rows:
            return _normalize_instance_directory(rows)

    layout = scene_context.get("last_layout") or scene_context.get("seed_layout") or {}
    placements = [
        item
        for item in (layout.get("placements") or [])
        if isinstance(item, dict)
        and item.get("instance_id")
        and item.get("model_asset_id")
    ]
    model_counts = {}
    rows = []
    for item in sorted(
        placements,
        key=lambda value: (
            str(value.get("model_asset_id")),
            str(value.get("instance_id")),
        ),
    ):
        model_asset_id = str(item["model_asset_id"])
        model_counts[model_asset_id] = model_counts.get(model_asset_id, 0) + 1
        quantity_index = model_counts[model_asset_id]
        rows.append(
            {
                "instance_id": str(item["instance_id"]),
                "model_asset_id": model_asset_id,
                "quantity_index": quantity_index,
                "stable_alias": str(item.get("stable_alias") or "").strip()
                or f"{model_asset_id}#{quantity_index}",
            }
        )
    return _normalize_instance_directory(rows)


def _normalize_instance_directory(value):
    if value is None:
        return []
    if not isinstance(value, list):
        raise RAGBlockedError(
            "invalid_instance_directory",
            "The current equipment instance directory must be a list.",
        )
    allowed = {"instance_id", "model_asset_id", "quantity_index", "stable_alias"}
    rows = []
    instance_ids = set()
    model_indices = set()
    aliases = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict) or set(item) != allowed:
            raise RAGBlockedError(
                "invalid_instance_directory",
                "The instance directory contains non-minimal or missing fields.",
                details={"row": index, "allowed_fields": sorted(allowed)},
            )
        instance_id = str(item.get("instance_id") or "").strip()
        model_asset_id = str(item.get("model_asset_id") or "").strip()
        quantity_index = item.get("quantity_index")
        raw_alias = item.get("stable_alias")
        if isinstance(raw_alias, dict):
            if set(raw_alias) != {"zh", "en"} or any(
                not isinstance(raw_alias.get(language), str)
                or not raw_alias[language].strip()
                for language in ("zh", "en")
            ):
                raise RAGBlockedError(
                    "invalid_instance_directory",
                    "Localized stable aliases require non-empty zh/en strings.",
                    details={"row": index},
                )
            stable_alias = {
                "zh": raw_alias["zh"].strip(),
                "en": raw_alias["en"].strip(),
            }
            alias_keys = {
                (language, stable_alias[language].casefold())
                for language in ("zh", "en")
            }
        elif isinstance(raw_alias, str) and raw_alias.strip():
            stable_alias = raw_alias.strip()
            alias_keys = {("legacy", stable_alias.casefold())}
        else:
            raise RAGBlockedError(
                "invalid_instance_directory",
                "A stable alias must be a localized zh/en object or legacy string.",
                details={"row": index},
            )
        if (
            not instance_id
            or not model_asset_id
            or isinstance(quantity_index, bool)
            or not isinstance(quantity_index, int)
            or quantity_index < 1
        ):
            raise RAGBlockedError(
                "invalid_instance_directory",
                "The instance directory contains an invalid identity or 1-based quantity index.",
                details={"row": index},
            )
        model_index = (model_asset_id, quantity_index)
        if (
            instance_id in instance_ids
            or model_index in model_indices
            or bool(alias_keys & aliases)
        ):
            raise RAGBlockedError(
                "invalid_instance_directory",
                "Instance IDs, model quantity indices, and stable aliases must be unique.",
                details={"row": index},
            )
        instance_ids.add(instance_id)
        model_indices.add(model_index)
        aliases.update(alias_keys)
        rows.append(
            {
                "instance_id": instance_id,
                "model_asset_id": model_asset_id,
                "quantity_index": quantity_index,
                "stable_alias": stable_alias,
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            item["model_asset_id"],
            item["quantity_index"],
            item["instance_id"],
        ),
    )


def _monolithic_evidence_ids(rag_evidence):
    evidence_ids = []
    for result in rag_evidence.get("results") or []:
        evidence_id = result.get("evidence_id") if isinstance(result, dict) else None
        if evidence_id and evidence_id not in evidence_ids:
            evidence_ids.append(str(evidence_id))
    for match in rag_evidence.get("asset_matches") or []:
        for value in (match.get("mention_id"), match.get("source_case_id")):
            if value and str(value) not in evidence_ids:
                evidence_ids.append(str(value))
    if not evidence_ids:
        backend = rag_evidence.get("backend") or {}
        evidence_ids.append(
            f"rag_backend:{backend.get('service_class') or 'unknown'}"
        )
    return evidence_ids


def _validate_monolithic_brief(
    frame,
    user_message,
    rag_evidence,
    adapter,
    *,
    instance_directory=None,
):
    if not isinstance(frame, dict):
        raise RAGBlockedError(
            "invalid_monolithic_design_brief",
            "Monolithic design brief must be a JSON object.",
        )
    forbidden_keys = {
        "coordinates",
        "generations",
        "layout",
        "placements",
        "population_size",
        "position",
        "rotation",
        "solver",
        "solver_options",
        "x_m",
        "y_m",
        "yaw_deg",
        "z_m",
    }

    def walk(value, path="brief"):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in forbidden_keys:
                    raise RAGBlockedError(
                        "monolithic_geometry_forbidden",
                        f"Monolithic design brief may not contain solver geometry at {path}.{key}.",
                    )
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(frame)
    allowed_asset_ids = {
        str(match.get("selected_model_asset_id"))
        for match in rag_evidence.get("asset_matches") or []
        if match.get("selected_model_asset_id")
    }
    for match in rag_evidence.get("asset_matches") or []:
        for candidate in match.get("candidate_model_asset_ids") or []:
            model_asset_id = candidate.get("model_asset_id") if isinstance(candidate, dict) else candidate
            if model_asset_id:
                allowed_asset_ids.add(str(model_asset_id))
    message = str(user_message or "")
    message_folded = " ".join(message.casefold().split())
    allowed_asset_ids.update(
        model_asset_id
        for model_asset_id in adapter.assets_by_id
        if model_asset_id.casefold() in message.casefold()
    )
    for mention in frame.get("asset_mentions") or []:
        if not isinstance(mention, dict):
            continue
        model_asset_id = mention.get("model_asset_id")
        surface = " ".join(str(mention.get("surface_text") or "").casefold().split())
        if model_asset_id and str(model_asset_id) not in allowed_asset_ids:
            raise RAGBlockedError(
                "monolithic_asset_not_grounded",
                f"Asset ID is not grounded in Engineering RAG evidence: {model_asset_id}",
            )
        if not model_asset_id and (not surface or surface not in message_folded):
            raise RAGBlockedError(
                "monolithic_asset_not_grounded",
                "Every asset mention must occur in the request or resolve to a RAG candidate.",
            )
    _validate_process_relation_grounding(
        frame,
        user_message,
        allowed_asset_ids,
        instance_directory,
        blocker_type="monolithic_process_relation_not_grounded",
    )
    site = frame.get("site")
    if isinstance(site, dict) and site:
        source_text = " ".join(str(site.get("source_text") or "").casefold().split())
        if not source_text or source_text not in message_folded:
            raise RAGBlockedError(
                "monolithic_site_not_grounded",
                "Site dimensions in the monolithic brief must cite an exact request span.",
            )
    for field_name in ("constraints", "objectives"):
        for item in frame.get(field_name) or []:
            if not isinstance(item, dict):
                continue
            source_text = " ".join(str(item.get("source_text") or "").casefold().split())
            is_default_no_overlap = (
                field_name == "constraints"
                and item.get("constraint_type") == "no_overlap"
                and source_text == "default"
            )
            if not is_default_no_overlap and (
                not source_text or source_text not in message_folded
            ):
                raise RAGBlockedError(
                    "monolithic_requirement_not_grounded",
                    f"{field_name} must cite an exact request span.",
                )


def _valid_intent_contract(*, repairs=None):
    return {
        "schema_version": "layout-agentic-intent-contract-v2",
        "status": "VALID",
        "issues": [],
        "repairs": copy.deepcopy(repairs or []),
    }


def _known_site_features(context):
    context = context or {}
    for site in (context.get("site_snapshot"), (context.get("last_layout") or {}).get("site"),
                 ((context.get("last_flp_ir") or {}).get("site_ref") or {}).get("site_snapshot")):
        if isinstance(site, dict) and site.get("site_features"):
            return site["site_features"]
    return []


def _normalize_linguistic_frame(
    frame,
    user_message,
    *,
    conversation_history=None,
    site_features=None,
):
    """Normalize the pre-RAG frame while retaining only literal mention spans."""

    if not isinstance(frame, dict):
        return {
            "intent_type": "unknown_or_unsupported",
            "confidence": 0.0,
            "site": None,
            "asset_mentions": [],
            "domain_intent": None,
            "constraints": [],
            "objectives": [],
            "process_relations": [],
            "edit_operations": [],
            "requires_existing_layout": False,
            "missing_information": [
                "The linguistic slot extractor did not return a JSON object."
            ],
            "clarification_questions": [],
            "reasoning_summary": "Linguistic slot extraction requires revision.",
            "extraction_issues": ["invalid_linguistic_frame"],
        }
    literal_sources = _linguistic_literal_sources(
        user_message,
        conversation_history,
    )
    raw_source_messages = _raw_user_source_messages(
        user_message,
        conversation_history,
    )
    declared_site_features = _declared_site_feature_references(raw_source_messages)
    extraction_issues = []
    asset_mentions = []
    for index, item in enumerate(frame.get("asset_mentions") or []):
        if not isinstance(item, dict):
            extraction_issues.append(f"asset_mentions[{index}]_not_object")
            continue
        surface_text = str(item.get("surface_text") or "").strip()
        if not _literal_span_in_sources(surface_text, literal_sources):
            extraction_issues.append(
                f"asset_mentions[{index}]_not_literal_span"
            )
            continue
        from tool.logistics.facility_logistics import existing_facility_surface
        if existing_facility_surface(surface_text, site_features):
            declared_site_features[_folded_text(surface_text)] = {"feature_kind":"existing_facility_group","source_text":surface_text}
        if declared_site_features.get(_folded_text(surface_text)) or facility_role(surface_text) in declared_site_features:
            extraction_issues.append(
                f"asset_mentions[{index}]_declared_site_feature"
            )
            continue
        raw_quantity = item.get("quantity")
        quantity = int(_optional_float(raw_quantity) or 1)
        asset_mentions.append(
            {
                "surface_text": surface_text,
                "quantity": max(1, quantity),
            }
        )
    objectives, objective_source_repairs = _repair_objective_source_slots(
        frame.get("objectives"),
        literal_sources,
        raw_source_messages,
    )
    extraction_issues.extend(
        f"objectives[{item['index']}]_{item['action']}"
        for item in objective_source_repairs
    )
    result = {
        "intent_type": str(
            frame.get("intent_type") or "unknown_or_unsupported"
        ).strip(),
        "confidence": _bounded_float(frame.get("confidence"), default=0.0),
        "site": _normalize_site_slot(frame.get("site")),
        "asset_mentions": asset_mentions,
        "domain_intent": _normalize_domain_intent(frame.get("domain_intent")),
        "constraints": _normalize_constraint_slots(frame.get("constraints")),
        "objectives": objectives,
        "process_relations": copy.deepcopy(
            [
                item
                for item in frame.get("process_relations") or []
                if isinstance(item, dict)
            ]
        ),
        "edit_operations": copy.deepcopy(
            [
                item
                for item in frame.get("edit_operations") or []
                if isinstance(item, dict)
            ]
        ),
        "requires_existing_layout": bool(
            frame.get("requires_existing_layout")
        ),
        "missing_information": _without_site_feature_asset_identity_requests(
            _string_list(frame.get("missing_information")),
            declared_site_features,
        ),
        "clarification_questions": _without_site_feature_asset_identity_requests(
            _string_list(frame.get("clarification_questions")),
            declared_site_features,
        ),
        "reasoning_summary": str(
            frame.get("reasoning_summary")
            or "Linguistic slots extracted before asset grounding."
        )[:400],
        "extraction_issues": extraction_issues,
    }
    if _explicit_central_aisle_width_repair(user_message):
        result["intent_type"] = "constraint_repair"
        result["requires_existing_layout"] = True
        result["edit_operations"] = []
        result["asset_mentions"] = [
            item
            for item in result["asset_mentions"]
            if not _site_repair_reference_surface(item.get("surface_text"))
        ]
        result["missing_information"] = [
            item
            for item in result["missing_information"]
            if not _site_repair_known_information(item)
        ]
        result["clarification_questions"] = [
            item
            for item in result["clarification_questions"]
            if not _site_repair_known_information(item)
        ]
    return result


def _declared_site_feature_references(raw_source_messages):
    """Map literal named floor features that the site grammar has already defined.

    A mention is removed from asset grounding only when its exact name comes
    from a source span that declares a fixed, dimensioned buffer area and a
    center/centre location.  This keeps physical equipment such as a “buffer
    station” eligible for RAG even though it contains the same word.
    """

    features = {}
    for source in raw_source_messages or []:
        source_text = str(source or "")
        for role in declared_facility_roles(source_text):
            features[role] = {"feature_kind": "facility_logistics_endpoint", "source_text": source_text}
        spans = natural_site_literal_spans(source)
        buffer_literal = str(spans.get("buffer_requirements") or "").strip()
        if not re.search(
            r"\b\d+(?:\.\d+)?\s*(?:m|met(?:er|re)s?)\s*"
            r"(?:by|x|×)\s*\d+(?:\.\d+)?\s*(?:m|met(?:er|re)s?)\s+"
            r"(?:fixed\s+)?buffers?\b",
            source_text,
            re.IGNORECASE,
        ):
            continue
        for match in re.finditer(
            r"\b(?:center|centre)\s+(?:the\s+)?"
            r"(?P<name>[a-z][a-z0-9-]*(?:\s+[a-z][a-z0-9-]*){0,4})\s+buffer\b",
            source_text,
            re.IGNORECASE,
        ):
            surface = f"{match.group('name').strip()} buffer"
            folded = _folded_text(surface)
            if folded:
                features[folded] = {
                    "feature_kind": "fixed_buffer_area",
                    "source_text": buffer_literal or source_text,
                }
    return features


def _without_site_feature_asset_identity_requests(values, declared_site_features):
    """Drop only erroneous asset-identity questions for declared floor features."""

    retained = []
    for value in values or []:
        text = str(value or "").strip()
        folded = _folded_text(text)
        refers_to_feature = any(name in folded for name in declared_site_features)
        asks_for_asset_identity = bool(
            re.search(
                r"\b(?:model|asset|catalog|equipment|machine|storage\s+unit)\b",
                folded,
            )
        )
        if refers_to_feature and asks_for_asset_identity:
            continue
        retained.append(text)
    return retained


def _explicit_central_aisle_width_repair(value):
    text = str(value or "")
    from tool.layout.natural_site_requirements import _aisle_width_update
    return bool(
        re.search(r"\b(?:aisle|corridor)\b", text, re.IGNORECASE)
        and re.search(
            r"\b(?:must\s+now|update|widen|wider|make\s+room)\b",
            text,
            re.IGNORECASE,
        )
        and _aisle_width_update(text) is not None
        and re.search(r"\b(?:center|centre)line\b[^.\n]*\b(?:unchanged|fixed)\b", text, re.IGNORECASE)
    )


def _site_repair_reference_surface(value):
    tokens = _process_surface_tokens(value)
    return bool(tokens) and tokens <= {
        "central",
        "logistic",
        "aisle",
        "corridor",
        "unit",
        "buffer",
        "area",
    }


def _site_repair_known_information(value):
    lowered = _folded_text(value)
    return bool(
        re.search(r"\b(?:which\s+)?units?\s+(?:currently\s+)?obstruct", lowered)
        or re.search(r"\bcurrent\s+positions?\b", lowered)
        or re.search(r"\b(?:exact\s+)?(?:y-?coordinate|centerline|centreline)\b", lowered)
    )


def _linguistic_asset_surfaces(frame, user_message=None):
    message_folded = (
        _folded_text(user_message) if user_message is not None else None
    )
    surfaces = []
    for item in (frame or {}).get("asset_mentions") or []:
        if not isinstance(item, dict):
            continue
        surface_text = str(item.get("surface_text") or "").strip()
        if (
            not surface_text
            or (
                message_folded is not None
                and _folded_text(surface_text) not in message_folded
            )
        ):
            continue
        if surface_text not in surfaces:
            surfaces.append(surface_text)
    for operation in (frame or {}).get("edit_operations") or []:
        if operation.get("op_type") not in {"add","replace"}:continue
        surface=str(operation.get("new_asset_surface_text") or "").strip()
        if surface and (message_folded is None or _folded_text(surface) in message_folded) and surface not in surfaces:
            surfaces.append(surface)
    return surfaces


def _has_resolved_rag_selection(match):
    if not isinstance(match, dict):
        return False
    selected_model_asset_id = str(
        match.get("selected_model_asset_id") or ""
    ).strip()
    resolution = str(
        match.get("decision") or match.get("ambiguity_status") or ""
    ).strip().upper()
    return bool(
        selected_model_asset_id
        and (not resolution or resolution in {"RESOLVED", "RESOLVE"})
    )


def _repair_grounded_intent_frame(
    frame,
    user_message,
    rag_grounding,
    linguistic_frame,
    *,
    instance_directory=None,
    conversation_history=None,
):
    """Apply evidence-preserving repairs before asking an LLM to revise."""

    if not isinstance(frame, dict):
        return frame, []
    repaired = copy.deepcopy(frame)
    linguistic_frame = copy.deepcopy(linguistic_frame or {})
    literal_sources = _linguistic_literal_sources(
        user_message,
        conversation_history,
    )
    natural_source_messages = _raw_user_source_messages(
        user_message,
        conversation_history,
    )
    repairs = []

    def record(field_path, action, basis):
        repairs.append(
            {
                "field_path": field_path,
                "action": action,
                "basis": basis,
            }
        )

    repaired_objectives, objective_source_repairs = (
        _repair_objective_source_slots(
            repaired.get("objectives"),
            literal_sources,
            natural_source_messages,
        )
    )
    repaired["objectives"] = repaired_objectives
    for objective_repair in objective_source_repairs:
        record(
            f"objectives[{objective_repair['index']}]",
            objective_repair["action"],
            objective_repair["basis"],
        )

    linguistic_intent = str(
        linguistic_frame.get("intent_type") or ""
    ).strip()
    if _explicit_central_aisle_width_repair(user_message):
        repaired["intent_type"] = "constraint_repair"
        repaired["requires_existing_layout"] = True
        repaired["edit_operations"] = []
        repaired["missing_information"] = [
            item
            for item in repaired.get("missing_information") or []
            if not _site_repair_known_information(item)
        ]
        repaired["clarification_questions"] = [
            item
            for item in repaired.get("clarification_questions") or []
            if not _site_repair_known_information(item)
        ]
        record(
            "intent_type",
            "route_explicit_central_aisle_width_repair",
            "literal_current_user_input",
        )
    if (
        str(repaired.get("intent_type") or "").strip()
        in {"", "unknown_or_unsupported"}
        and linguistic_intent
        and linguistic_intent != "unknown_or_unsupported"
    ):
        repaired["intent_type"] = linguistic_intent
        record("intent_type", "restore_linguistic_intent", "linguistic_frame")

    linguistic_site = linguistic_frame.get("site")
    current_site = repaired.get("site")
    if isinstance(linguistic_site, dict):
        linguistic_source = str(
            linguistic_site.get("source_text") or ""
        ).strip()
        if not isinstance(current_site, dict) or not current_site:
            repaired["site"] = copy.deepcopy(linguistic_site)
            record("site", "restore_missing_slot", "linguistic_frame")
        elif (
            linguistic_source
            and _literal_span_in_sources(
                linguistic_source,
                literal_sources,
            )
            and (
                _optional_float(current_site.get("width_m"))
                == _optional_float(linguistic_site.get("width_m"))
                and _optional_float(current_site.get("depth_m"))
                == _optional_float(linguistic_site.get("depth_m"))
            )
            and not _literal_span_in_sources(
                current_site.get("source_text"),
                literal_sources,
            )
        ):
            repaired["site"]["source_text"] = linguistic_source
            record("site.source_text", "restore_literal_span", "linguistic_frame")

    for field_name, type_name in (
        ("constraints", "constraint_type"),
        ("objectives", "objective_type"),
    ):
        linguistic_by_type = {
            str(item.get(type_name) or ""): item
            for item in linguistic_frame.get(field_name) or []
            if isinstance(item, dict) and item.get(type_name)
        }
        if not repaired.get(field_name) and linguistic_by_type:
            repaired[field_name] = copy.deepcopy(
                list(linguistic_by_type.values())
            )
            record(field_name, "restore_missing_slots", "linguistic_frame")
        for index, item in enumerate(repaired.get(field_name) or []):
            if not isinstance(item, dict):
                continue
            source_text = str(item.get("source_text") or "").strip()
            if (
                field_name == "constraints"
                and item.get("constraint_type") == "no_overlap"
                and _folded_text(source_text) == "default"
            ):
                continue
            if _literal_span_in_sources(source_text, literal_sources):
                if field_name == "constraints":
                    canonical_type = canonical_natural_site_constraint_type(
                        user_message,
                        item.get("source_text"),
                        source_messages=natural_source_messages,
                    )
                    if (
                        canonical_type
                        and item.get("constraint_type") != canonical_type
                    ):
                        item["constraint_type"] = canonical_type
                        record(
                            f"{field_name}[{index}].constraint_type",
                            "canonicalize_recognized_natural_site_constraint",
                            "natural_site_requirements",
                        )
                elif field_name == "objectives":
                    canonical_type = canonical_objective_type_from_literal(
                        user_message,
                        item.get("source_text"),
                    )
                    if (
                        canonical_type
                        and item.get("objective_type") != canonical_type
                    ):
                        item["objective_type"] = canonical_type
                        record(
                            f"{field_name}[{index}].objective_type",
                            "canonicalize_recognized_weighted_flow_objective",
                            "literal_user_input",
                        )
                continue
            linguistic_item = linguistic_by_type.get(
                str(item.get(type_name) or "")
            )
            literal_source = str(
                (linguistic_item or {}).get("source_text") or ""
            ).strip()
            if _literal_span_in_sources(
                literal_source,
                literal_sources,
            ):
                item["source_text"] = literal_source
                record(
                    f"{field_name}[{index}].source_text",
                    "restore_literal_span",
                    "linguistic_frame",
                )
                if field_name == "objectives":
                    canonical_type = canonical_objective_type_from_literal(
                        user_message,
                        item.get("source_text"),
                    )
                    if (
                        canonical_type
                        and item.get("objective_type") != canonical_type
                    ):
                        item["objective_type"] = canonical_type
                        record(
                            f"{field_name}[{index}].objective_type",
                            "canonicalize_recognized_weighted_flow_objective",
                            "literal_user_input",
                        )
                continue
            if (
                field_name == "constraints"
                and item.get("constraint_type") == "area_assignment"
            ):
                group_regions = natural_site_literal_spans(user_message).get(
                    "group_regions"
                )
                group_directions = {
                    direction.casefold()
                    for direction in re.findall(
                        r"(?im)^\s*(?:[-*]\s*)?(south|north|west|east)\s*:",
                        str(group_regions or ""),
                    )
                }
                if (
                    len(group_directions) >= 2
                    and _literal_span_in_sources(group_regions, literal_sources)
                ):
                    item["source_text"] = group_regions
                    record(
                        f"{field_name}[{index}].source_text",
                        "restore_group_region_literal_span",
                        "natural_site_requirements",
                    )
            if (
                field_name == "constraints"
                and item.get("constraint_type")
                in {"buffer_clearance", "buffer_unobstructed"}
            ):
                unobstructed_areas = natural_site_literal_spans(user_message).get(
                    "unobstructed_areas"
                )
                if _literal_span_in_sources(
                    unobstructed_areas,
                    literal_sources,
                ):
                    item["source_text"] = unobstructed_areas
                    record(
                        f"{field_name}[{index}].source_text",
                        "restore_unobstructed_area_literal_span",
                        "natural_site_requirements",
                    )
            if field_name == "constraints":
                canonical_type = canonical_natural_site_constraint_type(
                    user_message,
                    item.get("source_text"),
                    source_messages=natural_source_messages,
                )
                if (
                    canonical_type
                    and item.get("constraint_type") != canonical_type
                ):
                    item["constraint_type"] = canonical_type
                    record(
                        f"{field_name}[{index}].constraint_type",
                        "canonicalize_recognized_natural_site_constraint",
                        "natural_site_requirements",
                    )
            elif field_name == "objectives":
                canonical_type = canonical_objective_type_from_literal(
                    user_message,
                    item.get("source_text"),
                )
                if (
                    canonical_type
                    and item.get("objective_type") != canonical_type
                ):
                    item["objective_type"] = canonical_type
                    record(
                        f"{field_name}[{index}].objective_type",
                        "canonicalize_recognized_weighted_flow_objective",
                        "literal_user_input",
                    )

    rag_matches = [
        item
        for item in rag_grounding.get("asset_matches") or []
        if _has_resolved_rag_selection(item)
    ]
    by_surface = {
        _folded_text(item.get("surface_text")): item
        for item in rag_matches
        if item.get("surface_text")
    }
    from tool.context.conversation_deltas import retain_confirmed_role_declarations
    grounded_roles = [dict(surface_text=m.get("surface_text"),quantity=m.get("quantity"),
        model_asset_id=(by_surface.get(_folded_text(m.get("surface_text"))) or {}).get("selected_model_asset_id"))
        for m in linguistic_frame.get("asset_mentions") or []]
    before_role_restore = copy.deepcopy(repaired.get("asset_mentions") or [])
    repaired, _unused_grounding, restored_roles = retain_confirmed_role_declarations(
        repaired, rag_grounding, grounded_roles)
    if restored_roles:
        for mention in repaired.get("asset_mentions") or []:
            selected = (by_surface.get(_folded_text(mention.get("surface_text"))) or {}).get("selected_model_asset_id")
            sources = [m for m in before_role_restore
                       if selected and m.get("model_asset_id") == selected
                       and m.get("quantity") == mention.get("quantity")]
            if len(sources) == 1:
                for key in ("planning_overrides", "instance_labels"):
                    if sources[0].get(key):
                        mention[key] = copy.deepcopy(sources[0][key])
        record("asset_mentions", "restore_confirmed_station_role_aliases", "linguistic_declarations+retrieved_identities")
    by_asset = {}
    for item in rag_matches:
        selected = item.get("selected_model_asset_id")
        if selected:
            by_asset.setdefault(str(selected), item)
        # Retrieval alternatives are not aliases for the selected equipment.
        # Binding an alternative here can silently merge unrelated declarations.
    linguistic_quantities = {
        _folded_text(item.get("surface_text")): max(
            1,
            int(_optional_float(item.get("quantity")) or 1),
        )
        for item in linguistic_frame.get("asset_mentions") or []
        if isinstance(item, dict) and item.get("surface_text")
    }

    def equivalent_declaration(mentions, surface, model_id, quantity):
        key = _folded_text(surface)
        candidates = [m for m in mentions
                      if model_id and str(m.get("model_asset_id") or "") == str(model_id)
                      and int(_optional_float(m.get("quantity")) or 1) == quantity
                      and (key in _folded_text(m.get("surface_text"))
                           or _folded_text(m.get("surface_text")) in key)]
        return candidates[0] if len(candidates) == 1 else None

    repaired_mentions = []
    for index, mention in enumerate(repaired.get("asset_mentions") or []):
        if not isinstance(mention, dict):
            continue
        surface_key = _folded_text(mention.get("surface_text"))
        model_asset_id = str(mention.get("model_asset_id") or "").strip()
        match = by_surface.get(surface_key) or by_asset.get(model_asset_id)
        if match is None and surface_key:
            compatible_matches = [
                item
                for key, item in by_surface.items()
                if surface_key in key or key in surface_key
            ]
            if len(compatible_matches) == 1:
                match = compatible_matches[0]
        if match:
            literal_surface = str(match.get("surface_text") or "").strip()
            selected_asset_id = str(
                match.get("selected_model_asset_id") or model_asset_id
            ).strip()
            if literal_surface and surface_key != _folded_text(literal_surface):
                mention["surface_text"] = literal_surface
                record(
                    f"asset_mentions[{index}].surface_text",
                    "restore_rag_mention_span",
                    "engineering_rag_grounding",
                )
            if selected_asset_id and model_asset_id != selected_asset_id:
                mention["model_asset_id"] = selected_asset_id
                record(
                    f"asset_mentions[{index}].model_asset_id",
                    "bind_selected_rag_asset",
                    "engineering_rag_grounding",
                )
            quantity = linguistic_quantities.get(
                _folded_text(literal_surface)
            )
            if quantity is not None and mention.get("quantity") != quantity:
                mention["quantity"] = quantity
                record(
                    f"asset_mentions[{index}].quantity",
                    "restore_linguistic_quantity",
                    "linguistic_frame",
                )
        repaired_mentions.append(mention)
    present_surfaces = {
        _folded_text(item.get("surface_text"))
        for item in repaired_mentions
        if isinstance(item, dict)
    }
    for match in rag_matches:
        surface_text = str(match.get("surface_text") or "").strip()
        selected_asset_id = str(
            match.get("selected_model_asset_id") or ""
        ).strip()
        surface_key = _folded_text(surface_text)
        if (
            not surface_text
            or not selected_asset_id
            or surface_key not in linguistic_quantities
            or surface_key in present_surfaces
            or equivalent_declaration(repaired_mentions, surface_text, selected_asset_id,
                                      linguistic_quantities[surface_key]) is not None
        ):
            continue
        repaired_mentions.append(
            {
                "surface_text": surface_text,
                "model_asset_id": selected_asset_id,
                "quantity": linguistic_quantities[surface_key],
            }
        )
        present_surfaces.add(surface_key)
        record(
            f"asset_mentions[{len(repaired_mentions) - 1}]",
            "restore_resolved_rag_mention",
            "linguistic_frame+engineering_rag_grounding",
        )
    for declaration in linguistic_frame.get("asset_mentions") or []:
        if not isinstance(declaration, dict):
            continue
        surface_key = _folded_text(declaration.get("surface_text"))
        matching = next((m for m in repaired_mentions
                         if _folded_text(m.get("surface_text")) == surface_key), None)
        if matching is None:
            selected = (by_surface.get(surface_key) or {}).get("selected_model_asset_id")
            matching = equivalent_declaration(
                repaired_mentions, declaration.get("surface_text"), selected,
                max(1, int(_optional_float(declaration.get("quantity")) or 1)))
        if matching is None and surface_key:
            matching = copy.deepcopy(declaration)
            matching["model_asset_id"] = (
                by_surface.get(surface_key) or {}
            ).get("selected_model_asset_id")
            repaired_mentions.append(matching)
            record("asset_mentions", "retain_requested_inventory", "linguistic_frame")
        if matching is not None:
            for key in ("planning_overrides", "instance_labels"):
                if declaration.get(key) and not matching.get(key):
                    matching[key] = copy.deepcopy(declaration[key])
    deduplicated = []
    mention_index = {}
    for mention in repaired_mentions:
        key = (
            str(mention.get("model_asset_id") or ""),
            _folded_text(mention.get("surface_text")),
        )
        if key in mention_index:
            existing = deduplicated[mention_index[key]]
            existing["quantity"] = max(
                int(_optional_float(existing.get("quantity")) or 1),
                int(_optional_float(mention.get("quantity")) or 1),
            )
            continue
        mention_index[key] = len(deduplicated)
        deduplicated.append(mention)
    directory_counts = {}
    for item in _normalize_instance_directory(instance_directory or []):
        model_asset_id = item["model_asset_id"]
        directory_counts[model_asset_id] = directory_counts.get(model_asset_id, 0) + 1
    keep_indices = set(range(len(deduplicated)))
    for model_asset_id, capacity in directory_counts.items():
        positions = [
            index
            for index, mention in enumerate(deduplicated)
            if str(mention.get("model_asset_id") or "") == model_asset_id
        ]
        quantities = {
            index: max(
                1,
                int(_optional_float(deduplicated[index].get("quantity")) or 1),
            )
            for index in positions
        }
        if sum(quantities.values()) <= capacity:
            continue
        exact = next(
            (index for index in positions if quantities[index] == capacity),
            None,
        )
        selected = (exact,) if exact is not None else None
        if selected is None:
            subsets = {0: ()}
            for index in positions:
                for total, subset in list(subsets.items()):
                    candidate_total = total + quantities[index]
                    if candidate_total <= capacity and candidate_total not in subsets:
                        subsets[candidate_total] = (*subset, index)
            selected = subsets.get(capacity)
        if selected is None:
            continue
        selected_set = set(selected)
        removed = [index for index in positions if index not in selected_set]
        keep_indices.difference_update(removed)
        if removed:
            record(
                "asset_mentions",
                "merge_coreferent_asset_mentions",
                "instance_directory+literal_user_spans",
            )
    repaired["asset_mentions"] = [
        mention
        for index, mention in enumerate(deduplicated)
        if index in keep_indices
    ]

    linguistic_edits = [
        item
        for item in linguistic_frame.get("edit_operations") or []
        if isinstance(item, dict)
    ]
    if linguistic_edits:
        bound_edits = _bind_linguistic_edit_operations(
            linguistic_edits,
            by_surface,
            existing_operations=repaired.get("edit_operations") or [],
            instance_directory=instance_directory,
        )
        if bound_edits != repaired.get("edit_operations"):
            repaired["edit_operations"] = bound_edits
            record(
                "edit_operations",
                "canonicalize_linguistic_edit_operations",
                "linguistic_frame+engineering_rag_grounding",
            )
    if _recover_explicit_move_deltas(
        repaired.get("edit_operations") or [],
        user_message,
    ):
        record(
            "edit_operations",
            "recover_explicit_move_delta",
            "literal_current_user_input",
        )

    linguistic_relations = [
        item
        for item in linguistic_frame.get("process_relations") or []
        if isinstance(item, dict)
    ]
    bound_relations = []
    relation_grounding = {}
    stage_type = str(
        repaired.get("intent_type")
        or linguistic_frame.get("intent_type")
        or ""
    ).casefold()
    read_context_routes = stage_type in {
        "create_initial_layout",
        "generate_layout",
        "generate_feasible_layout",
        "optimize_layout_from_scratch",
    }
    route_sources = [str(user_message or "")]
    if read_context_routes:
        route_sources.extend(
            str(item.get("text") or item.get("content") or "")
            for item in conversation_history or []
            if isinstance(item, dict)
            and str(item.get("role") or "").casefold()
            == "engineering_context"
        )
    arrow_relations = []
    seen_arrow_relations = set()
    for route_source in route_sources:
        for relation in _explicit_arrow_chain_relations(route_source):
            identity = (
                relation.get("source_text"),
                relation.get("from_surface_text"),
                relation.get("to_surface_text"),
                relation.get("flow_weight"),
            )
            if identity not in seen_arrow_relations:
                seen_arrow_relations.add(identity)
                arrow_relations.append(relation)
    if read_context_routes and len(route_sources) > 1:
        arrow_relations = _apply_context_carrier_assignments(
            arrow_relations,
            "\n".join(route_sources[1:]),
        )
    binding_source_text = "\n".join(
        source for source in route_sources if source.strip()
    )
    arrow_bound = _fully_bound_arrow_relations(
        arrow_relations,
        repaired.get("asset_mentions") or [],
        by_surface,
        binding_source_text,
        literal_sources,
        resource_relations=list(repaired.get("process_relations") or [])+linguistic_relations,
        instance_directory=instance_directory,
    )
    if arrow_bound is not None and repaired.get("process_relations") and any(item.get("flow_weight") is None for item in arrow_bound[0]):
        # An unweighted restatement cannot replace already reviewed volumes.
        arrow_bound = None
    if arrow_bound is not None:
        bound_relations, relation_grounding = arrow_bound
        repaired["process_relations"] = bound_relations
        record(
            "process_relations",
            "replace_with_fully_bound_explicit_arrow_chains",
            (
                "stable_engineering_context+engineering_rag_grounding"
                if len(route_sources) > 1
                else "literal_user_input+engineering_rag_grounding"
            ),
        )
    if linguistic_relations and arrow_bound is None:
        relation_grounding = dict(by_surface)
        relation_grounding.update(
            _declared_process_endpoint_bindings(
                linguistic_relations,
                linguistic_frame.get("asset_mentions") or [],
                by_surface,
            )
        )
        selected_asset_ids = {
            str(item.get("selected_model_asset_id") or "").strip()
            for item in rag_matches
            if item.get("selected_model_asset_id")
        }
        family_models = {}
        existing_by_source = {}
        for item in repaired.get("process_relations") or []:
            if not isinstance(item, dict) or not item.get("source_text"):
                continue
            existing_by_source.setdefault(
                _folded_text(item.get("source_text")), []
            ).append(item)
        for relation in linguistic_relations:
            existing = existing_by_source.get(
                _folded_text(relation.get("source_text")), []
            )
            if not existing:
                continue
            for prefix in ("from", "to"):
                surface = _folded_text(
                    relation.get(f"{prefix}_surface_text")
                )
                model_asset_ids = {
                    str(item.get(f"{prefix}_model_asset_id") or "")
                    for item in existing
                    if item.get(f"{prefix}_model_asset_id")
                }
                if (
                    surface
                    and surface not in relation_grounding
                    and len(model_asset_ids) == 1
                    and next(iter(model_asset_ids)) in selected_asset_ids
                ):
                    model_asset_id = next(iter(model_asset_ids))
                    relation_grounding[surface] = {
                        "selected_model_asset_id": model_asset_id
                    }
                    family = _process_alias_family(surface)
                    family_models.setdefault(family, set()).add(
                        model_asset_id
                    )
        for relation in linguistic_relations:
            for prefix in ("from", "to"):
                surface = _folded_text(
                    relation.get(f"{prefix}_surface_text")
                )
                family = _process_alias_family(surface)
                candidates = family_models.get(family) or set()
                if (
                    surface
                    and surface not in relation_grounding
                    and len(candidates) == 1
                ):
                    relation_grounding[surface] = {
                        "selected_model_asset_id": next(iter(candidates))
                    }
        bound_relations = _bind_linguistic_process_relations(
            linguistic_relations,
            relation_grounding,
        )
        merged_relations = [
            copy.deepcopy(item)
            for item in repaired.get("process_relations") or []
            if isinstance(item, dict)
        ]
        known_relations = {
            _process_relation_identity(item)
            for item in merged_relations
        }
        for relation in bound_relations:
            if _matching_semantic_relations(relation, merged_relations):
                continue
            # A route clause can support several directed edges. Its text alone
            # is therefore not a relation identity. Restore a linguistic edge
            # only when it has literal source evidence and grounded endpoints.
            if not _can_restore_linguistic_relation(
                relation,
                literal_sources,
            ):
                continue
            relation_identity = _process_relation_identity(relation)
            if relation_identity in known_relations:
                continue
            merged_relations.append(relation)
            known_relations.add(relation_identity)
        if merged_relations != repaired.get("process_relations"):
            repaired["process_relations"] = merged_relations
            record(
                "process_relations",
                "restore_explicit_process_relations",
                "linguistic_frame+engineering_rag_grounding",
            )
    for relation in repaired.get("process_relations") or []:
        for prefix in ("from", "to"):
            role = relation.get(f"{prefix}_site_feature_name") or facility_role(relation.get(f"{prefix}_surface_text"))
            if role:
                relation[f"{prefix}_site_feature_name"] = role
                relation.pop(f"{prefix}_model_asset_id", None)
    for index, relation in enumerate(repaired.get("process_relations") or []):
        if not isinstance(relation, dict):
            continue
        linguistic_relation = _unique_matching_linguistic_relation(
            relation,
            bound_relations,
        )
        if not linguistic_relation:
            continue
        source_text = str(relation.get("source_text") or "").strip()
        literal_source = str(
            linguistic_relation.get("source_text") or ""
        ).strip()
        if (
            not _literal_span_in_sources(source_text, literal_sources)
            and literal_source
            and _literal_span_in_sources(
                literal_source,
                literal_sources,
            )
        ):
            relation["source_text"] = literal_source
            record(
                f"process_relations[{index}].source_text",
                "restore_literal_span",
                "linguistic_frame",
            )
        for prefix in ("from", "to"):
            surface = str(
                linguistic_relation.get(f"{prefix}_surface_text") or ""
            ).strip()
            match = relation_grounding.get(_folded_text(surface))
            selected = str(
                (match or {}).get("selected_model_asset_id") or ""
            ).strip()
            field = f"{prefix}_model_asset_id"
            if selected and relation.get(field) != selected:
                relation[field] = selected
                record(
                    f"process_relations[{index}].{field}",
                    "bind_selected_rag_asset",
                    "linguistic_frame+engineering_rag_grounding",
                )
            quantity_field = f"{prefix}_quantity_index"
            linguistic_index = linguistic_relation.get(quantity_field)
            if (
                relation.get(quantity_field) is None
                and isinstance(linguistic_index, int)
                and not isinstance(linguistic_index, bool)
                and linguistic_index > 0
            ):
                relation[quantity_field] = linguistic_index
                record(
                    f"process_relations[{index}].{quantity_field}",
                    "restore_linguistic_quantity_index",
                    "linguistic_frame",
                )
            if relation.get(quantity_field) is None:
                inferred_indices = [
                    quantity_index
                    for quantity_index in range(1, 13)
                    if _source_identifies_quantity(
                        surface,
                        None,
                        quantity_index,
                    )
                ]
                if len(inferred_indices) == 1:
                    relation[quantity_field] = inferred_indices[0]
                    record(
                        f"process_relations[{index}].{quantity_field}",
                        "recover_explicit_ordinal",
                        "literal_relation_source",
                    )
            if (
                surface
                and _literal_span_in_sources(surface, literal_sources)
                and _folded_text(surface) in _folded_text(
                    relation.get("source_text")
                )
            ):
                relation[f"{prefix}_surface_text"] = surface
    if _bind_authoritative_process_relation_instances(
        repaired.get("process_relations") or [],
        instance_directory,
    ):
        record(
            "process_relations",
            "bind_authoritative_process_instances",
            "literal_relation_source+instance_directory",
        )
    rebound_edits = _bind_authoritative_edit_operation_instances(
        repaired,
        instance_directory,
        source_messages=natural_source_messages,
    )
    if rebound_edits.get("edit_operations") != repaired.get("edit_operations"):
        repaired = rebound_edits
        record(
            "edit_operations",
            "bind_authoritative_edit_instance",
            "literal_target_surface+declared_labels+instance_directory",
        )
    return repaired, repairs


def _bind_linguistic_edit_operations(
    operations,
    rag_by_surface,
    *,
    existing_operations=None,
    instance_directory=None,
):
    existing_operations = [
        item
        for item in (existing_operations or [])
        if isinstance(item, dict)
    ]
    bound = []
    for index, item in enumerate(operations):
        operation = copy.deepcopy(item)
        operation["op_type"] = str(
            operation.get("op_type")
            or operation.get("operation")
            or operation.get("operation_type")
            or ""
        ).casefold()
        if operation["op_type"] in {
            "translate",
            "translation",
            "shift",
        }:
            operation["op_type"] = "move"
        if operation["op_type"]=="move" and operation.get("dy_m") is not None and operation.get("dz_m") is None:
            literal=str(operation.get("source_text") or "")
            cardinal=list(re.finditer(r"(?<![\d.])(\d+(?:\.\d+)?)\s*m(?:eters?|etres?)?\s+(north|south)\b",literal,re.I))
            if len(cardinal)==1 and not re.search(r"\b(?:vertically|up|down|raise|lower|height)\b",literal,re.I):
                operation["dz_m"]=float(cardinal[0].group(1))*(1 if cardinal[0].group(2).lower()=="north" else -1)
                operation["dy_m"]=None
        for legacy_field, canonical_field in (
            ("delta_x", "dx_m"),
            ("delta_y", "dy_m"),
            ("delta_z", "dz_m"),
        ):
            if (
                operation.get(canonical_field) is None
                and operation.get(legacy_field) is not None
            ):
                operation[canonical_field] = operation[legacy_field]
        if operation["op_type"] == "move" and any(
            operation.get(field) is not None
            for field in ("dx_m", "dy_m", "dz_m")
        ):
            operation["move_mode"] = "relative"
        raw_target_surface = str(
            operation.pop("target_surface_text", None)
            or operation.get("surface_text")
            or ""
        ).strip()
        directory_match = None
        directory_ambiguous = False
        if operation["op_type"] != "add":
            directory_match, directory_alias, directory_ambiguous = (
                _edit_target_instance_alias(raw_target_surface, instance_directory)
            )
            if directory_alias:
                target_surface = directory_alias
            elif directory_ambiguous:
                # Several existing roles occur in this target field. Keep the
                # literal unresolved instead of letting a replacement RAG hit
                # stand in for the old object.
                target_surface = raw_target_surface
            else:
                target_surface = _resolved_edit_target_surface(
                    raw_target_surface,
                    operation.get("source_text"),
                    rag_by_surface,
                )
        else:
            target_surface = _resolved_edit_target_surface(
                raw_target_surface,
                operation.get("source_text"),
                rag_by_surface,
            )
        new_surface = str(
            operation.pop("new_asset_surface_text", None) or ""
        ).strip()
        if new_surface:
            new_surface = _resolved_edit_target_surface(
                new_surface, new_surface, rag_by_surface
            )
        target_match = (
            None
            if directory_ambiguous
            else rag_by_surface.get(_folded_text(target_surface))
        )
        new_match = rag_by_surface.get(_folded_text(new_surface))
        existing = (
            existing_operations[index]
            if index < len(existing_operations)
            else {}
        )
        existing_target_surface = str(
            existing.get("target_surface_text") or existing.get("surface_text") or ""
        ).strip()
        existing_matches_target = bool(
            existing_target_surface
            and _folded_text(existing_target_surface)
            == _folded_text(target_surface)
        )
        for field in ("new_model_asset_id", "new_library_id"):
            if not operation.get(field) and existing.get(field):
                operation[field] = existing[field]
        if not directory_match and existing_matches_target:
            for field in ("instance_id", "target_model_asset_id"):
                if not operation.get(field) and existing.get(field):
                    operation[field] = existing[field]
        if target_match and _has_resolved_rag_selection(target_match):
            operation["target_model_asset_id"] = target_match.get(
                "selected_model_asset_id"
            )
        if directory_match:
            # Existing role identity in the active session is authoritative
            # for the old edit target. It must not be replaced by the RAG hit
            # for a newly introduced replacement asset.
            operation["instance_id"] = directory_match["instance_id"]
            operation["target_model_asset_id"] = directory_match["model_asset_id"]
        if new_match:
            operation["new_model_asset_id"] = new_match.get("selected_model_asset_id") if _has_resolved_rag_selection(new_match) else None
        # Keep the operation's old-target span from the linguistic extraction.
        # The new asset has its own slot and may not replace this role label.
        operation["surface_text"] = target_surface or new_surface
        bound.append(operation)
    return bound


def _edit_target_instance_alias(target_surface, instance_directory):
    """Return one active instance whose stable role alias is in the target."""

    target_key = re.sub(
        r"^(?:a|an|the)\s+", "", _folded_text(target_surface)
    )
    if not target_key:
        return None, None, False
    directory = _normalize_instance_directory(instance_directory or [])
    exact = []
    for item in directory:
        aliases = item["stable_alias"]
        alias_values = aliases.values() if isinstance(aliases, dict) else [aliases]
        for alias in alias_values:
            alias_key = re.sub(r"^(?:a|an|the)\s+", "", _folded_text(alias))
            if alias_key == target_key:
                exact.append((item, str(alias).strip()))
    if len(exact) == 1:
        return exact[0][0], exact[0][1], False
    if len(exact) > 1:
        return None, None, True

    contained = []
    for item in directory:
        aliases = item["stable_alias"]
        alias_values = aliases.values() if isinstance(aliases, dict) else [aliases]
        for alias in alias_values:
            alias_text = str(alias or "").strip()
            alias_key = re.sub(r"^(?:a|an|the)\s+", "", _folded_text(alias_text))
            if alias_key and re.search(
                rf"(?<![a-z0-9]){re.escape(alias_key)}(?![a-z0-9])",
                target_key,
            ):
                contained.append((item, alias_text, len(alias_key)))
    longest = max((length for _item, _alias, length in contained), default=0)
    best = [entry for entry in contained if entry[2] == longest]
    instance_ids = {entry[0]["instance_id"] for entry in best}
    if len(best) == 1 and len(instance_ids) == 1:
        return best[0][0], best[0][1], False
    if best:
        return None, None, True
    return None, None, False


def _resolved_edit_target_surface(target_surface, source_text, rag_by_surface):
    """Recover one resolved literal equipment span from an edit action phrase.

    Linguistic models occasionally put the entire imperative (for example,
    ``move CNC machine B 0.5 m east``) in ``surface_text``. The target is
    grounded when exactly one resolved RAG surface occurs within that target
    field. The full operation source is deliberately excluded: a replacement
    description elsewhere in that source is a new asset, not evidence for the
    old edit target.
    """

    direct = rag_by_surface.get(_folded_text(target_surface))
    if _has_resolved_rag_selection(direct):
        return str(direct.get("surface_text") or target_surface).strip()
    # A labeled shorthand may omit a type qualifier ("cabinet B" versus
    # "tool cabinet B"). Expand only a unique actually grounded full name.
    short_name = _folded_text(target_surface)
    if re.search(r"\s(?:[a-z]|\d+)$", short_name):
        expanded = {
            str(match.get("surface_text") or "").strip()
            for match in rag_by_surface.values()
            if _has_resolved_rag_selection(match)
            and _folded_text(match.get("surface_text")).endswith(" " + short_name)
        }
        if len(expanded) == 1:
            return next(iter(expanded))
    haystacks = [_folded_text(target_surface)]
    candidates = []
    for match in rag_by_surface.values():
        if not _has_resolved_rag_selection(match):
            continue
        surface = str(match.get("surface_text") or "").strip()
        folded_surface = _folded_text(surface)
        if not folded_surface or not any(
            folded_surface in haystack for haystack in haystacks if haystack
        ):
            continue
        candidates.append(surface)
    longest = max((len(_folded_text(surface)) for surface in candidates), default=0)
    longest_candidates = {
        surface
        for surface in candidates
        if len(_folded_text(surface)) == longest
    }
    return next(iter(longest_candidates)) if len(longest_candidates) == 1 else target_surface


def _recover_explicit_move_deltas(operations, user_message):
    """Recover a literal relative move that the semantic model omitted.

    This is deliberately narrow: it requires both an explicit metric distance
    and an unambiguous axis/direction phrase in the current user turn.  Vague
    requests such as "move it right" remain incomplete and still clarify.
    """

    text = str(user_message or "")
    distance_matches = list(
        re.finditer(
            r"(?<![\d.])(?P<distance>\d+(?:\.\d+)?)\s*"
            r"(?:m(?:eter|etre)?s?|米)(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
    )
    if len(distance_matches) != 1:
        return False
    distance = float(distance_matches[0].group("distance"))
    folded = _folded_text(text)
    directions = []
    if (
        re.search(r"(?:positive|\+)\s*x(?:\s*(?:axis|direction))?", folded)
        or re.search(r"\b(?:to\s+the\s+)?right\b", folded)
        or "向右" in text
        or "右移" in text
    ):
        directions.append(("dx_m", distance))
    if (
        re.search(r"(?:negative|-)\s*x(?:\s*(?:axis|direction))?", folded)
        or re.search(r"\b(?:to\s+the\s+)?left\b", folded)
        or "向左" in text
        or "左移" in text
    ):
        directions.append(("dx_m", -distance))
    if (
        re.search(r"(?:positive|\+)\s*z(?:\s*(?:axis|direction))?", folded)
        or "向正z" in folded
    ):
        directions.append(("dz_m", distance))
    if (
        re.search(r"(?:negative|-)\s*z(?:\s*(?:axis|direction))?", folded)
        or "向负z" in folded
    ):
        directions.append(("dz_m", -distance))
    if len(directions) != 1:
        return False

    explicit_absolute_cue = bool(
        re.search(
            r"\b(?:to|at)\s+(?:the\s+)?(?:position|coordinate)\b"
            r"|(?:^|[^\w])(?:x|y|z)\s*=",
            folded,
        )
    )
    delta_to_absolute = {
        "dx_m": "x_m",
        "dy_m": "y_m",
        "dz_m": "z_m",
    }
    direction_field, direction_value = directions[0]
    direction_absolute_field = delta_to_absolute[direction_field]
    eligible = []
    for operation in operations:
        if not isinstance(operation, dict):
            continue
        operation_type = str(operation.get("op_type") or "").casefold()
        if operation_type not in {
            "move",
            "translate",
            "translation",
            "shift",
        }:
            continue
        absolute_fields = {
            field: operation.get(field)
            for field in ("x_m", "y_m", "z_m")
            if operation.get(field) is not None
        }
        delta_fields = {
            field: operation.get(field)
            for field in ("dx_m", "dy_m", "dz_m")
            if operation.get(field) is not None
        }
        try:
            has_nonzero_delta = any(
                abs(float(value)) > 1e-9 for value in delta_fields.values()
            )
        except (TypeError, ValueError):
            has_nonzero_delta = True
        if has_nonzero_delta:
            continue
        if absolute_fields:
            # Some frozen semantic frames encoded "translate 0.5 m in the
            # positive x direction" as the absolute coordinate x=0.5.  The
            # source turn is authoritative: only reinterpret that narrow
            # shape when no absolute-coordinate wording is present.
            if explicit_absolute_cue or set(absolute_fields) != {
                direction_absolute_field
            }:
                continue
            try:
                encoded_distance = abs(
                    float(absolute_fields[direction_absolute_field])
                )
            except (TypeError, ValueError):
                continue
            if abs(encoded_distance - distance) > 1e-9:
                continue
        eligible.append(operation)
    if len(eligible) != 1:
        return False

    operation = eligible[0]
    for absolute_field in ("x_m", "y_m", "z_m"):
        operation[absolute_field] = None
    operation["op_type"] = "move"
    operation["move_mode"] = "relative"
    operation[direction_field] = direction_value
    return True


def _bind_linguistic_process_relations(relations, rag_by_surface):
    bound = []
    for item in relations:
        relation = copy.deepcopy(item)
        for prefix in ("from", "to", "carrier"):
            surface = str(
                relation.get(f"{prefix}_surface_text") or ""
            ).strip()
            role = facility_role(surface)
            if role:
                relation[f"{prefix}_site_feature_name"] = role
                relation.pop(f"{prefix}_model_asset_id", None)
                continue
            match = rag_by_surface.get(_folded_text(surface))
            if match:
                selected_model_asset_id = str(
                    match.get("selected_model_asset_id") or ""
                ).strip()
                if selected_model_asset_id:
                    relation[f"{prefix}_model_asset_id"] = (
                        selected_model_asset_id
                    )
                quantity_index = match.get("quantity_index")
                if (
                    isinstance(quantity_index, int)
                    and not isinstance(quantity_index, bool)
                    and quantity_index > 0
                ):
                    relation[f"{prefix}_quantity_index"] = quantity_index
        bound.append(relation)
    return bound


def _explicit_arrow_chain_relations(user_message):
    """Parse only literal arrow chains that already state an ordered route."""

    relations = []
    source_message = str(user_message or "")
    if "\nCurrent user request:\n" in source_message:
        source_message = source_message.rsplit("\nCurrent user request:\n",1)[-1]
    for raw_line in source_message.splitlines():
        if "->" not in raw_line and "→" not in raw_line:
            continue
        source_text = re.sub(r"^\s*[-*]\s*", "", raw_line).strip()
        header, delimiter, chain = source_text.partition(":")
        chain = (chain if delimiter else source_text).strip().rstrip(".。")
        endpoints = [
            endpoint.strip()
            for endpoint in re.split(r"\s*(?:->|→)\s*", chain)
            if endpoint.strip()
        ]
        if len(endpoints) < 2:
            continue
        weight = _explicit_arrow_route_weight(source_text)
        for sequence_order, (left, right) in enumerate(
            zip(endpoints, endpoints[1:]),
            start=1,
        ):
            relation = {
                "from_surface_text": left,
                "to_surface_text": right,
                "source_text": source_text,
                "product": header.strip() if delimiter else "",
                "flow_type": "tool_service" if re.search(r"\btool\s+service\b",header,re.I) else "material",
                "sequence_order": sequence_order,
                "required_connection": True,
            }
            if weight is not None:
                relation["flow_weight"] = weight
            relations.append(relation)
    return relations


def _explicit_arrow_route_weight(source_text):
    match = re.search(
        r"(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?:(?:batches?|trips?)\s*(?:per|/)\s*day|(?:batches?|trips?)\s+daily)\b",
        str(source_text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = float(match.group("value"))
    return int(value) if value.is_integer() else value


def _context_carrier_rules(engineering_context):
    """Read carrier assignments as literal source rules, without asset IDs."""

    rules = []
    for raw_line in str(engineering_context or "").splitlines():
        line = re.sub(r"^\s*[-*]\s*", "", raw_line).strip()
        if not line:
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line)
        for sentence in sentences:
            sentence = sentence.strip()
            agv = re.search(r"\b(AGV\s+[A-Za-z0-9]+)\s+carries\b", sentence, re.I)
            conveyor = re.search(
                r"\b(?P<resource>(?:(?:the|a|an)\s+)?"
                r"(?:[A-Za-z0-9-]+\s+){0,2}conveyor(?:\s+[A-Za-z0-9]+)?)\s+carries\b",
                sentence,
                re.I,
            )
            operator = re.search(r"\b(Operators?)\s+(?:handle|perform)\b", sentence, re.I)
            if agv:
                mode, carrier = "agv", agv.group(1).strip()
            elif conveyor:
                mode = "conveyor"
                carrier = re.sub(r"^(?:the|a|an)\s+", "", conveyor.group("resource"), flags=re.I).strip()
            elif operator:
                mode, carrier = "operator", operator.group(1).strip()
            else:
                continue

            if re.search(r"\btool[- ]service round trips\b", sentence, re.I):
                rules.append(
                    {
                        "mode": "operator",
                        "carrier_surface_text": carrier,
                        "carrier_source_text": sentence,
                        "flow_type": "tool_service",
                    }
                )
                continue

            if operator and re.search(r"\bhandle\b", sentence, re.I):
                body = re.split(r"\bhandle\b", sentence, maxsplit=1, flags=re.I)[1]
                clauses = re.split(r",\s*(?:and\s+)?|\s+and\s+", body.rstrip(".。"))
                for clause in clauses:
                    pair = re.match(
                        r"\s*(?P<left>.+?)\s+to\s+(?P<right>.+?)\s*$",
                        clause,
                        flags=re.I,
                    )
                    if pair:
                        rules.append(
                            {
                                "mode": mode,
                                "carrier_surface_text": carrier,
                                "carrier_source_text": sentence,
                                "from_phrase": pair.group("left").strip(),
                                "to_phrase": pair.group("right").strip(),
                            }
                        )

            for pair in re.finditer(
                r"\bfrom\s+(?P<left>[^,;.]+?)\s+to\s+(?P<right>[^,;.]+?)"
                r"(?=(?:,\s*(?:and\s+)?(?:from|between)\b)|[.;]|$)",
                sentence,
                flags=re.I,
            ):
                rules.append(
                    {
                        "mode": mode,
                        "carrier_surface_text": carrier,
                        "carrier_source_text": sentence,
                        "from_phrase": pair.group("left").strip(),
                        "to_phrase": pair.group("right").strip(),
                    }
                )

            successive = re.search(
                r"\bbetween\s+(?:the\s+)?successive\s+(?P<kind>[^,;.]+)",
                sentence,
                flags=re.I,
            )
            if successive:
                rules.append(
                    {
                        "mode": mode,
                        "carrier_surface_text": carrier,
                        "carrier_source_text": sentence,
                        "from_phrase": successive.group("kind").strip(),
                        "to_phrase": successive.group("kind").strip(),
                        "both_endpoints": True,
                    }
                )

            for pair in re.finditer(
                r"\bbetween\s+(?:the\s+)?(?P<left>[^,;.]+?)\s+and\s+"
                r"(?P<right>[^,;.]+?)(?=[.;]|$)",
                sentence,
                flags=re.I,
            ):
                rules.append(
                    {
                        "mode": mode,
                        "carrier_surface_text": carrier,
                        "carrier_source_text": sentence,
                        "from_phrase": pair.group("left").strip(),
                        "to_phrase": pair.group("right").strip(),
                        "between_path": True,
                    }
                )
    return rules


def _context_carrier_endpoint_kind(surface, machining_declarations):
    text = _folded_text(surface)
    facility = facility_role(surface)
    if facility:
        return facility.replace(" ", "_")
    tokens = set(_process_surface_tokens(text))
    if "raw" in tokens and "material" in tokens:
        return "raw_material_storage"
    if "finished" in tokens and "goods" in tokens:
        return "finished_goods_storage"
    if "line" in tokens and "side" in tokens:
        return "line_side_storage"
    if "loading" in tokens and (
        "table" in tokens or "end" in tokens or tokens == {"loading"}
    ):
        return "loading_table"
    if "unloading" in tokens and (
        "table" in tokens or "end" in tokens or tokens == {"unloading"}
    ):
        return "unloading_table"
    if "conveyor" in tokens:
        return "conveyor"
    if "temperature" in tokens and "chamber" in tokens:
        return "temperature_chamber"
    if "laser" in tokens or "marking" in tokens:
        return "laser_marking"
    if "fixture" in tokens or "assembly" in tokens:
        return "fixture_workstation"
    if "test" in tokens or "testing" in tokens:
        return "test_workstation"
    if tokens & {"storage", "rack"}:
        return "storage"
    if any(_declared_surface_matches(surface, item) for item in machining_declarations):
        return "machining_station"
    if tokens & {"machining", "machine", "mill", "drilling", "drill", "cnc", "edm"}:
        return "machining_station"
    return " ".join(sorted(tokens))


def _context_carrier_kind_matches(expected, actual):
    if expected == "storage":
        return actual in {
            "storage",
            "raw_material_storage",
            "finished_goods_storage",
            "line_side_storage",
        }
    return bool(expected) and expected == actual


def _apply_context_carrier_assignments(relations, engineering_context):
    """Attach only carrier assignments explicitly supported by source text."""

    result = copy.deepcopy(relations or [])
    rules = _context_carrier_rules(engineering_context)
    if not result or not rules:
        return result
    inventory = engineering_inventory_items(engineering_context)
    machining_declarations = [
        item
        for item in inventory
        if re.search(r"\b(?:machining|machine|mill|drilling|drill|CNC|EDM)\b", item, re.I)
    ]
    kinds = {
        id(relation): (
            _context_carrier_endpoint_kind(
                relation.get("from_surface_text"), machining_declarations
            ),
            _context_carrier_endpoint_kind(
                relation.get("to_surface_text"), machining_declarations
            ),
        )
        for relation in result
    }
    assignments = {id(relation): [] for relation in result}

    for rule in rules:
        if rule.get("flow_type") == "tool_service":
            for relation in result:
                if relation.get("flow_type") == "tool_service":
                    assignments[id(relation)].append(rule)
            continue
        expected_from = _context_carrier_endpoint_kind(
            rule.get("from_phrase"), machining_declarations
        )
        expected_to = _context_carrier_endpoint_kind(
            rule.get("to_phrase"), machining_declarations
        )
        if rule.get("both_endpoints"):
            for relation in result:
                actual_from, actual_to = kinds[id(relation)]
                if (
                    _context_carrier_kind_matches(expected_from, actual_from)
                    and _context_carrier_kind_matches(expected_to, actual_to)
                ):
                    assignments[id(relation)].append(rule)
            continue
        if rule.get("between_path"):
            grouped = {}
            for relation in result:
                grouped.setdefault(str(relation.get("source_text") or ""), []).append(relation)
            for group in grouped.values():
                group.sort(key=lambda relation: int(relation.get("sequence_order") or 0))
                if not group:
                    continue
                nodes = [group[0].get("from_surface_text")]
                nodes.extend(relation.get("to_surface_text") for relation in group)
                node_kinds = [
                    _context_carrier_endpoint_kind(node, machining_declarations)
                    for node in nodes
                ]
                for start_index, start_kind in enumerate(node_kinds):
                    if not _context_carrier_kind_matches(expected_from, start_kind):
                        continue
                    for end_index in range(start_index + 1, len(node_kinds)):
                        if not _context_carrier_kind_matches(expected_to, node_kinds[end_index]):
                            continue
                        for relation in group[start_index:end_index]:
                            assignments[id(relation)].append(rule)
                        break
            continue
        for relation in result:
            actual_from, actual_to = kinds[id(relation)]
            if (
                _context_carrier_kind_matches(expected_from, actual_from)
                and _context_carrier_kind_matches(expected_to, actual_to)
            ):
                assignments[id(relation)].append(rule)

    for relation in result:
        matches = assignments[id(relation)]
        signatures = {
            (
                item.get("mode"),
                _folded_text(item.get("carrier_surface_text")),
                item.get("carrier_source_text"),
            )
            for item in matches
        }
        if len(signatures) != 1:
            continue
        mode, _carrier_key, source_text = next(iter(signatures))
        selected = next(
            item
            for item in matches
            if item.get("mode") == mode
            and _folded_text(item.get("carrier_surface_text")) == _carrier_key
            and item.get("carrier_source_text") == source_text
        )
        relation.update(
            transport_mode=selected["mode"],
            carrier_surface_text=selected["carrier_surface_text"],
            carrier_source_text=selected["carrier_source_text"],
        )
    return result


def _raw_declaration_asset_mentions(asset_mentions, user_message, rag_by_surface):
    declarations = []
    for mention in asset_mentions or []:
        if not isinstance(mention, dict):
            continue
        surface_text = str(mention.get("surface_text") or "").strip()
        match = rag_by_surface.get(_folded_text(surface_text))
        selected_model_asset_id = str(
            (match or {}).get("selected_model_asset_id") or ""
        ).strip()
        if not surface_text or not selected_model_asset_id:
            continue
        declaration = copy.deepcopy(mention)
        declaration["surface_text"] = _raw_declaration_literal(
            surface_text,
            user_message,
        )
        declaration["_selected_model_asset_id"] = selected_model_asset_id
        declarations.append(declaration)
    return declarations


def _raw_declaration_literal(surface_text, user_message):
    text = str(user_message or "")
    match = re.search(re.escape(str(surface_text)), text, flags=re.IGNORECASE)
    if not match:
        return str(surface_text).strip()
    quantity = _leading_quantity_from_surface(surface_text) or 1
    literal = text[match.start():match.end()]
    if quantity==1 or declaration_labels(surface_text,quantity):
        return literal
    # Recover only an immediately following label list, never neighboring equipment.
    tail = text[match.end():]
    pattern = r"^\s+(?:(?:labeled|labelled)\s+)?([A-Za-z](?:\s*(?:,|\band\b|\bor\b)\s*[A-Za-z]){"+str(quantity-1)+r"})(?=\s*[,.;]|$)"
    labels = re.match(pattern,tail,flags=re.I)
    if labels:
        remainder = tail[labels.end():]
        extra = re.match(r"^\s*(?:,|and|or)\s*([A-Za-z0-9_-]+)",remainder,flags=re.I)
        if not extra or extra.group(1).casefold() in {"a","an","one","two","three","four","five","six","seven","eight","nine","ten"}:
            return literal+tail[:labels.end()]
    return literal


def _same_transport_task(left,right):
    """A carrier citation can differ from the clause declaring the route."""
    for prefix in ("from","to"):
        lid,rid=left.get(prefix+"_instance_id"),right.get(prefix+"_instance_id")
        if lid and rid:
            if lid!=rid:return False
        elif left.get(prefix+"_site_feature_name") or right.get(prefix+"_site_feature_name"):
            if left.get(prefix+"_site_feature_name")!=right.get(prefix+"_site_feature_name"):return False
        elif not left.get(prefix+"_model_asset_id") or (left.get(prefix+"_model_asset_id"),left.get(prefix+"_quantity_index"))!=(right.get(prefix+"_model_asset_id"),right.get(prefix+"_quantity_index")):
            return False
    a,b=left.get("flow_weight"),right.get("flow_weight")
    if a is not None and b is not None and float(a)!=float(b):return False
    a,b=(str(r.get("product") or "").split(",",1)[0].strip().casefold() for r in (left,right))
    return not a or not b or a==b


def _fully_bound_arrow_relations(
    relations,
    asset_mentions,
    rag_by_surface,
    user_message,
    literal_sources,
    resource_relations=None,
    instance_directory=None,
):
    if not relations:
        return None

    carrier_fields = (
        "transport_mode",
        "carrier_model_asset_id",
        "carrier_quantity_index",
        "carrier_surface_text",
        "carrier_source_text",
    )

    def merge_resource_carrier_evidence(bound_rows, resource_rows):
        for row in bound_rows:
            if (
                row.get("transport_mode") in {"operator", "agv", "conveyor"}
                and _literal_span_in_sources(
                    row.get("carrier_source_text"), literal_sources
                )
            ):
                # Stable source rules were matched to this exact route before
                # model output was reconciled. A partial or conflicting LLM
                # paraphrase cannot replace that assignment.
                continue
            matches = [
                item
                for item in resource_rows
                if _relation_views_match(row, item)
                or _same_transport_task(row, item)
            ]
            grounded = [
                item
                for item in matches
                if item.get("transport_mode") in {"operator", "agv", "conveyor"}
                and _literal_span_in_sources(
                    item.get("carrier_source_text"), literal_sources
                )
            ]
            signatures = {
                tuple(item.get(field) for field in carrier_fields)
                for item in grounded
            }
            if len(signatures) == 1:
                row.update(
                    {
                        field: grounded[0].get(field)
                        for field in carrier_fields
                    }
                )

    directory = _normalize_instance_directory(instance_directory or [])
    if directory:
        # Existing role names refer to already selected physical instances.
        # A new generic catalog hit must not replace their identity.
        relation_grounding = dict(rag_by_surface)
        for item in directory:
            aliases = item["stable_alias"].values() if isinstance(item["stable_alias"],dict) else [item["stable_alias"]]
            for alias in aliases:
                relation_grounding[_folded_text(alias)] = dict(selected_model_asset_id=item["model_asset_id"],quantity_index=item["quantity_index"])
        bound = _bind_linguistic_process_relations(relations,relation_grounding)
        resource_bound = _bind_linguistic_process_relations(resource_relations or [],relation_grounding)
        _bind_authoritative_process_relation_instances(bound,directory)
        _bind_authoritative_process_relation_instances(resource_bound,directory)
        merge_resource_carrier_evidence(bound, resource_bound)
        for row in bound:
            if not _literal_span_in_sources(row.get("source_text"),literal_sources):return None
            if any(not row.get(prefix+"_site_feature_name") and not row.get(prefix+"_instance_id") for prefix in ("from","to")):return None
        return bound,relation_grounding
    declarations = _raw_declaration_asset_mentions(
        asset_mentions,
        user_message,
        rag_by_surface,
    )
    if not declarations:
        return None
    relation_grounding = dict(rag_by_surface)
    relation_grounding.update(
        _declared_process_endpoint_bindings(
            [*relations, *(resource_relations or [])],
            declarations,
            rag_by_surface,
        )
    )
    bound = _bind_linguistic_process_relations(relations, relation_grounding)
    resource_bound = _bind_linguistic_process_relations(resource_relations or [], relation_grounding)
    merge_resource_carrier_evidence(bound, resource_bound)
    quantities = {}
    for item in declarations:
        mid = item.get("_selected_model_asset_id")
        if mid:
            quantities[mid] = quantities.get(mid, 0) + max(1, int(item.get("quantity") or 1))
    for relation in bound:
        if not _literal_span_in_sources(
            relation.get("source_text"),
            literal_sources,
        ):
            return None
        for prefix in ("from", "to"):
            if relation.get(f"{prefix}_site_feature_name"):
                continue
            model_asset_id = str(
                relation.get(f"{prefix}_model_asset_id") or ""
            ).strip()
            quantity = quantities.get(model_asset_id)
            quantity_index = relation.get(f"{prefix}_quantity_index")
            if not model_asset_id or quantity is None:
                return None
            if quantity > 1 and quantity_index is None:
                return None
            if quantity_index is not None and (
                isinstance(quantity_index, bool)
                or not isinstance(quantity_index, int)
                or quantity_index < 1
                or quantity_index > quantity
            ):
                return None
    return bound, relation_grounding


def _declared_process_endpoint_bindings(
    relations,
    asset_mentions,
    rag_by_surface,
):
    """Resolve literal route endpoints to one declared, RAG-selected asset."""

    declarations = []
    quantity_cursor = {}
    for mention in asset_mentions:
        if not isinstance(mention, dict):
            continue
        surface_text = str(mention.get("surface_text") or "").strip()
        match = rag_by_surface.get(_folded_text(surface_text))
        selected_model_asset_id = str(
            mention.get("_selected_model_asset_id")
            or (match or {}).get("selected_model_asset_id")
            or ""
        ).strip()
        quantity = int(_optional_float(mention.get("quantity")) or 1)
        if not surface_text or not selected_model_asset_id or quantity < 1:
            continue
        start_index = quantity_cursor.get(selected_model_asset_id, 0) + 1
        quantity_cursor[selected_model_asset_id] = start_index + quantity - 1
        declarations.append(
            {
                "surface_text": surface_text,
                "surface_key": _folded_text(surface_text),
                "model_asset_id": selected_model_asset_id,
                "quantity": quantity,
                "quantity_start": start_index,
                "labels": _explicit_declared_labels(surface_text, quantity),
            }
        )

    bindings = {}
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        for prefix in ("from", "to", "carrier"):
            surface_text = str(
                relation.get(f"{prefix}_surface_text") or ""
            ).strip()
            surface_key = _folded_text(surface_text)
            if not surface_key or surface_key in bindings:
                continue
            binding = _resolve_declared_process_endpoint(
                surface_text,
                declarations,
            )
            if binding:
                bindings[surface_key] = binding
    return bindings


def _resolve_declared_process_endpoint(surface_text, declarations):
    matches = [
        declaration
        for declaration in declarations
        if _declared_surface_matches(surface_text, declaration["surface_text"])
    ]
    if len(matches) != 1:
        return None
    declaration = matches[0]
    binding = {"selected_model_asset_id": declaration["model_asset_id"]}
    if declaration["quantity"] == 1:
        binding["quantity_index"] = declaration.get("quantity_start", 1)
    else:
        quantity_index = _declared_label_quantity_index(
            surface_text,
            declaration,
        )
        if quantity_index is not None:
            binding["quantity_index"] = declaration.get("quantity_start", 1) + quantity_index - 1
    return binding


def _declared_surface_matches(endpoint_surface, declaration_surface):
    endpoint_key = _folded_text(endpoint_surface)
    declaration_key = _folded_text(declaration_surface)
    if not endpoint_key or not declaration_key:
        return False
    if endpoint_key == declaration_key:
        return True
    endpoint_base, _endpoint_ordinal = _explicit_endpoint_ordinal(endpoint_key)
    declaration_base, _declaration_ordinal = _explicit_endpoint_ordinal(
        declaration_key
    )
    if endpoint_base == declaration_base:
        return True
    # Chinese equipment names must match exactly after removing only an
    # explicit instance ordinal. Do not use token overlap for CJK text.
    if _contains_cjk(endpoint_key) or _contains_cjk(declaration_key):
        return False
    endpoint_tokens = _process_surface_tokens(endpoint_key)
    declaration_tokens = _process_surface_tokens(declaration_key)
    return bool(endpoint_tokens and endpoint_tokens <= declaration_tokens)


def _explicit_endpoint_ordinal(surface_text):
    """Return a literal instance base name and ordinal without fuzzy matching."""

    value = _folded_text(surface_text)
    if not value:
        return value, None
    patterns = (
        r"^第\s*(?P<ordinal>\d+)\s*(?:台|个|套|号)\s*(?P<base>.+)$",
        r"^(?P<ordinal>\d+)\s*号\s*(?P<base>.+)$",
        r"^(?P<base>.+?)\s*第\s*(?P<ordinal>\d+)\s*(?:台|个|套|号)$",
        r"^(?P<base>.+?)\s*(?P<ordinal>\d+)\s*号$",
        r"^no\.?\s*(?P<ordinal>\d+)\s+(?P<base>.+)$",
        r"^(?P<base>.+?)\s+no\.?\s*(?P<ordinal>\d+)$",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value, flags=re.IGNORECASE)
        if not match:
            continue
        ordinal = int(match.group("ordinal"))
        base = _folded_text(match.group("base"))
        if ordinal > 0 and base:
            return base, ordinal
    return value, None


def _contains_cjk(value):
    return bool(re.search(r"[\u3400-\u9fff]", str(value or "")))


def _process_surface_tokens(value):
    tokens = set()
    for token in re.findall(r"[a-z0-9]+", _folded_text(value)):
        if len(token) > 3 and token.endswith("s"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _explicit_declared_labels(surface_text, quantity):
    return declaration_labels(surface_text, quantity)


def _declared_label_quantity_index(endpoint_surface, declaration):
    endpoint_base, explicit_ordinal = _explicit_endpoint_ordinal(endpoint_surface)
    declaration_base, _declaration_ordinal = _explicit_endpoint_ordinal(
        declaration.get("surface_text")
    )
    if explicit_ordinal is not None:
        if (
            endpoint_base == declaration_base
            and 1 <= explicit_ordinal <= declaration.get("quantity", 0)
        ):
            return explicit_ordinal
        return None
    labels = declaration.get("labels") or ()
    if not labels:
        return None
    endpoint_tokens = re.findall(
        r"[a-z0-9]+",
        _folded_text(endpoint_surface),
    )
    if not endpoint_tokens:
        return None
    terminal_label = endpoint_tokens[-1]
    if terminal_label not in labels:
        return None
    return labels.index(terminal_label) + 1


def _source_identifies_declared_quantity_label(
    source_text,
    endpoint_surface_text,
    asset_mentions,
    model_asset_id,
    quantity_index,
):
    """Prove an A/B-style endpoint from its literal declaration and route span."""

    source_key = _folded_text(source_text)
    endpoint_key = _folded_text(endpoint_surface_text)
    if not source_key or not endpoint_key or endpoint_key not in source_key:
        return False
    matches = []
    quantity_cursor = 0
    for mention in asset_mentions or []:
        if not isinstance(mention, dict):
            continue
        if str(mention.get("model_asset_id") or "") != str(model_asset_id):
            continue
        quantity = int(_optional_float(mention.get("quantity")) or 1)
        start_index = quantity_cursor + 1
        quantity_cursor += quantity
        declaration = {
            "surface_text": str(mention.get("surface_text") or "").strip(),
            "quantity": quantity,
            "labels": _explicit_declared_labels(
                mention.get("surface_text"),
                quantity,
            ),
        }
        if not _declared_surface_matches(
            endpoint_surface_text,
            declaration["surface_text"],
        ):
            continue
        local_index = 1 if quantity == 1 else _declared_label_quantity_index(endpoint_surface_text, declaration)
        if local_index is not None and start_index + local_index - 1 == quantity_index:
            matches.append(declaration)
    return len(matches) == 1


def _process_relation_identity(relation):
    """Keep endpoint identity with route evidence so route edges remain distinct."""

    return (
        str(relation.get("from_model_asset_id") or relation.get("from_site_feature_name") or ""),
        relation.get("from_quantity_index"),
        str(relation.get("to_model_asset_id") or relation.get("to_site_feature_name") or ""),
        relation.get("to_quantity_index"),
        _folded_text(relation.get("source_text")),
        relation.get("sequence_order"),
        relation.get("flow_weight"),
        str(relation.get("flow_type") or "material"),
    )


def _relation_views_match(semantic_relation, linguistic_relation):
    for prefix in ("from", "to"):
        site_name = linguistic_relation.get(f"{prefix}_site_feature_name")
        if site_name:
            if site_name != semantic_relation.get(f"{prefix}_site_feature_name"):
                return False
            continue
        linguistic_asset_id = str(
            linguistic_relation.get(f"{prefix}_model_asset_id") or ""
        ).strip()
        semantic_asset_id = str(
            semantic_relation.get(f"{prefix}_model_asset_id") or ""
        ).strip()
        if not linguistic_asset_id or linguistic_asset_id != semantic_asset_id:
            return False
        linguistic_index = linguistic_relation.get(f"{prefix}_quantity_index")
        semantic_index = semantic_relation.get(f"{prefix}_quantity_index")
        if (
            linguistic_index is not None
            and semantic_index is not None
            and linguistic_index != semantic_index
        ):
            return False
    linguistic_weight = _linguistic_relation_weight(linguistic_relation)
    semantic_weight = _optional_float(semantic_relation.get("flow_weight"))
    if (
        linguistic_weight is not None
        and semantic_weight is not None
        and linguistic_weight != semantic_weight
    ):
        return False
    linguistic_source = _folded_text(re.sub(r"^\s*[-*]\s*","",str(linguistic_relation.get("source_text") or ""))).rstrip(".。")
    semantic_source = _folded_text(re.sub(r"^\s*[-*]\s*","",str(semantic_relation.get("source_text") or ""))).rstrip(".。")
    if linguistic_source:
        return linguistic_source == semantic_source
    product = _folded_text(linguistic_relation.get("product"))
    return not product or product in semantic_source


def _matching_semantic_relations(linguistic_relation, semantic_relations):
    return [
        relation
        for relation in semantic_relations
        if _relation_views_match(relation, linguistic_relation)
    ]


def _linguistic_relation_weight(relation):
    for field in ("flow_weight", "batches_per_day", "frequency_per_day"):
        if relation.get(field) is not None:
            return _optional_float(relation.get(field))
    return None


def _literal_endpoint_identity(relation):
    return tuple(
        value
        for prefix in ("from", "to")
        for value in (
            _folded_text(relation.get(f"{prefix}_surface_text")),
            str(relation.get(f"{prefix}_model_asset_id") or ""),
            relation.get(f"{prefix}_quantity_index"),
        )
    )


def _unique_matching_linguistic_relation(semantic_relation, linguistic_relations):
    matches = [
        relation
        for relation in linguistic_relations
        if _relation_views_match(semantic_relation, relation)
    ]
    if len(matches) == 1:
        return matches[0]
    # Distinct product routes may repeat the same physical edge. They do not
    # need to be merged to attach endpoint evidence. Attach it only when every
    # matching route record names the same literal endpoints and instances.
    return (
        matches[0]
        if matches
        and len({_literal_endpoint_identity(item) for item in matches}) == 1
        else None
    )


def _can_restore_linguistic_relation(relation, literal_sources):
    source_text = str(relation.get("source_text") or "").strip()
    return bool(
        _literal_span_in_sources(source_text, literal_sources)
        and (relation.get("from_model_asset_id") or relation.get("from_site_feature_name"))
        and (relation.get("to_model_asset_id") or relation.get("to_site_feature_name"))
    )


def _bind_authoritative_process_relation_instances(relations, instance_directory):
    """Bind explicit relation aliases to the authoritative instance directory."""

    directory = _normalize_instance_directory(instance_directory or [])
    changed = False
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        source_text = str(relation.get("source_text") or "")
        header = re.split(r"[:：]", source_text, maxsplit=1)[0]
        parts = re.split(
            r"\s+(?:to|->)\s+|[到至→]",
            header,
            maxsplit=1,
            flags=re.IGNORECASE,
        )
        endpoint_text = {
            "from": parts[0] if parts else "",
            "to": parts[1] if len(parts) == 2 else "",
        }
        for prefix in ("from", "to", "carrier"):
            if relation.get(f"{prefix}_site_feature_name") or facility_role(relation.get(f"{prefix}_surface_text")):
                continue
            surface = _folded_text(relation.get(f"{prefix}_surface_text"))
            exact = [item for item in directory if surface and any(surface==_folded_text(alias) for alias in (item["stable_alias"].values() if isinstance(item["stable_alias"],dict) else [item["stable_alias"]]))]
            model_index = [item for item in directory if item["model_asset_id"]==relation.get(f"{prefix}_model_asset_id") and item["quantity_index"]==relation.get(f"{prefix}_quantity_index")]
            instance_id = str(
                relation.get(f"{prefix}_instance_id") or ""
            ).strip()
            selected = exact[0] if len(exact)==1 else model_index[0] if len(model_index)==1 else next(
                (
                    item
                    for item in directory
                    if item["instance_id"] == instance_id
                ),
                None,
            )
            if selected is None and len(parts)==2 and prefix in endpoint_text:
                segment = _folded_text(endpoint_text[prefix])
                alias_matches = [
                    item
                    for item in directory
                    if any(
                        _folded_text(alias) in segment
                        for alias in (
                            item["stable_alias"].values()
                            if isinstance(item["stable_alias"], dict)
                            else [item["stable_alias"]]
                        )
                    )
                ]
                if len(alias_matches) == 1:
                    selected = alias_matches[0]
            if selected is None:
                model_asset_id = str(
                    relation.get(f"{prefix}_model_asset_id") or ""
                ).strip()
                model_matches = [
                    item
                    for item in directory
                    if item["model_asset_id"] == model_asset_id
                ]
                if len(model_matches) == 1:
                    selected = model_matches[0]
            if selected is None:
                continue
            values = {
                f"{prefix}_instance_id": selected["instance_id"],
                f"{prefix}_model_asset_id": selected["model_asset_id"],
                f"{prefix}_quantity_index": selected["quantity_index"],
            }
            for field, value in values.items():
                if relation.get(field) != value:
                    relation[field] = value
                    changed = True
    return changed


def _folded_text(value):
    return " ".join(str(value or "").casefold().split())


def _process_alias_family(value):
    folded = _folded_text(value)
    folded = re.sub(r"(?<!\w)#\s*0*\d+(?!\d)", " ", folded)
    folded = re.sub(r"(?<!\w)no\.?\s*0*\d+(?!\d)", " ", folded)
    folded = re.sub(
        r"(?<!\w)(?:first|second|third|fourth|fifth|sixth|seventh|"
        r"eighth|ninth|tenth)(?!\w)",
        " ",
        folded,
    )
    return " ".join(folded.split())


def _linguistic_literal_sources(user_message, conversation_history=None):
    sources = [str(user_message or "")]
    for item in conversation_history or []:
        if isinstance(item, str):
            sources.append(item)
            continue
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").casefold()
        fields = ("user_message", "pending_request")
        if role != "assistant":
            fields += ("content", "text")
        for field in fields:
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                sources.append(value)
    return tuple(
        folded
        for value in sources
        if (folded := _folded_text(value))
    )


def _raw_user_source_messages(user_message, conversation_history=None):
    """Keep literal user wording available when a later turn preserves it."""

    values = [str(user_message or "")]
    for item in conversation_history or []:
        if isinstance(item, str):
            values.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").casefold() == "assistant":
            continue
        for field in ("user_message", "pending_request", "content", "text"):
            value = item.get(field)
            if isinstance(value, str) and value.strip():
                values.append(value)
    return tuple(value for value in values if value.strip())


def _literal_objective_source_for_type(objective_type, raw_source_messages):
    """Return a literal objective sentence for one recognized objective type."""

    target_type = OBJECTIVE_SYNONYMS.get(
        str(objective_type or "").strip(),
        str(objective_type or "").strip(),
    )
    if not target_type:
        return None
    for raw_message in raw_source_messages or ():
        text = str(raw_message or "")
        candidates = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            candidates.extend(
                item.strip()
                for item in re.split(r"(?<=[.!?])\s+(?!\d)", stripped)
                if item.strip()
            )
        if not candidates and text.strip():
            candidates.append(text.strip())
        for candidate in candidates:
            if (
                canonical_objective_type_from_literal(text, candidate)
                == target_type
            ):
                return candidate
    return None


def _route_facts_without_explicit_objective(source_text):
    """Recognize a route summary that must remain process data, not a user goal."""

    source = str(source_text or "")
    lowered = source.casefold()
    has_route_fact = bool(
        "->" in source
        or re.search(r"\b\d+(?:\.\d+)?\s*batches?\s+per\s+day\b", lowered)
        or re.search(r"\b(?:material\s+)?routes?\b", lowered)
    )
    explicit_goal = bool(
        re.search(
            r"\b(?:minimi[sz]e|maximi[sz]e|optim(?:ize|ise)|"
            r"reduc(?:e|ing)|shorten|improv(?:e|ing)|penalty|objective)\b",
            lowered,
        )
    )
    return has_route_fact and not explicit_goal


def _repair_objective_source_slots(
    value,
    literal_sources,
    raw_source_messages,
):
    """Keep real user objectives while removing route-only model inferences.

    A process route and its batch volume supply flow facts. They become a user
    objective only when the request explicitly asks to optimize them. A
    paraphrased source for a recognized objective is repaired from the literal
    request; an unrecognized goal remains for ordinary validation.
    """

    objectives = []
    repairs = []
    for index, item in enumerate(_normalize_objective_slots(value)):
        source_text = str(item.get("source_text") or "").strip()
        if _literal_span_in_sources(source_text, literal_sources):
            objectives.append(item)
            continue
        literal_source = _literal_objective_source_for_type(
            item.get("objective_type"),
            raw_source_messages,
        )
        if literal_source:
            repaired = copy.deepcopy(item)
            repaired["source_text"] = literal_source
            canonical_type = canonical_objective_type_from_literal(
                next(
                    (
                        str(message)
                        for message in raw_source_messages or ()
                        if _literal_span_in_sources(literal_source, (_folded_text(message),))
                    ),
                    "",
                ),
                literal_source,
            )
            if canonical_type:
                repaired["objective_type"] = canonical_type
            objectives.append(repaired)
            repairs.append(
                {
                    "index": index,
                    "action": "restore_literal_objective_source",
                    "basis": "recognized_explicit_user_objective",
                }
            )
            continue
        if _route_facts_without_explicit_objective(source_text):
            repairs.append(
                {
                    "index": index,
                    "action": "discard_route_fact_inferred_objective",
                    "basis": "route_volumes_are_process_facts_without_an_explicit_user_objective",
                }
            )
            continue
        objectives.append(item)
    return objectives, repairs


def _literal_span_in_sources(value, literal_sources):
    span = _folded_text(value)
    return bool(
        span
        and any(span in source for source in literal_sources)
    )


def _intent_contract_issue(
    issues,
    issue_type,
    field_path,
    message,
    *,
    resolution_policy="revise_intent",
    details=None,
):
    issues.append(
        {
            "issue_id": f"{issue_type}_{len(issues) + 1:03d}",
            "issue_type": issue_type,
            "field_path": field_path,
            "message": message,
            "severity": "blocking",
            "resolution_policy": resolution_policy,
            "details": copy.deepcopy(details or {}),
        }
    )


def _validate_grounded_intent_frame(
    frame,
    user_message,
    rag_grounding,
    *,
    instance_directory=None,
    conversation_history=None,
):
    """Return repairable intent diagnostics instead of terminating execution."""

    if not isinstance(frame, dict):
        return {
            "schema_version": "layout-agentic-intent-contract-v2",
            "status": "REPAIRABLE",
            "issues": [
                {
                    "issue_id": "invalid_intent_llm_response_001",
                    "issue_type": "invalid_intent_llm_response",
                    "field_path": None,
                    "message": "RAG-grounded intent output must be a JSON object.",
                    "severity": "blocking",
                    "resolution_policy": "revise_intent",
                    "details": {},
                }
            ],
            "repairs": [],
        }
    issues = []
    resolved_asset_ids_by_surface = {}

    def binding_surface(value):
        folded = _folded_text(value)
        folded = re.sub(r"^(?:(?:a|an|the)\s+)+", "", folded)
        # A leading numeral can be an equipment-class property (for example,
        # a three-axis versus four-axis CNC). Drop it only for the explicit
        # labeled multi-unit declaration form used to identify repeated units.
        if re.search(r"\b(?:labeled|labelled)\b", folded):
            folded = re.sub(
                r"^(?:(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+)+",
                "",
                folded,
            )
        return folded

    for match in rag_grounding.get("asset_matches") or []:
        if not isinstance(match, dict):
            continue
        surface = binding_surface(match.get("surface_text"))
        selected_model_asset_id = str(
            match.get("selected_model_asset_id") or ""
        ).strip()
        resolution = str(
            match.get("decision") or match.get("ambiguity_status") or ""
        ).strip().upper()
        # Compact cached groundings retain a selected ID but not its status. Live
        # RAG output supplies a resolution status, so a nonempty non-resolution
        # status must not be treated as an identity binding.
        if (
            not surface
            or not selected_model_asset_id
            or (resolution and resolution not in {"RESOLVED", "RESOLVE"})
        ):
            continue
        resolved_asset_ids_by_surface.setdefault(surface, set()).add(
            selected_model_asset_id
        )

    literal_sources = _linguistic_literal_sources(
        user_message,
        conversation_history,
    )
    raw_source_messages = _raw_user_source_messages(
        user_message,
        conversation_history,
    )
    stable_context_sources = [
        str(item.get("text") or item.get("content") or "")
        for item in conversation_history or []
        if isinstance(item, dict)
        and str(item.get("role") or "").casefold() == "engineering_context"
        and str(item.get("text") or item.get("content") or "").strip()
    ]

    def selected_for_relation_endpoint(endpoint_surface_text, source_text):
        endpoint_key = binding_surface(endpoint_surface_text)
        authoritative = {item["model_asset_id"] for item in _normalize_instance_directory(instance_directory or [])
                         if any(binding_surface(alias)==endpoint_key for alias in (item["stable_alias"].values() if isinstance(item["stable_alias"],dict) else [item["stable_alias"]]))}
        if authoritative:
            return authoritative
        direct = resolved_asset_ids_by_surface.get(endpoint_key, set())
        if direct:
            return direct
        source_key = _folded_text(source_text)
        endpoint_literal = _folded_text(endpoint_surface_text)
        if not endpoint_literal or endpoint_literal not in source_key:
            return set()
        declaration_matches = []
        for mention in frame.get("asset_mentions") or []:
            if not isinstance(mention, dict):
                continue
            declaration_surface = str(mention.get("surface_text") or "").strip()
            selected = resolved_asset_ids_by_surface.get(
                binding_surface(declaration_surface),
                set(),
            )
            if (
                len(selected) == 1
                and _declared_surface_matches(
                    endpoint_surface_text,
                    declaration_surface,
                )
            ):
                declaration_matches.append(selected)
        return declaration_matches[0] if len(declaration_matches) == 1 else set()

    declared_quantities = {}
    for index, mention in enumerate(frame.get("asset_mentions") or []):
        if not isinstance(mention, dict):
            _intent_contract_issue(
                issues,
                "intent_asset_not_grounded",
                f"asset_mentions[{index}]",
                "Asset mentions must be JSON objects.",
            )
            continue
        model_asset_id = str(mention.get("model_asset_id") or "").strip()
        surface = binding_surface(mention.get("surface_text"))
        selected_for_surface = resolved_asset_ids_by_surface.get(surface, set())
        if model_asset_id not in selected_for_surface:
            _intent_contract_issue(
                issues,
                "intent_asset_not_grounded",
                f"asset_mentions[{index}].model_asset_id",
                "Use the identity selected for this equipment mention by Engineering RAG.",
                resolution_policy=(
                    "ask_user" if not selected_for_surface else "revise_intent"
                ),
                details={"model_asset_id": model_asset_id or None},
            )
        if not _literal_span_in_sources(surface, literal_sources):
            _intent_contract_issue(
                issues,
                "intent_asset_not_grounded",
                f"asset_mentions[{index}].surface_text",
                "Copy the exact equipment mention span from the request.",
            )
        raw_quantity = mention.get("quantity")
        if (
            isinstance(raw_quantity, bool)
            or not isinstance(raw_quantity, int)
            or raw_quantity < 1
        ):
            _intent_contract_issue(
                issues,
                "intent_asset_quantity_invalid",
                f"asset_mentions[{index}].quantity",
                "Restore the positive integer quantity from the linguistic frame.",
            )
        elif model_asset_id:
            declared_quantities[model_asset_id] = (
                declared_quantities.get(model_asset_id, 0) + raw_quantity
            )

    site = frame.get("site")
    if isinstance(site, dict) and site:
        source_text = _folded_text(site.get("source_text"))
        if not _literal_span_in_sources(source_text, literal_sources):
            _intent_contract_issue(
                issues,
                "intent_site_not_grounded",
                "site.source_text",
                "Copy the exact span that states the site dimensions.",
            )

    for field_name in ("constraints", "objectives"):
        for index, item in enumerate(frame.get(field_name) or []):
            if not isinstance(item, dict):
                _intent_contract_issue(
                    issues,
                    "intent_requirement_not_grounded",
                    f"{field_name}[{index}]",
                    f"{field_name} entries must be JSON objects.",
                )
                continue
            source_text = _folded_text(item.get("source_text"))
            is_default_no_overlap = (
                field_name == "constraints"
                and item.get("constraint_type") == "no_overlap"
                and source_text == "default"
            )
            if not is_default_no_overlap and (
                not _literal_span_in_sources(
                    source_text,
                    literal_sources,
                )
            ):
                _intent_contract_issue(
                    issues,
                    "intent_requirement_not_grounded",
                    f"{field_name}[{index}].source_text",
                    f"Copy the exact request span for this {field_name[:-1]}.",
                )

    directory = _normalize_instance_directory(instance_directory or [])
    directory_by_model = {}
    for item in directory:
        directory_by_model.setdefault(item["model_asset_id"], []).append(item)
    for model_asset_id, declared_quantity in declared_quantities.items():
        available_quantity = len(directory_by_model.get(model_asset_id, []))
        if available_quantity and declared_quantity > available_quantity:
            _intent_contract_issue(
                issues,
                "intent_asset_quantity_conflicts_with_instance_directory",
                "asset_mentions",
                "Merge repeated references to equipment already declared in the request.",
                details={
                    "model_asset_id": model_asset_id,
                    "declared_quantity": declared_quantity,
                    "available_quantity": available_quantity,
                },
            )
    if (
        frame.get("intent_type")
        in {"generate_feasible_layout", "optimize_layout_from_scratch"}
        and not directory
        and stable_context_sources
    ):
        expected_context_routes = [
            relation
            for source in stable_context_sources
            for relation in _explicit_arrow_chain_relations(source)
        ]
        route_endpoints_ready = bool(expected_context_routes)
        for relation in expected_context_routes:
            for prefix in ("from", "to"):
                surface = str(relation.get(f"{prefix}_surface_text") or "").strip()
                if not surface or facility_role(surface):
                    continue
                if not selected_for_relation_endpoint(
                    surface,
                    relation.get("source_text") or "",
                ):
                    route_endpoints_ready = False
                    break
            if not route_endpoints_ready:
                break
        actual_route_spans = Counter(
            (
                _folded_text(relation.get("source_text")),
                relation.get("sequence_order"),
                relation.get("flow_weight"),
            )
            for relation in frame.get("process_relations") or []
            if isinstance(relation, dict)
        )
        missing_context_routes = [
            relation
            for relation in expected_context_routes
            if actual_route_spans[
                (
                    _folded_text(relation.get("source_text")),
                    relation.get("sequence_order"),
                    relation.get("flow_weight"),
                )
            ]
            <= 0
        ]
        if route_endpoints_ready and missing_context_routes:
            _intent_contract_issue(
                issues,
                "intent_context_process_relations_missing",
                "process_relations",
                "Restore every explicit route edge from stable engineering context; do not ask the user to repeat declared routes.",
                resolution_policy="revise_intent",
                details={
                    "expected_context_route_edges": len(expected_context_routes),
                    "missing_context_route_edges": len(missing_context_routes),
                },
            )
    for relation_index, relation in enumerate(
        frame.get("process_relations") or []
    ):
        field_prefix = f"process_relations[{relation_index}]"
        if not isinstance(relation, dict):
            _intent_contract_issue(
                issues,
                "intent_process_relation_not_grounded",
                field_prefix,
                "Process relations must be JSON objects.",
            )
            continue
        if any(relation.get(prefix+"_instance_id") is not None and not any(
            item["instance_id"]==relation[prefix+"_instance_id"] and item["model_asset_id"]==relation.get(prefix+"_model_asset_id") and item["quantity_index"]==relation.get(prefix+"_quantity_index")
            for item in directory) for prefix in ("from","to")):
            _intent_contract_issue(
                issues,
                "intent_process_relation_not_grounded",
                field_prefix,
                "Use model identity and explicit quantity index instead of internal instance IDs.",
            )
        source_text = str(relation.get("source_text") or "").strip()
        if not _literal_span_in_sources(source_text, literal_sources):
            _intent_contract_issue(
                issues,
                "intent_process_relation_not_grounded",
                f"{field_prefix}.source_text",
                "Copy the exact source span that states this process relation from the current request or stable engineering context.",
            )
        for prefix in ("from", "to"):
            site_name = relation.get(f"{prefix}_site_feature_name")
            if site_name:
                if site_name not in _declared_site_feature_references(_raw_user_source_messages(user_message, conversation_history)):
                    _intent_contract_issue(issues, "intent_process_relation_not_grounded", f"{field_prefix}.{prefix}_site_feature_name", "This facility area is not declared in the request.")
                continue
            model_asset_id = str(
                relation.get(f"{prefix}_model_asset_id") or ""
            ).strip()
            endpoint_surface_text = str(
                relation.get(f"{prefix}_surface_text") or ""
            ).strip()
            selected_for_endpoint = selected_for_relation_endpoint(
                endpoint_surface_text,
                source_text,
            )
            if model_asset_id not in selected_for_endpoint:
                _intent_contract_issue(
                    issues,
                    "intent_process_relation_not_grounded",
                    f"{field_prefix}.{prefix}_model_asset_id",
                    "Bind the literal process endpoint to its Engineering RAG-selected asset identity.",
                    resolution_policy=(
                        "ask_user" if not selected_for_endpoint else "revise_intent"
                    ),
                    details={"model_asset_id": model_asset_id or None},
                )
                continue
            model_directory = directory_by_model.get(model_asset_id, [])
            quantity = max(
                declared_quantities.get(model_asset_id, 0),
                len(model_directory),
                1,
            )
            quantity_index = relation.get(f"{prefix}_quantity_index")
            if quantity_index is None and quantity > 1:
                source_identifies_any = [
                    item
                    for item in model_directory
                    if _source_identifies_quantity(
                        source_text,
                        item.get("stable_alias"),
                        item["quantity_index"],
                    )
                ]
                _intent_contract_issue(
                    issues,
                    "intent_process_relation_not_grounded",
                    f"{field_prefix}.{prefix}_quantity_index",
                    "Bind the repeated process endpoint to the explicit ordinal or stable alias.",
                    resolution_policy=(
                        "revise_intent"
                        if len(source_identifies_any) == 1
                        else "ask_user"
                    ),
                    details={"quantity": quantity},
                )
            elif quantity_index is not None and (
                isinstance(quantity_index, bool)
                or not isinstance(quantity_index, int)
                or quantity_index < 1
                or quantity_index > quantity
            ):
                _intent_contract_issue(
                    issues,
                    "intent_process_relation_not_grounded",
                    f"{field_prefix}.{prefix}_quantity_index",
                    "Use a valid 1-based quantity index for this process endpoint.",
                    details={
                        "quantity_index": quantity_index,
                        "quantity": quantity,
                    },
                )
            elif (
                quantity_index is not None
                and quantity > 1
                and not (
                    _source_identifies_quantity(
                        source_text,
                        next(
                            (
                                item.get("stable_alias")
                                for item in model_directory
                                if item["quantity_index"] == quantity_index
                            ),
                            None,
                        ),
                        quantity_index,
                    )
                    or _source_identifies_declared_quantity_label(
                        source_text,
                        endpoint_surface_text,
                        frame.get("asset_mentions") or [],
                        model_asset_id,
                        quantity_index,
                    )
                )
            ):
                _intent_contract_issue(
                    issues,
                    "intent_process_relation_not_grounded",
                    f"{field_prefix}.{prefix}_quantity_index",
                    "The relation source must identify the selected repeated endpoint.",
                    resolution_policy="ask_user",
                    details={"quantity_index": quantity_index},
                )
    all_source_text = "\n".join(raw_source_messages)
    if frame.get("process_relations") and frame.get("intent_type") not in {"edit_layout","constraint_repair"} and re.search(r"\bAGV\s+[A-Za-z]\b", all_source_text, re.I):
        incomplete = []
        for index, relation in enumerate(frame.get("process_relations") or []):
            mode = relation.get("transport_mode")
            citation = relation.get("carrier_source_text")
            if mode not in {"operator","agv","conveyor"} or not _literal_span_in_sources(citation,literal_sources):
                incomplete.append(index)
            elif mode in {"agv","conveyor"}:
                mid = relation.get("carrier_model_asset_id")
                selected_ids = {m.get("selected_model_asset_id") for m in (rag_grounding or {}).get("asset_matches") or []}
                selected_ids.update(item.get("model_asset_id") for item in instance_directory or [])
                quantity = sum(int(m.get("quantity") or 0) for m in frame.get("asset_mentions") or [] if m.get("model_asset_id")==mid)
                quantity = max(quantity, sum(item.get("model_asset_id")==mid for item in instance_directory or []), 1)
                ordinal = relation.get("carrier_quantity_index")
                if mid not in selected_ids or quantity>1 and ordinal is None or ordinal is not None and (not isinstance(ordinal,int) or ordinal<1 or ordinal>quantity):
                    incomplete.append(index)
        expected_agvs = {(m.get("model_asset_id"),i) for m in frame.get("asset_mentions") or []
                         if re.search(r"\bAGVs?\b",str(m.get("surface_text") or ""),re.I) for i in range(1,int(m.get("quantity") or 0)+1)}
        expected_agvs.update((item.get("model_asset_id"),item.get("quantity_index")) for item in instance_directory or [] if re.search(r"\bAGV\s+[A-Za-z]\b",str(item.get("stable_alias") or ""),re.I))
        assigned_agvs = {(r.get("carrier_model_asset_id"),r.get("carrier_quantity_index")) for r in frame.get("process_relations") or [] if r.get("transport_mode")=="agv"}
        if expected_agvs-assigned_agvs:
            incomplete.append("named_agv_not_assigned")
        expected_conveyors = {m.get("model_asset_id") for m in frame.get("asset_mentions") or [] if re.search(r"\bconveyor\b",str(m.get("surface_text") or ""),re.I)}
        assigned_conveyors = {r.get("carrier_model_asset_id") for r in frame.get("process_relations") or [] if r.get("transport_mode")=="conveyor"}
        if expected_conveyors-assigned_conveyors:
            incomplete.append("conveyor_not_assigned")
        if incomplete:
            _intent_contract_issue(issues,"intent_transport_assignments_incomplete","process_relations",
                "Assign each transfer to the AGV, conveyor, or operators described in the request, and quote the sentence supplying that role.",
                resolution_policy="revise_intent",details={"relation_indices":incomplete})
    return {
        "schema_version": "layout-agentic-intent-contract-v2",
        "status": "VALID" if not issues else "REPAIRABLE",
        "issues": issues,
        "repairs": [],
    }


def _validate_process_relation_grounding(
    frame,
    user_message,
    allowed_asset_ids,
    instance_directory,
    *,
    blocker_type,
):
    relations = frame.get("process_relations") or []
    if not relations:
        return
    directory = _normalize_instance_directory(instance_directory or [])
    directory_by_model = {}
    for item in directory:
        directory_by_model.setdefault(item["model_asset_id"], []).append(item)

    declared_quantities = {}
    for mention in frame.get("asset_mentions") or []:
        if not isinstance(mention, dict) or not mention.get("model_asset_id"):
            continue
        model_asset_id = str(mention["model_asset_id"])
        raw_quantity = mention.get("quantity")
        if (
            isinstance(raw_quantity, bool)
            or not isinstance(raw_quantity, int)
            or raw_quantity < 1
        ):
            raise RAGBlockedError(
                blocker_type,
                "A process relation requires a positive integer asset declaration quantity.",
                details={"model_asset_id": model_asset_id},
            )
        declared_quantities[model_asset_id] = (
            declared_quantities.get(model_asset_id, 0) + raw_quantity
        )

    message_folded = " ".join(str(user_message or "").casefold().split())
    for relation_index, relation in enumerate(relations, start=1):
        if not isinstance(relation, dict):
            raise RAGBlockedError(
                blocker_type,
                "Process relations must be JSON objects.",
                details={"relation_index": relation_index},
            )
        unauthorized_instance_fields = [
            field
            for field in ("from_instance_id", "to_instance_id")
            if relation.get(field) is not None
        ]
        if unauthorized_instance_fields:
            raise RAGBlockedError(
                blocker_type,
                "LLM-authored process relations may not select internal instance IDs.",
                details={
                    "relation_index": relation_index,
                    "fields": unauthorized_instance_fields,
                },
            )
        source_text = str(relation.get("source_text") or "").strip()
        source_folded = " ".join(source_text.casefold().split())
        if not source_folded or source_folded not in message_folded:
            raise RAGBlockedError(
                blocker_type,
                "Process relations must cite an exact non-empty request span.",
                details={"relation_index": relation_index},
            )
        for prefix in ("from", "to"):
            site_name = relation.get(f"{prefix}_site_feature_name")
            if site_name:
                if site_name not in declared_facility_roles(user_message):
                    raise RAGBlockedError(blocker_type, "Facility process endpoints must name a declared floor area or door.", details={"relation_index": relation_index, "endpoint": prefix})
                continue
            model_asset_id = str(
                relation.get(f"{prefix}_model_asset_id") or ""
            ).strip()
            if not model_asset_id or model_asset_id not in allowed_asset_ids:
                raise RAGBlockedError(
                    blocker_type,
                    "Process relation endpoints must be grounded RAG asset IDs.",
                    details={
                        "relation_index": relation_index,
                        "endpoint": prefix,
                        "model_asset_id": model_asset_id or None,
                    },
                )
            model_directory = directory_by_model.get(model_asset_id, [])
            declared_quantity = declared_quantities.get(model_asset_id)
            if (
                model_directory
                and declared_quantity is not None
                and len(model_directory) != declared_quantity
            ):
                raise RAGBlockedError(
                    blocker_type,
                    "The RAG-resolved declaration quantity disagrees with the instance directory.",
                    details={
                        "model_asset_id": model_asset_id,
                        "declared_quantity": declared_quantity,
                        "directory_count": len(model_directory),
                    },
                )
            quantity = declared_quantity or len(model_directory) or 1
            quantity_index = relation.get(f"{prefix}_quantity_index")
            if quantity_index is None:
                if quantity > 1:
                    raise RAGBlockedError(
                        blocker_type,
                        "A repeated model endpoint requires an explicit 1-based quantity index.",
                        details={
                            "relation_index": relation_index,
                            "endpoint": prefix,
                            "model_asset_id": model_asset_id,
                            "quantity": quantity,
                        },
                    )
                continue
            if (
                isinstance(quantity_index, bool)
                or not isinstance(quantity_index, int)
                or quantity_index < 1
                or quantity_index > quantity
            ):
                raise RAGBlockedError(
                    blocker_type,
                    "Process relation quantity indices must be within the declared 1-based range.",
                    details={
                        "relation_index": relation_index,
                        "endpoint": prefix,
                        "model_asset_id": model_asset_id,
                        "quantity_index": quantity_index,
                        "quantity": quantity,
                    },
                )
            directory_match = next(
                (
                    item
                    for item in model_directory
                    if item["quantity_index"] == quantity_index
                ),
                None,
            )
            if model_directory and directory_match is None:
                raise RAGBlockedError(
                    blocker_type,
                    "The selected quantity index has no instance-directory identity.",
                    details={
                        "relation_index": relation_index,
                        "endpoint": prefix,
                        "model_asset_id": model_asset_id,
                        "quantity_index": quantity_index,
                    },
                )
            stable_alias = (
                directory_match.get("stable_alias") if directory_match else None
            )
            if not _source_identifies_quantity(
                source_text,
                stable_alias,
                quantity_index,
            ):
                raise RAGBlockedError(
                    blocker_type,
                    "Indexed process endpoints must cite the matching stable alias or ordinal.",
                    details={
                        "relation_index": relation_index,
                        "endpoint": prefix,
                        "model_asset_id": model_asset_id,
                        "quantity_index": quantity_index,
                    },
                )


def _source_identifies_quantity(source_text, stable_alias, quantity_index):
    folded = " ".join(str(source_text or "").casefold().split())
    if isinstance(stable_alias, dict):
        aliases = [stable_alias.get("zh"), stable_alias.get("en")]
    else:
        aliases = [stable_alias]
    aliases = [
        " ".join(str(alias).casefold().split())
        for alias in aliases
        if isinstance(alias, str) and alias.strip()
    ]
    if any((bool(re.search(r"(?<![A-Za-z0-9_-])"+re.escape(alias)+r"(?![A-Za-z0-9_-])", folded))
            if alias.isascii() else alias in folded) for alias in aliases):
        return True
    number = str(quantity_index)
    ordinal_words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }
    patterns = (
        rf"#\s*0*{number}(?!\d)",
        rf"第\s*0*{number}\s*(?:台|个|套|号)?",
        rf"(?<!\d)0*{number}\s*号",
        rf"(?<!\w)no\.?\s*0*{number}(?!\d)",
    )
    if any(re.search(pattern, folded) for pattern in patterns):
        return True
    ordinal = ordinal_words.get(quantity_index)
    return bool(ordinal and re.search(rf"(?<!\w){ordinal}(?!\w)", folded))


def _semantic_intent_step(
    llm_frame,
    user_message,
    *,
    active_layout_id,
    scene_context,
    agent_name,
    responsibility,
    output_artifact,
    linguistic_frame=None,
    intent_contract=None,
    deterministic_repairs=None,
    revision=1,
):
    linguistic_frame = copy.deepcopy(linguistic_frame or {})
    resolver = AuthoritativeContextResolver()
    scene_context = resolver.apply(user_message, scene_context or {})
    authoritative_context = scene_context["authoritative_context"]
    semantic_frame = _normalize_semantic_frame(llm_frame)
    semantic_frame["source"] = "llm"
    semantic_frame["llm_error"] = None
    semantic_frame = _apply_scene_context_to_semantic_frame(semantic_frame, scene_context)
    semantic_frame = resolver.enrich_semantic_frame(
        semantic_frame,
        authoritative_context,
    )
    intent_contract = resolver.filter_intent_contract(
        copy.deepcopy(intent_contract or _valid_intent_contract()),
        authoritative_context,
        semantic_frame=semantic_frame,
    )
    completeness, unresolved_requirements, semantic_evidence = _intent_completeness(
        user_message,
        semantic_frame,
        active_layout_id=active_layout_id,
        scene_context=scene_context,
        linguistic_frame=linguistic_frame,
    )
    for issue in intent_contract.get("issues") or []:
        issue_id = str(
            issue.get("issue_id")
            or f"intent_contract_{len(unresolved_requirements) + 1:03d}"
        )
        if any(
            item.get("requirement_id") == issue_id
            for item in unresolved_requirements
        ):
            continue
        unresolved_requirements.append(
            {
                "requirement_id": issue_id,
                "field_path": issue.get("field_path"),
                "description": issue.get("message")
                or "The semantic intent does not satisfy its grounding contract.",
                "severity": issue.get("severity") or "blocking",
                "resolution_policy": issue.get("resolution_policy")
                or "revise_intent",
            }
        )
    blocking = [
        item
        for item in unresolved_requirements
        if item.get("severity") == "blocking"
    ]
    completeness["blocking_count"] = len(blocking)
    completeness["warning_count"] = len(unresolved_requirements) - len(blocking)
    completeness["status"] = "complete" if not blocking else "blocking"
    completeness["can_enter_designer"] = bool(
        semantic_frame.get("intent_type")
        in SUPPORTED_DESIGN_INTENTS | EXISTING_LAYOUT_TRANSFORMATION_INTENTS
        and not blocking
    )
    semantic_frame["completeness"] = completeness
    semantic_frame["unresolved_requirements"] = unresolved_requirements
    semantic_frame["semantic_evidence"] = semantic_evidence
    semantic_frame["can_enter_designer"] = completeness["can_enter_designer"]
    semantic_frame["intent_contract"] = copy.deepcopy(intent_contract)
    semantic_frame["deterministic_repairs"] = copy.deepcopy(
        deterministic_repairs or []
    )
    semantic_frame["clarification_questions"] = _intent_clarification_questions(
        unresolved_requirements,
        semantic_frame.get("clarification_questions") or [],
    )
    compiler_message = _build_semantic_pipeline_message(user_message, semantic_frame)
    intent = _intent_from_semantic_frame(
        semantic_frame,
        route_intent("", active_layout_id=None),
        compiler_message,
        active_layout_id=active_layout_id,
    )
    intent["completeness"] = completeness
    intent["unresolved_requirements"] = unresolved_requirements
    intent["can_enter_designer"] = completeness["can_enter_designer"]
    return {
        "agent_name": agent_name,
        "intent": intent,
        "semantic_intent": semantic_frame,
        "linguistic_frame": linguistic_frame,
        "intent_contract": intent_contract,
        "compiler_message": compiler_message,
        "completeness": completeness,
        "unresolved_requirements": unresolved_requirements,
        "can_enter_designer": completeness["can_enter_designer"],
        "semantic_evidence": semantic_evidence,
        "intent_revision_id": f"intent_v{int(revision)}",
        "trace": _trace(
            agent_name,
            responsibility,
            "user_message + scene_context",
            output_artifact,
            iteration=revision,
            output_artifacts=[output_artifact],
            decision=(
                "continue_to_compiler"
                if completeness["can_enter_designer"]
                else "request_clarification"
            ),
            evidence=[item["evidence_id"] for item in semantic_evidence],
        ),
    }


def reconcile_authoritative_intent_step(
    intent_step,
    user_message,
    *,
    active_layout_id=None,
    scene_context=None,
):
    """Re-evaluate a frozen semantic frame against current authoritative state.

    The linguistic/LLM interpretation remains frozen for ablation fairness.
    Only deterministic context propagation, completeness, and the compiled
    intent artifact are refreshed.
    """

    frozen = copy.deepcopy(intent_step or {})
    semantic = frozen.get("semantic_intent")
    if not isinstance(semantic, dict):
        return frozen
    current_user_input = (
        (scene_context or {}).get("current_user_input") or user_message
    )
    linguistic_frame = frozen.get("linguistic_frame") or {}
    deterministic_repairs = list(
        semantic.get("deterministic_repairs") or []
    )
    if isinstance(linguistic_frame, dict) and linguistic_frame:
        semantic, cached_repairs = _repair_grounded_intent_frame(
            semantic,
            current_user_input,
            _grounding_from_cached_semantic_frame(semantic),
            linguistic_frame,
            instance_directory=_current_instance_directory(
                scene_context or {}
            ),
            conversation_history=(scene_context or {}).get(
                "conversation_history"
            )
            or [],
        )
        semantic = _bind_authoritative_edit_operation_instances(
            semantic,
            _current_instance_directory(scene_context or {}),
        )
        deterministic_repairs.extend(cached_repairs)
    revision = _intent_revision_number(frozen.get("intent_revision_id"))
    refreshed = _semantic_intent_step(
        semantic,
        user_message,
        active_layout_id=active_layout_id,
        scene_context=scene_context or {},
        agent_name=frozen.get("agent_name") or IntentUnderstandingAgent.name,
        responsibility=IntentUnderstandingAgent.responsibility,
        output_artifact=frozen.get("intent_revision_id") or f"intent_v{revision}",
        linguistic_frame=linguistic_frame,
        intent_contract=frozen.get("intent_contract") or _valid_intent_contract(),
        deterministic_repairs=deterministic_repairs,
        revision=revision,
    )
    for key in ("semantic_bundle_id", "grounding_identity"):
        if key in frozen:
            refreshed[key] = copy.deepcopy(frozen[key])
    refreshed["semantic_bundle_reused"] = True
    previous_trace = frozen.get("trace") or {}
    trace = refreshed.get("trace") or {}
    for key in (
        "rag_backend",
        "rag_payload_sha256",
        "evidence",
        "grounding_identity",
    ):
        if key in previous_trace:
            trace[key] = copy.deepcopy(previous_trace[key])
    trace["decision"] = "reuse_frozen_semantic_bundle_after_authoritative_resolution"
    trace["llm_call_count"] = 0
    trace["semantic_bundle_reused"] = True
    refreshed["trace"] = trace
    return enforce_agent_output(
        refreshed.get("agent_name") or IntentUnderstandingAgent.name,
        refreshed,
    )


def _grounding_from_cached_semantic_frame(semantic_frame):
    """Rebuild only the asset bindings needed for deterministic cache repair."""

    matches = []
    for item in (semantic_frame or {}).get("asset_mentions") or []:
        if not isinstance(item, dict):
            continue
        model_asset_id = str(item.get("model_asset_id") or "").strip()
        surface_text = str(item.get("surface_text") or "").strip()
        if not model_asset_id or not surface_text:
            continue
        matches.append(
            {
                "surface_text": surface_text,
                "selected_model_asset_id": model_asset_id,
                "candidate_model_asset_ids": [
                    {"model_asset_id": model_asset_id}
                ],
            }
        )
    return {"asset_matches": matches}


def _bind_authoritative_edit_operation_instances(
    frame,
    instance_directory,
    *,
    source_messages=None,
):
    """Bind an edit target only when model identity plus ordinal is unique."""

    result = copy.deepcopy(frame or {})
    directory = _normalize_instance_directory(instance_directory or [])
    for operation in result.get("edit_operations") or []:
        if not isinstance(operation, dict) or operation.get("instance_id"):
            continue
        model_asset_id = str(
            operation.get("target_model_asset_id") or ""
        ).strip()
        candidates = [
            item
            for item in directory
            if item["model_asset_id"] == model_asset_id
        ]
        if len(candidates) == 1:
            operation["instance_id"] = candidates[0]["instance_id"]
            continue
        source_text = (
            operation.get("surface_text")
            or operation.get("source_text")
            or ""
        )
        matches = [
            item
            for item in candidates
            if _source_identifies_quantity(
                source_text,
                item.get("stable_alias"),
                item["quantity_index"],
            )
        ]
        if len(matches) == 1:
            operation["instance_id"] = matches[0]["instance_id"]
            continue
        quantity_index = _declared_edit_label_quantity_index(
            source_text,
            candidates,
            source_messages or (),
        )
        if quantity_index is not None:
            selected = next(
                (
                    item
                    for item in candidates
                    if item["quantity_index"] == quantity_index
                ),
                None,
            )
            if selected is not None:
                operation["instance_id"] = selected["instance_id"]
    return result


def _declared_edit_label_quantity_index(
    target_surface,
    candidates,
    source_messages,
):
    """Resolve an A/B edit target only from a matching literal declaration."""

    candidate_count = len(candidates)
    if candidate_count < 2:
        return None
    for source in source_messages:
        for declaration in _literal_labeled_declarations(source, candidate_count):
            if not _edit_target_matches_labeled_declaration(
                target_surface,
                declaration,
            ):
                continue
            quantity_index = _declared_label_quantity_index(
                target_surface,
                declaration,
            )
            if quantity_index is not None:
                return quantity_index
    return None


def _edit_target_matches_labeled_declaration(target_surface, declaration):
    """Match an edit target to a literal labeled declaration conservatively."""

    declaration_surface = declaration.get("surface_text") or ""
    if _declared_surface_matches(target_surface, declaration_surface):
        return True
    labels = {str(label).casefold() for label in declaration.get("labels") or ()}
    tokens = re.findall(r"[a-z0-9]+", _folded_text(target_surface))
    if not tokens or tokens[-1] not in labels:
        return False
    base = " ".join(tokens[:-1])
    return bool(base) and _declared_surface_matches(base, declaration_surface)


def _literal_labeled_declarations(source, quantity):
    declarations = []
    for match in re.finditer(
        r"(?P<surface>[^.;\n]*?\b(?:labeled|labelled)\s+"
        r"(?P<labels>[^;.\n]+))",
        str(source or ""),
        flags=re.IGNORECASE,
    ):
        surface_text = re.sub(
            r"\b(?:labeled|labelled)\s*$",
            "",
            match.group("surface").strip(),
            flags=re.IGNORECASE,
        ).strip()
        labels = _leading_declared_labels(match.group("labels"), quantity)
        if not labels:
            continue
        declarations.append(
            {
                "surface_text": surface_text,
                "quantity": quantity,
                "labels": labels,
            }
        )
    return declarations


def _leading_declared_labels(value, quantity):
    """Read exactly the declared labels before any descriptive clause.

    Natural requirements commonly continue after a label list (for example,
    ``labeled A and B for general component machining``).  Only the leading
    comma/and/or-separated labels are identities; later descriptive words are
    not labels.
    """

    if quantity < 1:
        return ()
    token = r"[A-Za-z0-9][A-Za-z0-9_-]*"
    sequence = token + (rf"\s*(?:,|and|or)\s*{token}" * (quantity - 1))
    match = re.match(rf"^\s*(?P<labels>{sequence})(?=\s*(?:$|[;.]|,|\bfor\b|\bused\b|\bwith\b|\bthat\b|\bwhich\b))", str(value or ""), flags=re.IGNORECASE)
    if not match:
        return ()
    labels = tuple(
        item.casefold()
        for item in re.findall(token, match.group("labels"))
        if item.casefold() not in {"and", "or"}
    )
    if len(labels) != quantity or len(set(labels)) != quantity:
        return ()
    return labels


def _intent_revision_number(value):
    match = re.search(r"(\d+)$", str(value or ""))
    return max(1, int(match.group(1))) if match else 1


def _normalize_semantic_frame(frame):
    if not isinstance(frame, dict):
        raise RAGBlockedError(
            "invalid_intent_llm_response",
            "Intent LLM response must be a JSON object; rule fallback is disabled.",
            details={"fallback_enabled": False},
        )

    intent_type = str(frame.get("intent_type") or "unknown_or_unsupported").strip()
    intent_type = {
        "generate_layout": "generate_feasible_layout",
    }.get(intent_type, intent_type)
    if intent_type not in {
        "generate_feasible_layout",
        "optimize_layout_from_scratch",
        "optimize_existing_layout",
        "edit_layout",
        "constraint_repair",
        "validate_layout",
        "evaluate_layout",
        "unknown_or_unsupported",
    }:
        intent_type = "unknown_or_unsupported"

    normalized = {
        "intent_type": intent_type,
        "confidence": _bounded_float(frame.get("confidence"), default=0.0),
        "site": _normalize_site_slot(frame.get("site")),
        "asset_mentions": _normalize_asset_mentions(frame.get("asset_mentions")),
        "constraints": _normalize_constraint_slots(frame.get("constraints")),
        "objectives": _normalize_objective_slots(frame.get("objectives")),
        "process_relations": _normalize_process_relations(
            frame.get("process_relations")
        ),
        "edit_operations": _normalize_edit_operations(frame.get("edit_operations")),
        "controlled_initial_violations": copy.deepcopy(
            [
                item
                for item in frame.get("controlled_initial_violations") or []
                if isinstance(item, dict)
            ]
        ),
        "controlled_violation": copy.deepcopy(
            frame.get("controlled_violation")
            if isinstance(frame.get("controlled_violation"), dict)
            else None
        ),
        "repair_scope": copy.deepcopy(
            frame.get("repair_scope")
            if isinstance(frame.get("repair_scope"), dict)
            else {}
        ),
        "allowed_repair_operations": _string_list(
            frame.get("allowed_repair_operations")
        ),
        "edit_relaxation_policy": (
            str(frame.get("edit_relaxation_policy")).strip()
            if frame.get("edit_relaxation_policy")
            else None
        ),
        # Keep a first-class nullable slot so a clarification response can
        # prove that an omitted edit distance was not guessed.
        "move_distance": _optional_float(frame.get("move_distance")),
        "domain_intent": _normalize_domain_intent(frame.get("domain_intent")),
        "requires_existing_layout": bool(
            frame.get("requires_existing_layout")
            or intent_type in {
                "optimize_existing_layout",
                "edit_layout",
                "constraint_repair",
                "validate_layout",
                "evaluate_layout",
            }
        ),
        "missing_information": _string_list(frame.get("missing_information")),
        "clarification_questions": _string_list(frame.get("clarification_questions")),
        "reasoning_summary": str(frame.get("reasoning_summary") or "").strip()[:400],
    }
    if not normalized["reasoning_summary"]:
        normalized["reasoning_summary"] = "LLM extracted layout intent and required design slots."
    return normalized


def _apply_scene_context_to_semantic_frame(frame, scene_context):
    enriched = copy.deepcopy(frame)
    layout = scene_context.get("last_layout") or scene_context.get("seed_layout") or {}
    site = layout.get("site") or {}
    width = _optional_float(site.get("width_m") or (site.get("dimensions_m") or {}).get("width_m"))
    depth = _optional_float(site.get("depth_m") or (site.get("dimensions_m") or {}).get("depth_m"))
    if not enriched.get("site") and width and depth:
        enriched["site"] = {
            "width_m": width,
            "depth_m": depth,
            "source_text": "scene_context.last_layout.site",
            "value_source": "existing_layout",
        }
    if enriched.get("intent_type") in {
        "optimize_existing_layout",
        "edit_layout",
        "constraint_repair",
        "validate_layout",
        "evaluate_layout",
    } and not enriched.get("asset_mentions"):
        quantity_by_asset = {}
        for placement in layout.get("placements") or []:
            model_asset_id = placement.get("model_asset_id")
            if model_asset_id:
                quantity_by_asset[model_asset_id] = quantity_by_asset.get(model_asset_id, 0) + 1
        enriched["asset_mentions"] = [
            {
                "surface_text": model_asset_id,
                "model_asset_id": model_asset_id,
                "quantity": quantity,
                "value_source": "existing_layout",
            }
            for model_asset_id, quantity in sorted(quantity_by_asset.items())
        ]
        if quantity_by_asset:
            enriched["inherited_layout_id"] = layout.get("layout_id")
    overrides = scene_context.get("semantic_overrides") or {}
    override_intent = overrides.get("intent_type")
    if override_intent in (
        SUPPORTED_DESIGN_INTENTS
        | INSPECTION_INTENTS
        | EXISTING_LAYOUT_TRANSFORMATION_INTENTS
        | {"unknown_or_unsupported"}
    ):
        enriched["intent_type"] = override_intent
    override_site = _normalize_site_slot(overrides.get("site"))
    if override_site:
        override_site["value_source"] = "user_input"
        enriched["site"] = override_site
    if overrides.get("asset_mentions"):
        enriched["asset_mentions"] = _normalize_asset_mentions(overrides["asset_mentions"])
        for mention in enriched["asset_mentions"]:
            mention["value_source"] = "user_input"
    process_relations = (
        overrides.get("process_relations")
        or scene_context.get("process_relations")
    )
    if process_relations:
        enriched["process_relations"] = _normalize_process_relations(process_relations)
    if overrides.get("constraints"):
        enriched["constraints"] = _normalize_constraint_slots([
            item if isinstance(item, dict) else {
                "constraint_type": str(item),
                "severity": "hard",
                "source_text": "confirmed in requirement review",
            }
            for item in overrides["constraints"]
        ])
    if overrides.get("objectives"):
        enriched["objectives"] = _normalize_objective_slots([
            item if isinstance(item, dict) else {
                "objective_type": str(item),
                "source_text": "confirmed in requirement review",
            }
            for item in overrides["objectives"]
        ])
    if overrides.get("edit_operations"):
        enriched["edit_operations"] = _normalize_edit_operations(overrides["edit_operations"])
    if overrides:
        enriched["missing_information"] = []
        enriched["clarification_questions"] = []
    return _derive_explicit_central_aisle_repair_scope(
        enriched,
        scene_context,
    )


def _derive_explicit_central_aisle_repair_scope(frame, scene_context):
    """Resolve an active layout's newly blocked aisle units before contract review.

    The user can authorize moving "only the units that obstruct" an explicit
    aisle widening without knowing internal instance identifiers. This derives
    that narrow scope from the accepted parent layout and the updated canonical
    site. It does not select equipment, alter the requested aisle, or broaden
    the user's repair authority.
    """

    result = copy.deepcopy(frame or {})
    context = scene_context or {}
    authoritative_context = context.get("authoritative_context") or {}
    current_user_input = str(
        context.get("current_user_input")
        or authoritative_context.get("current_user_input")
        or ""
    )
    if (
        result.get("intent_type") != "constraint_repair"
        or not _explicit_central_aisle_width_repair(current_user_input)
    ):
        return result

    parent_layout = copy.deepcopy(
        context.get("last_layout") or context.get("seed_layout") or {}
    )
    parent_flp_ir = copy.deepcopy(
        context.get("last_flp_ir") or context.get("base_flp_ir") or {}
    )
    site_snapshot = _compiled_site_snapshot(parent_flp_ir, context)
    if not parent_layout or not parent_flp_ir or not site_snapshot:
        return result

    accepted_semantic_intent = (
        context.get("current_semantic_intent")
        or context.get("frozen_semantic_intent")
        or context.get("accepted_semantic_intent")
        or result
    )
    site_update = apply_central_aisle_width_update(
        current_user_input,
        site_snapshot,
        organization_text=context.get("engineering_context"),
        semantic_intent=accepted_semantic_intent,
        equipment_instances=parent_flp_ir.get("equipment_instances") or [],
        intent_id="INTENT_001",
    )
    if site_update.get("unresolved_items"):
        return result

    expected_instance_ids = [
        str(item.get("instance_id"))
        for item in parent_flp_ir.get("equipment_instances") or []
        if isinstance(item, dict) and item.get("instance_id")
    ]
    if not expected_instance_ids:
        expected_instance_ids = [
            str(item.get("instance_id"))
            for item in parent_layout.get("placements") or []
            if isinstance(item, dict) and item.get("instance_id")
        ]
    if not expected_instance_ids:
        return result

    # This is a classification pass only. The final repair contract narrows
    # the solver to the units that the updated corridor actually intersects.
    probe_flp_ir = copy.deepcopy(parent_flp_ir)
    probe_scope = copy.deepcopy(probe_flp_ir.get("optimization_scope") or {})
    probe_scope["locked_instances"] = []
    probe_scope["movable_instances"] = list(expected_instance_ids)
    probe_flp_ir["optimization_scope"] = probe_scope
    try:
        canonical_spec = CanonicalConstraintCompiler().compile(
            site_update.get("site_snapshot") or {},
            flp_ir=probe_flp_ir,
            reference_layout=parent_layout,
        )
        validation = CanonicalConstraintValidator().validate(
            parent_layout,
            canonical_spec,
            reference_layout=parent_layout,
            expected_instance_ids=expected_instance_ids,
        )
    except (KeyError, TypeError, ValueError):
        # The normal validation path will report incomplete authoritative state.
        return result

    affected_features = set(site_update.get("changed_site_feature_ids") or [])
    affected_zones = set(site_update.get("changed_zone_ids") or [])
    aisle_ids = {
        str(item.get("site_feature_id"))
        for item in (site_update.get("site_snapshot") or {}).get("site_features") or []
        if isinstance(item, dict)
        and str(item.get("site_feature_id") or "").startswith(
            "natural_site/central_"
        )
        and str(item.get("site_feature_id") or "").endswith("_aisle")
    }
    affected_features.update(aisle_ids)
    changed_rows = {
        int(match.group(1))
        for zone_id in affected_zones
        if (match := re.search(r"workstation_bay_row_(\d+)_column_\d+", str(zone_id)))
    }
    row_members = {}
    for bay in (parent_layout.get("site") or {}).get("workstation_bays") or []:
        if not isinstance(bay, dict):
            continue
        try:
            row_index = int(bay.get("row_index"))
        except (TypeError, ValueError):
            continue
        row_members.setdefault(row_index, [])
        for instance_id in bay.get("assigned_instance_ids") or []:
            instance_id = str(instance_id)
            if instance_id in expected_instance_ids and instance_id not in row_members[row_index]:
                row_members[row_index].append(instance_id)
    changed_row_members = {
        instance_id
        for row_index in changed_rows
        for instance_id in row_members.get(row_index, [])
    }
    allows_row_coupling = bool(
        re.search(
            r"\brow\s+members?\b.{0,100}\bmove\s+together\b",
            current_user_input,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    corridor_violations = []
    row_alignment_indexes = set()
    for item in validation.get("violations") or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "")
        obstacle_id = str(item.get("obstacle_id") or "")
        instance_ids = {
            str(instance_id) for instance_id in item.get("instance_ids") or []
        }
        allowed_region_ids = {
            str(region_id)
            for region_id in item.get("target_value") or []
        } if isinstance(item.get("target_value"), (list, tuple, set)) else set()
        direct_affected = (
            code in {"CORRIDOR_OBSTRUCTION", "ENTRANCE_EXIT_OBSTRUCTION"}
            and obstacle_id in affected_features
        ) or (
            code in {
                "FORBIDDEN_ZONE_INTERSECTION",
                "SERVICE_ENVELOPE_INTERSECTION",
            }
            and obstacle_id in affected_features | affected_zones
        ) or (
            code == "OUTSIDE_ALLOWED_REGION"
            and bool(allowed_region_ids & affected_zones)
        ) or (
            code == "SERVICE_ENVELOPE_OUTSIDE_BOUNDARY"
            and bool(instance_ids & changed_row_members)
        )
        if direct_affected:
            corridor_violations.append(copy.deepcopy(item))
        if code == "WORKSTATION_ROW_ALIGNMENT" and changed_rows:
            matching_rows = {
                row_index
                for row_index in changed_rows
                if instance_ids & set(row_members.get(row_index, []))
            }
            if matching_rows and allows_row_coupling:
                corridor_violations.append(copy.deepcopy(item))
                row_alignment_indexes.update(matching_rows)

    movable_instance_ids = []
    for violation in corridor_violations:
        for instance_id in violation.get("instance_ids") or []:
            instance_id = str(instance_id)
            if (
                instance_id in expected_instance_ids
                and instance_id not in movable_instance_ids
            ):
                movable_instance_ids.append(instance_id)
    if allows_row_coupling:
        for row_index in sorted(row_alignment_indexes):
            for instance_id in row_members.get(row_index, []):
                if instance_id not in movable_instance_ids:
                    movable_instance_ids.append(instance_id)
    movable_instance_set = set(movable_instance_ids)
    movable_instance_ids = [
        instance_id
        for instance_id in expected_instance_ids
        if instance_id in movable_instance_set
    ]
    if not movable_instance_ids:
        return result

    source_text = next(
        (
            str(item.get("source_text") or "").strip()
            for item in result.get("constraints") or []
            if isinstance(item, dict)
            and item.get("constraint_type") == "corridor_preservation"
            and str(item.get("source_text") or "").strip()
        ),
        current_user_input,
    )
    result["controlled_initial_violations"] = corridor_violations
    result["controlled_violation"] = {
        "constraint_type": "corridor_preservation",
        "source_text": source_text,
        # Keep the controlled violation at constraint scope. Each obstruction
        # is retained in the evidence list, and the executor must clear every
        # corridor obstruction rather than only the first affected unit.
        "derived_obstructing_instance_ids": list(movable_instance_ids),
        "derivation_source": "authoritative_parent_layout_canonical_validation",
    }
    result["repair_scope"] = {
        "movable_instance_ids": list(movable_instance_ids),
        "preserve_instance_ids": [
            instance_id
            for instance_id in expected_instance_ids
            if instance_id not in movable_instance_ids
        ],
        "derivation_source": "authoritative_parent_layout_canonical_validation",
    }
    if not result.get("allowed_repair_operations"):
        result["allowed_repair_operations"] = ["move"]
    result["missing_information"] = [
        item
        for item in result.get("missing_information") or []
        if not _site_repair_known_information(item)
    ]
    result["clarification_questions"] = [
        item
        for item in result.get("clarification_questions") or []
        if not _site_repair_known_information(item)
    ]
    result["repair_scope_metadata"] = {
        "updated_site_provenance": copy.deepcopy(
            site_update.get("provenance") or []
        ),
        "canonical_validation": {
            "status": validation.get("status"),
            "canonical_spec_sha256": validation.get("canonical_spec_sha256"),
            "corridor_violations": copy.deepcopy(corridor_violations),
            "changed_site_feature_ids": sorted(affected_features),
            "changed_zone_ids": sorted(affected_zones),
        },
    }
    return result


def _intent_completeness(
    user_message,
    frame,
    *,
    active_layout_id=None,
    scene_context=None,
    linguistic_frame=None,
):
    evidence = []
    scene_context = scene_context or {}
    linguistic_frame = linguistic_frame or {}

    def add_evidence(field_path, value_source, source_text, confidence):
        evidence.append(
            {
                "evidence_id": f"semantic_evidence_{len(evidence) + 1:03d}",
                "field_path": field_path,
                "value_source": value_source,
                "source_text": str(source_text or ""),
                "confidence": float(confidence),
            }
        )

    text_normalized = str(user_message or "").lower()
    site = frame.get("site") or {}
    site_source_text = str(site.get("source_text") or "")
    site_source = None
    if site:
        if site.get("value_source") in {
            "existing_layout",
            "authoritative_context",
        }:
            site_source = site.get("value_source")
        elif site.get("value_source") == "user_input" or site_source_text == "confirmed in requirement review":
            site_source = "user_input"
        elif _extract_site_slot(user_message) or (
            site_source_text and site_source_text.lower() in text_normalized
        ):
            site_source = "user_input"
        else:
            site_source = "model_inference"
        add_evidence("site", site_source, site_source_text, 1.0 if site_source == "user_input" else 0.9)

    for index, mention in enumerate(frame.get("asset_mentions") or []):
        surface = str(mention.get("surface_text") or mention.get("model_asset_id") or "")
        if mention.get("value_source") in {
            "existing_layout",
            "authoritative_context",
        }:
            source = mention.get("value_source")
        elif mention.get("value_source") == "user_input":
            source = "user_input"
        else:
            source = "user_input" if surface and surface.lower() in text_normalized else "model_inference"
        add_evidence(f"asset_mentions[{index}]", source, surface, 1.0 if source == "user_input" else 0.85)
    for index, constraint in enumerate(frame.get("constraints") or []):
        source_text = str(constraint.get("source_text") or "")
        source = "system_default" if source_text == "default" else "user_input"
        add_evidence(f"constraints[{index}]", source, source_text, 1.0 if source == "user_input" else 0.8)
    for index, objective in enumerate(frame.get("objectives") or []):
        source_text = str(objective.get("source_text") or "")
        source = "user_input" if (
            source_text == "confirmed in requirement review"
            or (source_text and source_text.lower() in text_normalized)
        ) else "model_inference"
        add_evidence(f"objectives[{index}]", source, source_text, 0.95 if source == "user_input" else 0.8)
    domain = (frame.get("domain_intent") or {}).get("domain")
    if domain:
        add_evidence("domain_intent", "model_inference", domain, (frame.get("domain_intent") or {}).get("confidence") or 0.0)
    existing_layout = scene_context.get("last_layout") or scene_context.get("seed_layout") or {}
    existing_flp_ir = scene_context.get("last_flp_ir") or {}
    existing_instance_count = len(existing_layout.get("placements") or [])
    if existing_instance_count:
        add_evidence(
            "existing_layout.equipment_instances",
            "existing_layout",
            existing_layout.get("layout_id") or active_layout_id,
            1.0,
        )

    unresolved = []

    def add_unresolved(requirement_id, field_path, description, severity="blocking", resolution="ask_user"):
        if any(item["requirement_id"] == requirement_id for item in unresolved):
            return
        unresolved.append(
            {
                "requirement_id": requirement_id,
                "field_path": field_path,
                "description": description,
                "severity": severity,
                "resolution_policy": resolution,
            }
        )

    intent_type = frame.get("intent_type")
    if intent_type in {"generate_feasible_layout", "optimize_layout_from_scratch"}:
        interface_source = "\n".join(
            value
            for value in (
                str(scene_context.get("engineering_context") or ""),
                str(user_message or ""),
            )
            if value.strip()
        )
        for index, question in enumerate(interface_questions(interface_source)):
            add_unresolved(f"manual_transfer_interface_{index+1}", "constraints.alignment", question)
    if intent_type in SUPPORTED_DESIGN_INTENTS and (not site or site_source == "model_inference"):
        add_unresolved(
            "missing_site_dimensions",
            "site",
            "A layout task requires user-provided site dimensions or dimensions inherited from an active layout; model-inferred dimensions are not accepted as facts.",
        )
    equipment_resolved = bool(
        frame.get("asset_mentions")
        or domain
        or (intent_type == "optimize_existing_layout" and existing_instance_count)
    )
    if intent_type in SUPPORTED_DESIGN_INTENTS and not equipment_resolved:
        add_unresolved(
            "missing_equipment_requirements",
            "asset_mentions",
            "At least one equipment reference or a resolvable engineering domain is required.",
        )
    if (
        intent_type == "optimize_existing_layout"
        and not active_layout_id
        and not existing_layout
    ):
        add_unresolved(
            "missing_active_layout",
            "active_layout_id",
            "Existing-layout optimization requires an active layout reference.",
        )
    if (
        intent_type
        in INSPECTION_INTENTS | EXISTING_LAYOUT_TRANSFORMATION_INTENTS
        and not (existing_layout and existing_flp_ir)
    ):
        add_unresolved(
            "missing_active_layout",
            "active_layout_id",
            "This task requires an active layout from the current session.",
        )
    if intent_type == "edit_layout" and not frame.get("edit_operations"):
        linguistic_operations = linguistic_frame.get("edit_operations") or []
        add_unresolved(
            "missing_edit_operations",
            "edit_operations",
            "Layout editing requires at least one move, rotate, resize_site, add, remove, or replace operation.",
            resolution=(
                "revise_intent" if linguistic_operations else "ask_user"
            ),
        )
    if intent_type == "edit_layout":
        preview_contract = TaskContractCompiler().compile(
            semantic_intent=frame,
            authoritative_context=scene_context.get(
                "authoritative_context"
            )
            or {},
        )
        missing_descriptions = {
            "edit_target": "Please identify the instance that should be edited.",
            "edit_operations": "Please specify the requested layout edit.",
            "move_distance": "Please specify the move distance.",
        }
        for field_path in preview_contract.missing_fields:
            add_unresolved(
                f"missing_edit_{field_path}",
                field_path,
                missing_descriptions.get(
                    field_path,
                    f"Please provide the missing edit field: {field_path}.",
                ),
            )
    if intent_type == "constraint_repair":
        preview_contract = TaskContractCompiler().compile(
            semantic_intent=frame,
            authoritative_context=scene_context.get(
                "authoritative_context"
            )
            or {},
        )
        for field_path in preview_contract.missing_fields:
            add_unresolved(
                f"missing_constraint_repair_{field_path}",
                field_path,
                f"ConstraintRepair requires {field_path}.",
            )
    if intent_type not in (
        SUPPORTED_DESIGN_INTENTS
        | INSPECTION_INTENTS
        | EXISTING_LAYOUT_TRANSFORMATION_INTENTS
    ):
        add_unresolved(
            "unsupported_design_intent",
            "intent_type",
            f"Intent '{intent_type}' cannot enter rough-layout design.",
            severity="blocking",
            resolution="route_or_clarify",
        )
    for index, item in enumerate(frame.get("missing_information") or []):
        lowered = str(item).lower()
        site_still_missing = not site or site_source == "model_inference"
        equipment_still_missing = not equipment_resolved
        active_layout_still_missing = (
            intent_type
            in {
                "optimize_existing_layout",
                "edit_layout",
                "constraint_repair",
                "validate_layout",
                "evaluate_layout",
            }
            and not (active_layout_id and existing_layout)
        )
        model_report_matches_missing_required_field = (
            site_still_missing
            and any(token in lowered for token in ("site", "dimension"))
        ) or (
            equipment_still_missing
            and any(token in lowered for token in ("equipment", "asset", "quantity"))
        ) or (active_layout_still_missing and "active layout" in lowered)
        severity = "blocking" if model_report_matches_missing_required_field else "warning"
        add_unresolved(
            f"model_reported_missing_{index + 1:03d}",
            None,
            str(item),
            severity=severity,
            resolution="ask_user" if severity == "blocking" else "record_warning",
        )

    blocking = [item for item in unresolved if item["severity"] == "blocking"]
    required_fields = {
        "site": "resolved" if site and site_source != "model_inference" else "missing",
        "equipment": "resolved" if equipment_resolved else "missing",
        "active_layout": (
            "resolved"
            if intent_type
            not in {
                "optimize_existing_layout",
                "edit_layout",
                "constraint_repair",
                "validate_layout",
                "evaluate_layout",
            }
            or existing_layout
            else "missing"
        ),
    }
    completeness = {
        "status": "complete" if not blocking else "blocking",
        "required_fields": required_fields,
        "blocking_count": len(blocking),
        "warning_count": len(unresolved) - len(blocking),
        "can_enter_designer": intent_type
        in SUPPORTED_DESIGN_INTENTS | EXISTING_LAYOUT_TRANSFORMATION_INTENTS
        and not blocking,
    }
    return completeness, unresolved, evidence


def _intent_clarification_questions(unresolved_requirements, existing_questions):
    questions = list(existing_questions)
    templates = {
        "missing_site_dimensions": "Please provide the site width and depth, for example 20m x 12m.",
        "missing_equipment_requirements": "Please identify the equipment, quantities, or the engineering domain to lay out.",
        "missing_active_layout": "Please select the existing layout that should be optimized.",
        "missing_edit_operations": "Please specify what to move, rotate, resize, add, remove, or replace.",
        "unsupported_design_intent": "Please clarify whether this is a new layout, existing-layout optimization, or an edit request.",
        "intent_asset_not_grounded": "Please choose the equipment description that best matches the requested role.",
        "intent_process_relation_not_grounded": "Please confirm the equipment choices before the route connections are applied.",
        "intent_requirement_not_grounded": "Please restate the constraint that should be preserved.",
    }
    for item in unresolved_requirements:
        if (
            item.get("severity") != "blocking"
            or item.get("resolution_policy") == "revise_intent"
        ):
            continue
        question = templates.get(item["requirement_id"], item["description"])
        if question not in questions:
            questions.append(question)
    return questions


def _intent_from_semantic_frame(frame, fallback_intent, compiler_message, *, active_layout_id=None):
    intent = copy.deepcopy(fallback_intent)
    intent_type = frame.get("intent_type") or "unknown_or_unsupported"
    intent["intent_type"] = intent_type
    intent["task_type"] = intent_type
    intent["user_message"] = compiler_message
    intent["confidence"] = _bounded_float(frame.get("confidence"), default=0.0)
    intent["requires_existing_layout"] = intent_type in {
        "optimize_existing_layout",
        "edit_layout",
        "constraint_repair",
        "validate_layout",
        "evaluate_layout",
    }
    intent["requires_site"] = intent_type in SUPPORTED_DESIGN_INTENTS
    intent["requires_confirmation"] = intent_type in {"unknown_or_unsupported", "edit_layout"}
    intent["active_layout_id"] = active_layout_id
    intent["detected_operations"] = [
        item["op_type"] for item in frame.get("edit_operations") or [] if item.get("op_type")
    ]
    intent["target_refs"] = []
    for operation in frame.get("edit_operations") or []:
        instance_id = operation.get("instance_id")
        model_asset_id = operation.get("target_model_asset_id")
        if instance_id:
            intent["target_refs"].append({
                "target_type": "equipment_instance",
                "surface_text": instance_id,
                "resolved_library_id": None,
                "resolved_instance_id": instance_id,
                "resolved_site_id": None,
                "resolved_model_asset_id": model_asset_id,
                "confidence": intent["confidence"],
            })
        elif model_asset_id:
            intent["target_refs"].append({
                "target_type": "model_asset",
                "surface_text": model_asset_id,
                "resolved_library_id": None,
                "resolved_instance_id": None,
                "resolved_site_id": None,
                "resolved_model_asset_id": model_asset_id,
                "confidence": intent["confidence"],
            })
    intent["provenance"] = [
        {
            "field_path": "intent.intent_type",
            "source_type": "llm_compiler",
            "source_id": "IntentUnderstandingAgent",
            "source_field": "intent_type",
            "confidence": intent["confidence"],
            "note": "Intent type supplied by the configured LLM; rule fallback is disabled.",
        }
    ]

    unresolved_items = []
    if intent["requires_existing_layout"] and not intent["active_layout_id"]:
        if not any(item.get("item_id") == "missing_active_layout" for item in unresolved_items):
            unresolved_items.append(
                {
                    "item_id": "missing_active_layout",
                    "item_type": "missing_field",
                    "field_path": "optimization_scope.base_layout_id",
                    "description": "Existing-layout optimization requires an active layout reference.",
                    "severity": "blocking",
                    "candidates": [],
                    "resolution_policy": "ask_user",
                }
            )
    intent["unresolved_items"] = unresolved_items
    return intent


def _build_semantic_pipeline_message(user_message, frame):
    parts = [str(user_message or "").strip()]
    site = frame.get("site") or {}
    width = site.get("width_m")
    depth = site.get("depth_m")
    if width and depth:
        parts.append(f"Site {float(width):g}m x {float(depth):g}m.")

    asset_fragments = []
    quantities_by_model = {}
    unbound_mentions = []
    for item in frame.get("asset_mentions") or []:
        model_asset_id = item.get("model_asset_id")
        surface_text = item.get("surface_text")
        quantity = max(1, int(_optional_float(item.get("quantity")) or 1))
        if model_asset_id:
            model_asset_id = str(model_asset_id)
            quantities_by_model[model_asset_id] = (
                quantities_by_model.get(model_asset_id, 0) + quantity
            )
        elif surface_text:
            unbound_mentions.append((str(surface_text), quantity))
    asset_fragments.extend(
        f"{quantity} units of {model_asset_id}"
        for model_asset_id, quantity in quantities_by_model.items()
    )
    asset_fragments.extend(
        f"{quantity} units of {surface_text}"
        for surface_text, quantity in unbound_mentions
    )
    if asset_fragments:
        parts.append("Selected model_asset_id " + " ".join(asset_fragments) + ".")

    constraint_types = {item.get("constraint_type") for item in frame.get("constraints") or []}
    if "no_overlap" in constraint_types:
        parts.append("Hard constraint: no overlap.")
    if "inside_boundary" in constraint_types:
        parts.append("Keep equipment inside the boundary.")

    objective_types = [item.get("objective_type") for item in frame.get("objectives") or [] if item.get("objective_type")]
    if objective_types:
        parts.append("Objectives: " + ", ".join(objective_types) + ".")
    if any("distance" in item for item in objective_types):
        parts.append("Minimize material handling distance.")
    return " ".join(part for part in parts if part)


def _extract_site_slot(message):
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*m?\s*(?:x|by|\*)\s*(\d+(?:\.\d+)?)\s*m?",
        str(message or ""),
        re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "width_m": float(match.group(1)),
        "depth_m": float(match.group(2)),
        "source_text": match.group(0),
    }


def _extract_asset_mentions(message):
    mentions = []
    text = str(message or "")
    for match in re.finditer(r"asset/fbx/[A-Za-z0-9_\-]+", text):
        mentions.append(
            {
                "surface_text": match.group(0),
                "model_asset_id": match.group(0),
                "quantity": _quantity_near_asset_mention(text, match),
            }
        )
    return mentions


def _quantity_near_asset_mention(text, match):
    prefix = text[max(0, match.start() - 40) : match.start()]
    suffix = text[match.end() : match.end() + 24]
    before = re.search(r"(\d+)\s*(?:台|个|套|units?|pcs?)?\s*(?:of\s+)?$", prefix, re.IGNORECASE)
    after = re.search(r"^\s*(\d+)\s*(?:台|个|套|units?|pcs?)?", suffix, re.IGNORECASE)
    quantity_match = before or after
    if not quantity_match:
        return 1
    return max(1, int(quantity_match.group(1)))


def _extract_constraint_slots(message):
    message_lower = str(message or "").lower()
    constraints = [{"constraint_type": "no_overlap", "severity": "hard", "source_text": "default"}]
    if any(term in message_lower for term in ["boundary", "inside", "site"]):
        constraints.append({"constraint_type": "inside_boundary", "severity": "hard", "source_text": "user_message"})
    return constraints


def _extract_objective_slots(message):
    message_lower = str(message or "").lower()
    if any(term in message_lower for term in ["distance", "logistics", "material handling", "物流", "距离"]):
        return [
            {
                "objective_type": "minimize_material_handling_distance",
                "source_text": "user_message",
            }
        ]
    return []


def _normalize_site_slot(value):
    if not isinstance(value, dict):
        return None
    width = _optional_float(value.get("width_m"))
    depth = _optional_float(value.get("depth_m"))
    if not width or not depth:
        return None
    return {
        "width_m": width,
        "depth_m": depth,
        "source_text": str(value.get("source_text") or ""),
    }


def _normalize_asset_mentions(value):
    if not isinstance(value, list):
        return []
    mentions = []
    for item in value:
        if not isinstance(item, dict):
            continue
        surface_text = str(item.get("surface_text") or item.get("model_asset_id") or "").strip()
        model_asset_id = str(item.get("model_asset_id") or "").strip()
        if not surface_text and not model_asset_id:
            continue
        mentions.append(
            {
                "surface_text": surface_text or model_asset_id,
                "model_asset_id": model_asset_id,
                "quantity": int(_optional_float(item.get("quantity")) or 1),
                **({"instance_labels": _string_list(item["instance_labels"])}
                   if item.get("instance_labels") else {}),
                **({"planning_overrides": copy.deepcopy(item["planning_overrides"])}
                   if isinstance(item.get("planning_overrides"), dict) else {}),
            }
        )
    return mentions


def _normalize_constraint_slots(value):
    if not isinstance(value, list):
        return []
    constraints = []
    for item in value:
        if not isinstance(item, dict):
            continue
        constraint_type = str(item.get("constraint_type") or "").strip()
        if not constraint_type:
            continue
        constraint_type = {
            "boundary": "inside_boundary",
            "site_boundary": "inside_boundary",
            "within_boundary": "inside_boundary",
            "inside_site_boundary": "inside_boundary",
        }.get(constraint_type, constraint_type)
        if any(existing["constraint_type"] == constraint_type for existing in constraints):
            continue
        constraints.append(
            {
                "constraint_type": constraint_type,
                "severity": str(item.get("severity") or "hard"),
                "source_text": str(item.get("source_text") or ""),
                **({"parameters": copy.deepcopy(item["parameters"])}
                   if isinstance(item.get("parameters"), dict) else {}),
            }
        )
    if not any(item["constraint_type"] == "no_overlap" for item in constraints):
        constraints.insert(0, {"constraint_type": "no_overlap", "severity": "hard", "source_text": "default"})
    return constraints


def _normalize_objective_slots(value):
    if not isinstance(value, list):
        return []
    objectives = []
    for item in value:
        if not isinstance(item, dict):
            continue
        objective_type = str(item.get("objective_type") or "").strip()
        if not objective_type:
            continue
        objectives.append(
            {
                "objective_type": objective_type,
                "source_text": str(item.get("source_text") or ""),
            }
        )
    return objectives


def _normalize_process_relations(value):
    if not isinstance(value, list):
        return []
    relations = []
    allowed_fields = {
        "from_site_feature_name",
        "to_site_feature_name",
        "from_surface_text",
        "to_surface_text",
        "product",
        "transport_mode",
        "carrier_model_asset_id",
        "carrier_instance_id",
        "carrier_quantity_index",
        "carrier_surface_text",
        "carrier_source_text",
        "from_instance_id",
        "to_instance_id",
        "from_model_asset_id",
        "to_model_asset_id",
        "from_quantity_index",
        "to_quantity_index",
        "flow_type",
        "flow_weight",
        "sequence_order",
        "required_connection",
        "source_text",
        "evidence_source",
        "evidence_ref",
    }
    for item in value:
        if not isinstance(item, dict):
            continue
        relation = {}
        for field in allowed_fields:
            raw = item.get(field)
            if field in {"from_quantity_index", "to_quantity_index", "carrier_quantity_index"}:
                parsed = _optional_float(raw)
                relation[field] = (
                    int(parsed)
                    if parsed is not None
                    and parsed.is_integer()
                    and parsed >= 1
                    else None
                )
            elif field == "sequence_order":
                relation[field] = int(_optional_float(raw)) if raw is not None else None
            elif field == "flow_weight":
                relation[field] = _optional_float(raw)
            elif field == "required_connection":
                relation[field] = bool(raw)
            else:
                relation[field] = str(raw).strip() if raw is not None else None
        has_from = relation.get("from_instance_id") or relation.get("from_model_asset_id") or relation.get("from_site_feature_name")
        has_to = relation.get("to_instance_id") or relation.get("to_model_asset_id") or relation.get("to_site_feature_name")
        if not has_from or not has_to:
            continue
        relation["flow_type"] = relation.get("flow_type") or "material"
        relation["flow_weight"] = (
            relation["flow_weight"] if relation.get("flow_weight") is not None else 1.0
        )
        relations.append(relation)
    return relations


def _normalize_edit_operations(value):
    if not isinstance(value, list):
        return []
    allowed = {"move", "rotate", "resize_site", "add", "remove", "replace"}
    fields = {
        "instance_id",
        "target_model_asset_id",
        "new_library_id",
        "new_model_asset_id",
        "quantity",
        "move_mode",
        "x_m",
        "y_m",
        "z_m",
        "dx_m",
        "dy_m",
        "dz_m",
        "rotation_mode",
        "target_yaw_deg",
        "delta_yaw_deg",
        "site_width_m",
        "site_depth_m",
        "placement_policy",
        "target_site_feature_id",
        "target_zone_id",
        "surface_text",
        "source_text",
    }
    operations = []
    for item in value:
        if not isinstance(item, dict):
            continue
        normalized_item = copy.deepcopy(item)
        op_type = str(
            normalized_item.get("op_type")
            or normalized_item.get("operation")
            or normalized_item.get("operation_type")
            or ""
        ).strip().lower()
        op_type = {
            "translate": "move",
            "translation": "move",
            "shift": "move",
            "delete": "remove",
        }.get(op_type, op_type)
        if op_type not in allowed:
            continue
        normalized_item["op_type"] = op_type
        if not normalized_item.get("surface_text") and normalized_item.get(
            "target_surface_text"
        ):
            normalized_item["surface_text"] = normalized_item[
                "target_surface_text"
            ]
        for legacy_field, canonical_field in (
            ("delta_x", "dx_m"),
            ("delta_y", "dy_m"),
            ("delta_z", "dz_m"),
        ):
            if (
                normalized_item.get(canonical_field) is None
                and normalized_item.get(legacy_field) is not None
            ):
                normalized_item[canonical_field] = normalized_item[legacy_field]
        operation = {"op_type": op_type}
        for field in fields:
            raw = normalized_item.get(field)
            if field in {"quantity"}:
                operation[field] = int(_optional_float(raw) or 1)
            elif field.endswith("_m") or field.endswith("_deg"):
                operation[field] = _optional_float(raw)
            else:
                normalized_value = str(raw).strip() if raw is not None else None
                if field == "placement_policy" and normalized_value in {"auto", "automatic"}:
                    normalized_value = "auto_place"
                operation[field] = normalized_value
        if op_type == "move":
            displacement_mode = operation.get("move_mode") in {
                "absolute_displacement",
                "relative",
                "translate",
                "translation",
                "shift",
            }
            if displacement_mode and any(
                operation.get(field) is not None
                for field in ("x_m", "y_m", "z_m")
            ):
                for axis in ("x", "y", "z"):
                    operation[f"d{axis}_m"] = operation.get(
                        f"{axis}_m"
                    )
                    operation[f"{axis}_m"] = None
                operation["move_mode"] = "relative"
            elif any(
                operation.get(field) is not None
                for field in ("dx_m", "dy_m", "dz_m")
            ):
                operation["move_mode"] = "relative"
            elif any(
                operation.get(field) is not None
                for field in ("x_m", "y_m", "z_m")
            ):
                operation["move_mode"] = "absolute"
            elif operation.get("move_mode") in {
                "absolute_displacement",
                "cartesian",
                "global",
                "translate",
                "translation",
                "shift",
            }:
                operation["move_mode"] = "relative"
        if op_type == "add" and not operation.get("new_model_asset_id"):
            operation["new_model_asset_id"] = operation.get("target_model_asset_id")
            if op_type == "add":
                operation["target_model_asset_id"] = None
        operations.append(operation)
    return operations


def _normalize_domain_intent(value):
    if not isinstance(value, dict):
        return None
    domain = value.get("domain")
    if not domain:
        return {"domain": None, "confidence": 0.0, "source_text": str(value.get("source_text") or "")}
    return {
        "domain": str(domain),
        "confidence": _bounded_float(value.get("confidence"), default=0.0),
        "source_text": str(value.get("source_text") or ""),
    }


def _optional_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bounded_float(value, *, default=0.0):
    number = _optional_float(value)
    if number is None:
        number = default
    return max(0.0, min(1.0, number))


def _string_list(value):
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _append_unique(items, value):
    if value not in items:
        items.append(value)


def _feedback_actions(feedback):
    if not feedback:
        return []
    actions = feedback.get("actions") or feedback.get("repair_actions") or []
    normalized = []
    for index, action in enumerate(actions, start=1):
        if isinstance(action, str):
            action = {"action_type": action}
        if not isinstance(action, dict):
            continue
        item = copy.deepcopy(action)
        item.setdefault("action_id", f"feedback_action_{index:03d}")
        item.setdefault("evidence_id", (feedback.get("source_revision_id") or "feedback") + f"_{index:03d}")
        normalized.append(item)
    return normalized


def _feedback_strategy(feedback):
    if not feedback:
        return "initial_solve"
    action_types = {item.get("action_type") for item in _feedback_actions(feedback)}
    if action_types & {
        "global_resolve",
        "continue_improvement_search",
        "intensify_search_within_fixed_budget",
        "move_swap_or_resolve",
    }:
        return "global_resolve_with_previous_layout_seed"
    if action_types:
        return "local_repair_then_resolve"
    return "resolve_with_feedback_context"


def _feedback_requires_fresh_start(feedback):
    source = str((feedback or {}).get("source") or "").casefold()
    action_types = {
        action.get("action_type") for action in _feedback_actions(feedback)
    }
    return source in {"validator", "validatoragent"} or bool(
        action_types
        & {
            "move_swap_or_resolve",
            "project_inside_boundary",
            "repair_overlap",
            "repair_layout",
        }
    )


def _user_constraints_preserved(flp_ir, parent_design):
    if not parent_design:
        return True
    previous = _user_hard_constraints(parent_design.get("flp_ir"))
    current = _user_hard_constraints(flp_ir)
    return previous == current


def _user_hard_constraints(flp_ir):
    return [
        copy.deepcopy(item)
        for item in (flp_ir or {}).get("constraints", {}).get("hard", [])
        if item.get("source") == "user_input"
    ]


def _solver_options_for_revision(
    requested_solver,
    revision,
    feedback,
    *,
    base_options=None,
    adaptive_budget=True,
):
    options = copy.deepcopy(base_options) if isinstance(base_options, dict) else {}
    if not adaptive_budget:
        if revision > 1:
            base_seed = int(options.get("seed") or 0)
            options["seed"] = base_seed + 100003 * (revision - 1)
        _apply_fixed_budget_feedback_policy(options, feedback)
        return options
    if revision <= 1:
        return options
    solver_name = normalize_solver_name(requested_solver) or DEFAULT_SOLVER_NAME
    if solver_name == "genetic_algorithm":
        adaptive_options = {
            "population_size": min(96, 48 + 12 * (revision - 1)),
            "generations": min(250, 100 + 50 * (revision - 1)),
            "seed": revision - 1,
            "stagnation_generations": 40,
        }
    else:
        adaptive_options = {}
    options.update(adaptive_options)
    for action in _feedback_actions(feedback):
        for key, value in (action.get("solver_parameter_changes") or {}).items():
            if key in options:
                options[key] = value
    return options


def _apply_fixed_budget_feedback_policy(options, feedback):
    actions = _feedback_actions(feedback)
    if not actions:
        return
    frozen_generations = options.get("generations")
    frozen_population = options.get("population_size")
    action_types = {action.get("action_type") for action in actions}
    source = str((feedback or {}).get("source") or "").casefold()
    if source in {"validator", "validatoragent"} or _feedback_requires_fresh_start(
        feedback
    ):
        options.update(
            seed=int(options.get("seed") or 0) + 10007,
            mutation_rate=0.4,
            crossover_rate=0.9,
            elite_fraction=0.05,
        )
    elif "global_resolve" in action_types:
        options.update(
            seed=int(options.get("seed") or 0) + 20011,
            mutation_rate=0.35,
            crossover_rate=0.9,
            elite_fraction=0.05,
        )
    elif "continue_improvement_search" in action_types:
        options.update(
            seed=int(options.get("seed") or 0) + 30011,
            mutation_rate=0.28,
            crossover_rate=0.92,
            elite_fraction=0.08,
        )
    elif "intensify_search_within_fixed_budget" in action_types:
        options.update(
            seed=int(options.get("seed") or 0) + 40009,
            mutation_rate=0.32,
            stagnation_generations=max(1, int(frozen_generations or 40)),
        )
    else:
        options.update(
            seed=int(options.get("seed") or 0) + 50021,
            mutation_rate=0.3,
        )
    allowed_tuning = {
        "crossover_rate",
        "elite_fraction",
        "mutation_rate",
        "mutation_step_m",
        "stagnation_generations",
        "tournament_size",
    }
    for action in actions:
        for key, value in (action.get("solver_parameter_changes") or {}).items():
            if key in allowed_tuning:
                options[key] = value
    if frozen_generations is not None:
        options["generations"] = frozen_generations
    if frozen_population is not None:
        options["population_size"] = frozen_population


def _revision_time_limit(base_limit, revision, *, adaptive_budget=True):
    if base_limit is None:
        return None
    if not adaptive_budget:
        return float(base_limit)
    return float(base_limit) * (1.0 + 0.5 * max(0, revision - 1))


def _design_revision(revision, flp_ir, rough_layout, feedback, *, parent_layout_id=None):
    return {
        "revision_index": int(revision),
        "design_revision_id": f"design_v{revision}",
        "flp_ir_revision_id": f"flp_ir_v{revision}" if flp_ir is not None else None,
        "layout_revision_id": f"layout_v{revision}" if rough_layout is not None else None,
        "parent_layout_id": parent_layout_id,
        "feedback_source": (feedback or {}).get("source"),
        "feedback_revision_id": (feedback or {}).get("source_revision_id"),
    }


def _design_mutation_summary(
    *,
    revision,
    parent_design,
    flp_ir,
    solver_options,
    baseline_solver_options,
    seed_layout,
    warm_start_policy,
    feedback,
):
    previous_report = (parent_design or {}).get("solver_report") or {}
    previous_options = previous_report.get("solver_options") or {}
    previous_flp_ir = (parent_design or {}).get("flp_ir")
    previous_layout = (parent_design or {}).get("rough_layout") or {}
    seed_layout = seed_layout if isinstance(seed_layout, dict) else {}
    action_types = [
        action.get("action_type")
        for action in _feedback_actions(feedback)
        if action.get("action_type")
    ]
    previous_seed = previous_options.get("seed")
    current_seed = solver_options.get("seed")
    previous_ir_hash = _json_sha256(previous_flp_ir)
    current_ir_hash = _json_sha256(flp_ir)
    structured_feedback = bool(action_types) and parent_design is not None
    action_solver_changes = {
        key: {
            "baseline": baseline_solver_options.get(key),
            "feedback": value,
        }
        for key, value in solver_options.items()
        if baseline_solver_options.get(key) != value
    }
    parent_layout_hash = _json_sha256(previous_layout)
    seed_layout_hash = _json_sha256(seed_layout) if seed_layout else None
    summary = {
        "mutation_type": (
            "initial_solve"
            if parent_design is None
            else ("structured_feedback" if structured_feedback else "bounded_retry")
        ),
        "feedback_source": (feedback or {}).get("source"),
        "feedback_action_types": action_types,
        "action_specific_solver_changes": action_solver_changes,
        "rng_seed": {
            "previous": previous_seed,
            "current": current_seed,
            "changed": parent_design is not None and previous_seed != current_seed,
        },
        "flp_ir": {
            "previous_sha256": previous_ir_hash,
            "current_sha256": current_ir_hash,
            "changed": parent_design is not None and previous_ir_hash != current_ir_hash,
            "user_hard_constraints_preserved": _user_constraints_preserved(
                flp_ir, parent_design
            ),
        },
        "solver_state": {
            "warm_start_policy": warm_start_policy,
            "previous_layout_id": previous_layout.get("layout_id"),
            "warm_start_layout_id": seed_layout.get("layout_id"),
            "warm_start_applied": bool(seed_layout),
            "previous_layout_sha256": parent_layout_hash,
            "warm_start_sha256": seed_layout_hash,
            "seed_layout_changed_vs_parent": (
                parent_design is not None and parent_layout_hash != seed_layout_hash
            ),
        },
    }
    action_specific_effective = bool(
        action_solver_changes
        or warm_start_policy == "discard_invalid_validator_candidate"
    )
    summary["action_specific_effective"] = action_specific_effective
    summary["effective_mutation"] = bool(
        action_specific_effective
        if structured_feedback
        else (
            summary["rng_seed"]["changed"]
            or summary["solver_state"]["seed_layout_changed_vs_parent"]
        )
    )
    if structured_feedback and not action_specific_effective:
        raise RAGBlockedError(
            "ineffective_structured_feedback",
            "Structured feedback did not differ from the bounded-retry solver state.",
            details={"revision": int(revision), "action_types": action_types},
        )
    return summary


def _json_sha256(value):
    if value is None:
        return None
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _append_revision_provenance(flp_ir, revision, feedback):
    actions = _feedback_actions(feedback)
    action_types = [item.get("action_type") for item in actions if item.get("action_type")]
    flp_ir.setdefault("provenance", []).append(
        {
            "field_path": "revision_history",
            "source_type": "computed",
            "source_id": (feedback or {}).get("source_revision_id") or f"feedback_v{revision - 1}",
            "source_field": ",".join(action_types) or "resolve_again",
            "confidence": 1.0,
            "note": f"Designer revision {revision} applied structured feedback without changing user hard constraints.",
        }
    )


def _requested_solver_from_context(scene_context):
    for key in ("solver", "solver_name", "optimizer", "method"):
        value = scene_context.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _log_solver_execution(report):
    print(
        "solver_execution "
        f"requested={report.get('solver_requested', DEFAULT_SOLVER_NAME)} "
        f"used={report.get('solver_used')} "
        f"selection_source={report.get('selection_source')} "
        f"default_fallback_used={report.get('default_fallback_used')} "
        f"runtime_s={float(report.get('runtime_s') or 0.0):.6f} "
        f"objective_value={report.get('objective_value')} "
        f"constraint_penalty={report.get('constraint_penalty')} "
        f"violation_count={report.get('violation_count', 0)}",
        flush=True,
    )


def _design_event_summary(designer_step):
    report = designer_step.get("solver_report") or {}
    if not report:
        return designer_step["design_status"]
    return (
        f"{designer_step['design_status']}; "
        f"requested={report.get('solver_requested')}; "
        f"used={report.get('solver_used')}; "
        f"default_fallback={report.get('default_fallback_used')}; "
        f"runtime_s={float(report.get('runtime_s') or 0.0):.6f}; "
        f"objective={report.get('objective_value')}; "
        f"violations={report.get('violation_count', 0)}"
    )


def _validate_flare_fidelity(
    flp_ir,
    task_contract,
    semantic_intent,
    scene_context,
):
    """Compare Designer-derived FLARE state with the frozen Semantic Intent."""

    if not semantic_intent:
        return {
            "schema_version": "flare-fidelity-validation-v1",
            "status": "NOT_EVALUATED",
            "violations": [],
            "violation_count": 0,
        }
    authoritative = scene_context.get("authoritative_context") or {}
    expected_contract = TaskContractCompiler().compile(
        {"intent_type": semantic_intent.get("intent_type")},
        semantic_intent,
        authoritative,
    ).to_dict()
    generated_instance_ids = (
        task_contract.get("task_type") == "Generate"
        and not expected_contract.get("expected_instance_ids")
    )
    violations = []

    def add(code, message, measured, target, operation, instance_ids=None):
        violations.append(
            {
                "code": code,
                "violation_type": "flare_fidelity",
                "message": message,
                "instance_ids": list(instance_ids or []),
                "obstacle_id": None,
                "measured_value": copy.deepcopy(measured),
                "target_value": copy.deepcopy(target),
                "minimum_repair_displacement": None,
                "allowed_repair_operations": [operation],
                "preserve_instance_ids": copy.deepcopy(
                    expected_contract.get("preserve_instance_ids") or []
                ),
                "repair_target": (
                    "instance_binding"
                    if "BINDING" in code
                    else (
                        "task_contract"
                        if "CONTRACT" in code or "SCOPE" in code
                        else "flp_ir"
                    )
                ),
                "system_derived_error": True,
            }
        )

    actual_task_type = task_contract.get("task_type")
    expected_task_type = expected_contract.get("task_type")
    if actual_task_type != expected_task_type:
        add(
            "FLARE_CONTRACT_TASK_TYPE_MISMATCH",
            "FLARE task type does not match the frozen Semantic Intent.",
            actual_task_type,
            expected_task_type,
            "recompile_task_contract",
        )

    instance_scope_fields = (
        ()
        if generated_instance_ids
        else ("expected_instance_ids", "movable_instance_ids", "preserve_instance_ids")
    )
    for field in instance_scope_fields:
        actual = sorted(set(task_contract.get(field) or []))
        expected = sorted(set(expected_contract.get(field) or []))
        if actual != expected:
            add(
                (
                    "FLARE_CONTRACT_INSTANCE_SET_MISMATCH"
                    if field == "expected_instance_ids"
                    else f"FLARE_{field.upper()}_SCOPE_MISMATCH"
                ),
                f"FLARE {field} differs from the frozen Semantic Intent.",
                actual,
                expected,
                (
                    "rebind_instances"
                    if field == "expected_instance_ids"
                    else "recompile_task_contract_scope"
                ),
                instance_ids=sorted(set(actual) ^ set(expected)),
            )

    actual_bindings = {
        str(item.get("instance_id")): str(item.get("model_asset_id"))
        for item in (flp_ir or {}).get("equipment_instances") or []
        if item.get("instance_id") and item.get("model_asset_id")
    }
    if generated_instance_ids:
        declared_assets = sorted(
            str(item.get("model_asset_id"))
            for item in semantic_intent.get("asset_mentions") or []
            if isinstance(item, dict) and item.get("model_asset_id")
            for _ in range(max(1, int(item.get("quantity") or 1)))
        )
        actual_assets = sorted(actual_bindings.values())
        bindings_match = actual_assets == declared_assets
        comparable_expected = declared_assets
        differing = sorted(actual_bindings) if not bindings_match else []
    else:
        expected_bindings = _expected_flare_instance_bindings(
            expected_contract,
            semantic_intent,
            scene_context,
        )
        comparable_expected = {
            instance_id: asset_id
            for instance_id, asset_id in expected_bindings.items()
            if instance_id in set(expected_contract.get("expected_instance_ids") or [])
        }
        bindings_match = actual_bindings == comparable_expected
        differing = sorted(
            instance_id
            for instance_id in set(actual_bindings) | set(comparable_expected)
            if actual_bindings.get(instance_id)
            != comparable_expected.get(instance_id)
        )
    if not bindings_match:
        add(
            "FLARE_INSTANCE_BINDING_MISMATCH",
            "FLARE instance-to-asset binding differs from the frozen asset declarations.",
            (
                sorted(actual_bindings.values())
                if generated_instance_ids
                else actual_bindings
            ),
            comparable_expected,
            "rebind_instances",
            instance_ids=differing,
        )

    expected_constraints = sorted(
        {
            _canonical_flare_constraint_type(item.get("constraint_type"))
            for item in semantic_intent.get("constraints") or []
            if isinstance(item, dict)
            and item.get("constraint_type")
            and str(item.get("severity") or "hard").casefold() == "hard"
        }
    )
    actual_constraints = sorted(
        {
            _canonical_flare_constraint_type(item.get("constraint_type"))
            for item in ((flp_ir or {}).get("constraints") or {}).get("hard", [])
            if isinstance(item, dict) and item.get("constraint_type")
        }
    )
    if not set(expected_constraints).issubset(actual_constraints):
        add(
            "FLARE_CONSTRAINT_SET_MISMATCH",
            "FLARE omits or changes a hard constraint from frozen Semantic Intent.",
            actual_constraints,
            expected_constraints,
            "recompile_flare_constraints",
        )

    expected_objectives = sorted(
        {
            _canonical_flare_objective_type(item.get("objective_type"))
            for item in semantic_intent.get("objectives") or []
            if isinstance(item, dict) and item.get("objective_type")
        }
    )
    actual_objectives = sorted(
        {
            _canonical_flare_objective_type(item.get("objective_type"))
            for item in (flp_ir or {}).get("objectives") or []
            if isinstance(item, dict) and item.get("objective_type")
        }
    )
    if not set(expected_objectives).issubset(actual_objectives):
        add(
            "FLARE_OBJECTIVE_SET_MISMATCH",
            "FLARE omits or changes an objective from frozen Semantic Intent.",
            actual_objectives,
            expected_objectives,
            "recompile_flare_objectives",
        )

    expected_flows, unresolved_semantic_flows = _expected_flare_flow_pairs(
        semantic_intent,
        flp_ir,
        scene_context,
    )
    actual_flows = {
        (
            str(item.get("from_instance_id")),
            str(item.get("to_instance_id")),
        )
        for item in (flp_ir or {}).get("flow_relations") or []
        if isinstance(item, dict)
        and item.get("from_instance_id")
        and item.get("to_instance_id")
    }
    if unresolved_semantic_flows or expected_flows != actual_flows:
        add(
            "FLARE_PROCESS_RELATION_MISMATCH",
            "FLARE process relations differ from frozen Semantic Intent.",
            sorted(actual_flows),
            (
                {
                    "resolved_pairs": sorted(expected_flows),
                    "unresolved_semantic_endpoints": unresolved_semantic_flows,
                }
                if unresolved_semantic_flows
                else sorted(expected_flows)
            ),
            "recompile_flare_process_relations",
        )

    return {
        "schema_version": "flare-fidelity-validation-v1",
        "status": "VALID" if not violations else "REPAIR_REQUIRED",
        "violations": violations,
        "violation_count": len(violations),
        "expected_task_type": expected_task_type,
    }


def _expected_flare_instance_bindings(
    expected_contract,
    semantic_intent,
    scene_context,
):
    bindings = {}
    directory = (
        semantic_intent.get("authoritative_instance_directory")
        or scene_context.get("instance_directory")
        or []
    )
    for item in directory:
        if isinstance(item, dict) and item.get("instance_id") and item.get(
            "model_asset_id"
        ):
            bindings[str(item["instance_id"])] = str(item["model_asset_id"])
    base_layout = expected_contract.get("base_layout") or {}
    for item in base_layout.get("placements") or []:
        if isinstance(item, dict) and item.get("instance_id") and item.get(
            "model_asset_id"
        ):
            bindings[str(item["instance_id"])] = str(item["model_asset_id"])
    for operation in expected_contract.get("edit_operations") or []:
        if not isinstance(operation, dict):
            continue
        operation_type = str(
            operation.get("op_type")
            or operation.get("operation_type")
            or operation.get("type")
            or ""
        ).casefold()
        instance_id = str(
            operation.get("instance_id")
            or operation.get("target_instance_id")
            or ""
        )
        if not instance_id:
            continue
        if operation_type in {"delete", "remove"}:
            bindings.pop(instance_id, None)
        elif operation_type in {"add", "replace"}:
            replacement_asset_id = operation.get("new_model_asset_id")
            if replacement_asset_id:
                bindings[instance_id] = str(replacement_asset_id)
    return bindings


def _canonical_flare_constraint_type(value):
    """Map only documented compiler aliases before FLARE set comparison."""

    constraint_type = str(value or "").strip()
    return CONSTRAINT_SYNONYMS.get(constraint_type, constraint_type)


def _canonical_flare_objective_type(value):
    objective_type = str(value or "").strip()
    return OBJECTIVE_SYNONYMS.get(objective_type, objective_type)


def _positive_quantity_index(value):
    if isinstance(value, bool):
        return None
    try:
        quantity_index = int(value)
    except (TypeError, ValueError):
        return None
    return quantity_index if quantity_index > 0 else None


def _expected_flare_flow_pairs(semantic_intent, flp_ir, scene_context):
    """Resolve source semantic route endpoints without reading candidate flows.

    New-layout semantic intent is grounded to model identities and declared
    ordinals before the FLP compiler assigns instance IDs.  The independent
    compiled instance directory provides the reversible bridge between those
    two representations.  A missing or ambiguous bridge remains a fidelity
    error instead of being silently dropped.
    """

    directory = []
    for candidate in (
        (scene_context or {}).get("planned_edit_instance_directory"),
        semantic_intent.get("authoritative_instance_directory"),
        (scene_context or {}).get("instance_directory"),
        (flp_ir or {}).get("equipment_instances"),
    ):
        if not isinstance(candidate, list) or not candidate:
            continue
        directory = [item for item in candidate if isinstance(item, dict)]
        if directory:
            break

    by_model_index = {}
    by_model = {}
    instance_ids = set()
    for item in directory:
        instance_id = str(item.get("instance_id") or "").strip()
        model_asset_id = str(item.get("model_asset_id") or "").strip()
        quantity_index = _positive_quantity_index(item.get("quantity_index"))
        if not instance_id or not model_asset_id:
            continue
        instance_ids.add(instance_id)
        by_model.setdefault(model_asset_id, []).append(instance_id)
        if quantity_index is None:
            continue
        key = (model_asset_id, quantity_index)
        # A repeated model/ordinal key cannot provide a trustworthy bridge.
        by_model_index[key] = (
            instance_id if key not in by_model_index else None
        )

    def resolve_endpoint(relation, prefix):
        site_name = relation.get(f"{prefix}_site_feature_name")
        if site_name:
            return resolve_facility_ref((((flp_ir or {}).get("site_ref") or {}).get("site_snapshot") or {}).get("site_features"), site_name)
        explicit_instance_id = str(
            relation.get(f"{prefix}_instance_id") or ""
        ).strip()
        if explicit_instance_id:
            feature_ids={f.get("site_feature_id") for f in (((flp_ir or {}).get("site_ref") or {}).get("site_snapshot") or {}).get("site_features",[])}
            return (
                explicit_instance_id
                if explicit_instance_id in instance_ids or explicit_instance_id in feature_ids
                else None
            )
        model_asset_id = str(
            relation.get(f"{prefix}_model_asset_id") or ""
        ).strip()
        if not model_asset_id:
            return None
        quantity_index = _positive_quantity_index(
            relation.get(f"{prefix}_quantity_index")
        )
        if quantity_index is not None:
            return by_model_index.get((model_asset_id, quantity_index))
        candidates = by_model.get(model_asset_id) or []
        return candidates[0] if len(candidates) == 1 else None

    from tool.context.conversation_deltas import service_duty_transfers
    source_directory = dict(equipment_instances=[dict(instance_id=i["instance_id"],
        role=i.get("stable_alias") or i.get("role") or "") for i in directory])
    source_text = str((scene_context or {}).get("current_user_input") or "").rsplit("Current user request:\n",1)[-1]
    try:
        duty_map = dict(service_duty_transfers(source_text, source_directory, dict(operations=[])))
    except ValueError:
        # An unresolved directive cannot justify rewriting expected endpoints.
        duty_map = {}
    pairs = set()
    unresolved = []
    for relation_index, relation in enumerate(
        semantic_intent.get("process_relations") or [], start=1
    ):
        if not isinstance(relation, dict):
            continue
        from_instance_id = resolve_endpoint(relation, "from")
        to_instance_id = resolve_endpoint(relation, "to")
        if relation.get("flow_type") == "tool_service":
            from_instance_id = duty_map.get(from_instance_id, from_instance_id)
            to_instance_id = duty_map.get(to_instance_id, to_instance_id)
        if from_instance_id and to_instance_id:
            pairs.add((from_instance_id, to_instance_id))
            continue
        unresolved.append(
            {
                "relation_index": relation_index,
                "from_model_asset_id": relation.get("from_model_asset_id"),
                "from_quantity_index": relation.get("from_quantity_index"),
                "to_model_asset_id": relation.get("to_model_asset_id"),
                "to_quantity_index": relation.get("to_quantity_index"),
            }
        )
    return pairs, unresolved


def _apply_validator_flare_repairs(flp_ir, semantic_intent, feedback):
    """Apply only Validator-requested FLARE fields from frozen semantics."""

    if not (
        isinstance(feedback, dict)
        and feedback.get("source") == "validator"
        and feedback.get("flare_repair_required")
    ):
        return flp_ir
    result = copy.deepcopy(flp_ir)
    constraints = result.setdefault("constraints", {}).setdefault("hard", [])
    existing_constraints = {
        item.get("constraint_type")
        for item in constraints
        if isinstance(item, dict)
    }
    for item in semantic_intent.get("constraints") or []:
        constraint_type = _canonical_flare_constraint_type(
            item.get("constraint_type") if isinstance(item, dict) else None
        )
        if (
            not constraint_type
            or str(item.get("severity") or "hard").casefold() != "hard"
            or constraint_type in existing_constraints
        ):
            continue
        constraints.append(
            {
                "constraint_id": f"constraint_{len(constraints) + 1:03d}",
                "constraint_type": constraint_type,
                "target_refs": [
                    {"target_type": "all_equipment", "target_id": "all"}
                ],
                "parameters": {},
                "source": "user_input",
                "evidence_refs": ["frozen_semantic_intent"],
            }
        )
        existing_constraints.add(constraint_type)

    objectives = result.setdefault("objectives", [])
    existing_objectives = {
        item.get("objective_type")
        for item in objectives
        if isinstance(item, dict)
    }
    for item in semantic_intent.get("objectives") or []:
        objective_type = _canonical_flare_objective_type(
            item.get("objective_type") if isinstance(item, dict) else None
        )
        if not objective_type or objective_type in existing_objectives:
            continue
        objectives.append(
            {
                "objective_id": f"objective_{len(objectives) + 1:03d}",
                "objective_type": objective_type,
                "sense": "minimize",
                "weight": 1.0,
                "parameters": {},
                "source": "user_input",
                "evidence_refs": ["frozen_semantic_intent"],
            }
        )
        existing_objectives.add(objective_type)
    return result


def _validation_recovery_feedback(
    report,
    typed_violations,
    task_contract,
    revision,
):
    incomplete = not task_contract.get("complete", True)
    flare_violations = [
        item
        for item in typed_violations or []
        if str(item.get("code") or "").startswith("FLARE_")
    ]
    non_typed_errors = [
        item
        for item in report.get("errors") or []
        if not item.get("code")
    ]
    if incomplete:
        disposition = "clarification_required"
    elif flare_violations:
        disposition = "auto_recovery"
    elif typed_violations:
        if (
            task_contract.get("task_type") == "DirectiveEdit"
            and task_contract.get("edit_relaxation_policy")
            != "nearest_feasible"
        ):
            disposition = "edit_conflict"
        else:
            disposition = "auto_recovery"
    elif non_typed_errors:
        legacy = repair_or_clarify(None, {"errors": non_typed_errors})
        disposition = legacy.get("disposition") or "fatal"
    else:
        disposition = "none"
    actions = []
    for index, violation in enumerate(typed_violations or [], start=1):
        allowed = list(violation.get("allowed_repair_operations") or [])
        actions.append(
            {
                "action_id": f"validation_recovery_{index:03d}",
                "action_type": (
                    allowed[0]
                    if allowed
                    else "canonical_ga_recovery"
                ),
                "target_agent": "DesignerAgent",
                "violation_code": violation.get("code"),
                "instance_ids": copy.deepcopy(
                    violation.get("instance_ids") or []
                ),
                "obstacle_id": violation.get("obstacle_id"),
                "minimum_repair_displacement": copy.deepcopy(
                    violation.get("minimum_repair_displacement")
                ),
                "allowed_repair_operations": allowed,
                "preserve_instance_ids": copy.deepcopy(
                    violation.get("preserve_instance_ids")
                    or task_contract.get("preserve_instance_ids")
                    or []
                ),
            }
        )
    clarification_questions = [
        f"Please provide {field_path}."
        for field_path in task_contract.get("missing_fields") or []
    ]
    return {
        "schema_version": "validation-driven-recovery-feedback-v1",
        "feedback_type": "validation_driven_recovery",
        "trigger_source": "validator_typed_violation",
        "source": "validator",
        "target_agent": "DesignerAgent",
        "source_revision_id": f"validation_v{revision}",
        "disposition": disposition,
        "should_revise": disposition == "auto_recovery",
        "can_auto_recover": disposition == "auto_recovery",
        "flare_repair_required": bool(flare_violations),
        "requires_user_clarification": disposition == "clarification_required",
        "fatal": disposition == "fatal",
        "actions": actions,
        "recovery_actions": copy.deepcopy(actions),
        "issues": copy.deepcopy(typed_violations),
        "typed_violations": copy.deepcopy(typed_violations),
        "clarification_questions": clarification_questions,
        "task_contract": {
            "task_type": task_contract.get("task_type"),
            "controlled_violation": copy.deepcopy(
                task_contract.get("controlled_violation")
            ),
            "movable_instance_ids": copy.deepcopy(
                task_contract.get("movable_instance_ids") or []
            ),
            "preserve_instance_ids": copy.deepcopy(
                task_contract.get("preserve_instance_ids") or []
            ),
        },
    }


def _viewport_precheck_violations(precheck):
    result = []
    operation_by_code = {
        "VIEWPORT_MISSING_INSTANCE": "restore_missing_instance",
        "VIEWPORT_UNEXPECTED_INSTANCE": "remove_unexpected_instance",
        "VIEWPORT_DUPLICATE_INSTANCE": "remove_duplicate_instance",
        "VIEWPORT_MODEL_MAPPING_INVALID": "restore_model_mapping",
        "VIEWPORT_POSITION_DATA_INVALID": "restore_missing_position",
        "VIEWPORT_CAMERA_VISIBILITY_INVALID": "project_to_camera_visible_region",
    }
    for item in (precheck or {}).get("errors") or []:
        code = item.get("code") or "VIEWPORT_PRECHECK_FAILED"
        result.append(
            {
                "schema_version": "typed-violation-v1",
                "code": code,
                "instance_ids": copy.deepcopy(
                    item.get("instance_ids")
                    or [
                        value.get("instance_id")
                        for value in item.get("instances") or []
                        if value.get("instance_id")
                    ]
                ),
                "obstacle_id": None,
                "measured_value": copy.deepcopy(item),
                "target_value": "renderable",
                "minimum_repair_displacement": None,
                "allowed_repair_operations": [
                    operation_by_code.get(code, "global_feasibility_solve")
                ],
                "preserve_instance_ids": [],
                "details": {"validation_stage": "ViewportPrecheck"},
            }
        )
    return result


def _typed_violation_as_validation_error(violation):
    code = str(violation.get("code") or "CANONICAL_CONSTRAINT_VIOLATION")
    instance_ids = copy.deepcopy(violation.get("instance_ids") or [])
    return {
        "error_type": code.casefold(),
        "code": code,
        "message": (
            f"{code} involves {', '.join(instance_ids) or 'the candidate layout'}"
            + (
                f" and obstacle {violation.get('obstacle_id')}"
                if violation.get("obstacle_id")
                else ""
            )
            + "."
        ),
        "field_path": "rough_layout.placements",
        "instance_ids": instance_ids,
        "obstacle_id": violation.get("obstacle_id"),
        "measured_value": copy.deepcopy(violation.get("measured_value")),
        "target_value": copy.deepcopy(violation.get("target_value")),
        "minimum_repair_displacement": copy.deepcopy(
            violation.get("minimum_repair_displacement")
        ),
        "allowed_repair_operations": copy.deepcopy(
            violation.get("allowed_repair_operations") or []
        ),
        "preserve_instance_ids": copy.deepcopy(
            violation.get("preserve_instance_ids") or []
        ),
    }


def _validate_layout_candidate(rough_layout, flp_ir):
    if rough_layout is None:
        issues = []
        if flp_ir and flp_ir.get("equipment_instances") and not flp_ir.get("unresolved_items"):
            issues.append(
                {
                    "error_type": "solver_no_layout",
                    "message": "Solver-ready FLP-IR did not produce a layout candidate.",
                    "field_path": "rough_layout",
                    "instance_ids": [],
                    "severity": "hard",
                    "repairability": "fatal",
                }
            )
        return {
            "checked": False,
            "status": "NOT_AVAILABLE" if not issues else "INVALID",
            "issues": issues,
            "violation_count": len(issues),
        }

    issue_map = {
        "overlap": ("overlap", "Move or swap the affected equipment and re-solve."),
        "boundary": ("outside_boundary", "Project the equipment back into a placement region and re-solve."),
        "non_placeable_zone": ("blocked_area_overlap", "Move equipment away from walls, aisles, and blocked areas."),
        "clearance": ("safety_clearance_violation", "Increase spacing around the affected equipment group."),
    }
    issues = []
    for violation in layout_violations(rough_layout, flp_ir):
        error_type, action = issue_map.get(
            violation.get("violation_type"),
            (violation.get("violation_type") or "layout_violation", "Re-solve the layout."),
        )
        issues.append(
            {
                "error_type": error_type,
                "message": action,
                "field_path": "rough_layout.placements",
                "instance_ids": list(violation.get("instances") or []),
                "zone_id": violation.get("zone_id"),
                "severity": "hard",
                "repairability": "auto_repair",
            }
        )

    expected_ids = {item.get("instance_id") for item in (flp_ir or {}).get("equipment_instances", [])}
    placements = rough_layout.get("placements") or []
    actual_ids = {item.get("instance_id") for item in placements}
    for instance_id in sorted(expected_ids - actual_ids):
        issues.append(
            {
                "error_type": "missing_layout_instance",
                "message": f"Layout is missing required instance {instance_id}.",
                "field_path": "rough_layout.placements",
                "instance_ids": [instance_id],
                "severity": "hard",
                "repairability": "auto_repair",
            }
        )

    instances_by_id = {
        item.get("instance_id"): item
        for item in (flp_ir or {}).get("equipment_instances", [])
    }
    for placement in placements:
        instance = instances_by_id.get(placement.get("instance_id"))
        if not instance:
            continue
        allowed = instance.get("allowed_orientations_deg") or []
        if not allowed:
            continue
        yaw = float((placement.get("rotation") or {}).get("yaw_deg", placement.get("yaw_deg") or 0.0))
        if not any(abs((yaw - float(value)) % 360.0) <= 1e-6 for value in allowed):
            issues.append(
                {
                    "error_type": "invalid_orientation",
                    "message": f"Instance {placement.get('instance_id')} uses disallowed yaw {yaw}.",
                    "field_path": "rough_layout.placements.rotation",
                    "instance_ids": [placement.get("instance_id")],
                    "severity": "hard",
                    "repairability": "auto_repair",
                }
            )

    issues.extend(_relation_constraint_issues(placements, flp_ir))
    return {
        "checked": True,
        "status": "VALID" if not issues else "INVALID",
        "issues": issues,
        "violation_count": len(issues),
    }


def _relation_constraint_issues(placements, flp_ir):
    placement_by_id = {item.get("instance_id"): item for item in placements}
    issues = []
    relation_types = {"adjacency", "near", "far_from", "forbidden_adjacency", "non_adjacency"}
    for constraint in (flp_ir or {}).get("constraints", {}).get("hard", []):
        constraint_type = constraint.get("constraint_type")
        if constraint_type not in relation_types:
            continue
        instance_ids = _constraint_instance_ids(constraint)
        if len(instance_ids) < 2 or any(instance_id not in placement_by_id for instance_id in instance_ids[:2]):
            continue
        left = placement_by_id[instance_ids[0]]
        right = placement_by_id[instance_ids[1]]
        distance = _footprint_edge_distance(left["footprint"], right["footprint"])
        parameters = constraint.get("parameters") or {}
        if constraint_type in {"adjacency", "near"}:
            threshold = _first_numeric_parameter(
                parameters,
                "max_distance_m",
                "adjacency_distance_m",
                "threshold_m",
            )
            if threshold is None:
                issues.append(
                    {
                        "error_type": "missing_required_field",
                        "message": f"Hard {constraint_type} constraint requires max_distance_m.",
                        "field_path": f"constraints.hard.{constraint.get('constraint_id')}.parameters",
                        "instance_ids": instance_ids[:2],
                        "severity": "hard",
                        "repairability": "clarification_required",
                    }
                )
            elif distance > threshold + GEOMETRY_EPSILON_M:
                issues.append(
                    {
                        "error_type": "required_adjacency_violation",
                        "message": f"Required relation distance {distance:.3f}m exceeds {threshold:.3f}m.",
                        "field_path": "rough_layout.placements",
                        "instance_ids": instance_ids[:2],
                        "severity": "hard",
                        "repairability": "auto_repair",
                        "measured_distance_m": distance,
                        "required_distance_m": threshold,
                    }
                )
        else:
            threshold = _first_numeric_parameter(
                parameters,
                "min_distance_m",
                "minimum_separation_m",
                "threshold_m",
            )
            if threshold is None:
                issues.append(
                    {
                        "error_type": "missing_required_field",
                        "message": f"Hard {constraint_type} constraint requires min_distance_m.",
                        "field_path": f"constraints.hard.{constraint.get('constraint_id')}.parameters",
                        "instance_ids": instance_ids[:2],
                        "severity": "hard",
                        "repairability": "clarification_required",
                    }
                )
            elif distance + GEOMETRY_EPSILON_M < threshold:
                issues.append(
                    {
                        "error_type": "forbidden_adjacency_violation",
                        "message": f"Forbidden relation distance {distance:.3f}m is below {threshold:.3f}m.",
                        "field_path": "rough_layout.placements",
                        "instance_ids": instance_ids[:2],
                        "severity": "hard",
                        "repairability": "auto_repair",
                        "measured_distance_m": distance,
                        "required_distance_m": threshold,
                    }
                )
    return issues


def _constraint_instance_ids(constraint):
    instance_ids = [
        item.get("target_id")
        for item in constraint.get("target_refs") or []
        if item.get("target_type") == "equipment_instance" and item.get("target_id")
    ]
    parameters = constraint.get("parameters") or {}
    for key in ("instance_ids", "equipment_instance_ids"):
        for instance_id in parameters.get(key) or []:
            _append_unique(instance_ids, instance_id)
    for key in ("from_instance_id", "to_instance_id", "left_instance_id", "right_instance_id"):
        if parameters.get(key):
            _append_unique(instance_ids, parameters[key])
    return instance_ids


def _first_numeric_parameter(parameters, *keys):
    for key in keys:
        value = _optional_float(parameters.get(key))
        if value is not None:
            return value
    return None


def _footprint_edge_distance(left, right):
    x_gap = max(right["min_x"] - left["max_x"], left["min_x"] - right["max_x"], 0.0)
    z_gap = max(right["min_z"] - left["max_z"], left["min_z"] - right["max_z"], 0.0)
    return (x_gap * x_gap + z_gap * z_gap) ** 0.5


def _evaluate_layout(rough_layout, *, solver_report=None, flp_ir=None):
    placements = rough_layout["placements"]
    site = rough_layout["site"]
    total_area = 0.0
    for placement in placements:
        footprint = placement["footprint"]
        width_m = _optional_float(footprint.get("width_m"))
        depth_m = _optional_float(footprint.get("depth_m"))
        if width_m is None:
            width_m = float(footprint["max_x"]) - float(footprint["min_x"])
        if depth_m is None:
            depth_m = float(footprint["max_z"]) - float(footprint["min_z"])
        total_area += max(0.0, width_m) * max(0.0, depth_m)
    site_area = site["width_m"] * site["depth_m"]
    violations = layout_violations(rough_layout, flp_ir)
    solver_report = solver_report or {}
    if placements:
        min_x = min(item["footprint"]["min_x"] for item in placements)
        max_x = max(item["footprint"]["max_x"] for item in placements)
        min_z = min(item["footprint"]["min_z"] for item in placements)
        max_z = max(item["footprint"]["max_z"] for item in placements)
        envelope_area = max(GEOMETRY_EPSILON_M, (max_x - min_x) * (max_z - min_z))
    else:
        envelope_area = 0.0
    compactness = total_area / envelope_area if envelope_area > 0 else 0.0
    violation_count = len(violations)
    relation_quality = _relationship_quality(placements, flp_ir)
    grouping_score = _flow_grouping_score(placements, site, flp_ir)
    clearance_margin = _minimum_clearance_margin(placements, flp_ir)
    blocked_instance_ids = {
        instance_id
        for item in violations
        if item.get("violation_type") == "non_placeable_zone"
        for instance_id in item.get("instances") or []
    }
    circulation_access_ratio = (
        (len(placements) - len(blocked_instance_ids)) / len(placements) if placements else 0.0
    )
    quality_components = [compactness, circulation_access_ratio, grouping_score]
    if relation_quality is not None:
        quality_components.append(relation_quality)
    quality_score = sum(quality_components) / len(quality_components) if quality_components else 0.0
    if violation_count:
        quality_score = 0.0
    return {
        "placed_instances": len(placements),
        "site_utilization": total_area / site_area if site_area > 0 else 0.0,
        "no_overlap": _no_overlap(placements),
        "inside_boundary": _inside_boundary(placements, site),
        "objective_coverage": "solver_backed" if solver_report else "rough_layout_only",
        "objective_value": solver_report.get("objective_value", objective_value(rough_layout, flp_ir or {})),
        "flow_distance": objective_value(rough_layout, flp_ir or {}),
        "objective_values": objective_vector(rough_layout, flp_ir or {}),
        "solver_status": solver_report.get("status"),
        "solver": solver_report.get("solver"),
        "solver_used": solver_report.get("solver_used"),
        "solver_requested": solver_report.get("solver_requested"),
        "solver_selection_source": solver_report.get("selection_source"),
        "default_fallback_used": solver_report.get("default_fallback_used"),
        "constraint_penalty": solver_report.get("constraint_penalty"),
        "violation_count": solver_report.get("violation_count", violation_count),
        "solver_runtime_s": solver_report.get("runtime_s"),
        "layout_compactness": compactness,
        "circulation_access_ratio": circulation_access_ratio,
        "equipment_relationship_satisfaction": relation_quality,
        "equipment_grouping_score": grouping_score,
        "minimum_clearance_margin_m": clearance_margin,
        "quality_score": max(0.0, min(1.0, quality_score)),
        "gap": solver_report.get("gap"),
        "overlap_violation_count": sum(1 for item in violations if item["violation_type"] == "overlap"),
        "boundary_violation_count": sum(1 for item in violations if item["violation_type"] == "boundary"),
        "non_placeable_violation_count": sum(1 for item in violations if item["violation_type"] == "non_placeable_zone"),
        "clearance_violation_count": sum(1 for item in violations if item["violation_type"] == "clearance"),
    }


def _relationship_quality(placements, flp_ir):
    placement_by_id = {item.get("instance_id"): item for item in placements}
    checks = []
    constraints = list((flp_ir or {}).get("constraints", {}).get("hard", []))
    constraints.extend((flp_ir or {}).get("constraints", {}).get("soft", []))
    for constraint in constraints:
        constraint_type = constraint.get("constraint_type")
        if constraint_type not in {"adjacency", "near", "far_from", "forbidden_adjacency", "non_adjacency"}:
            continue
        instance_ids = _constraint_instance_ids(constraint)
        if len(instance_ids) < 2 or any(instance_id not in placement_by_id for instance_id in instance_ids[:2]):
            continue
        distance = _footprint_edge_distance(
            placement_by_id[instance_ids[0]]["footprint"],
            placement_by_id[instance_ids[1]]["footprint"],
        )
        parameters = constraint.get("parameters") or {}
        if constraint_type in {"adjacency", "near"}:
            threshold = _first_numeric_parameter(
                parameters,
                "max_distance_m",
                "adjacency_distance_m",
                "threshold_m",
            )
            if threshold is not None:
                checks.append(distance <= threshold + GEOMETRY_EPSILON_M)
        else:
            threshold = _first_numeric_parameter(
                parameters,
                "min_distance_m",
                "minimum_separation_m",
                "threshold_m",
            )
            if threshold is not None:
                checks.append(distance + GEOMETRY_EPSILON_M >= threshold)
    return sum(1 for item in checks if item) / len(checks) if checks else None


def _flow_grouping_score(placements, site, flp_ir):
    placement_by_id = {item.get("instance_id"): item for item in placements}
    diagonal = max(
        GEOMETRY_EPSILON_M,
        (float(site.get("width_m") or 0.0) ** 2 + float(site.get("depth_m") or 0.0) ** 2) ** 0.5,
    )
    weighted_distance = 0.0
    total_weight = 0.0
    for flow in (flp_ir or {}).get("flow_relations", []):
        left = placement_by_id.get(flow.get("from_instance_id"))
        right = placement_by_id.get(flow.get("to_instance_id"))
        if not left or not right:
            continue
        weight = float(flow.get("flow_weight") or 1.0)
        distance = _center_distance(left, right)
        weighted_distance += weight * distance
        total_weight += weight
    if total_weight <= 0:
        return 1.0
    normalized = weighted_distance / total_weight / diagonal
    return max(0.0, min(1.0, 1.0 - normalized))


def _minimum_clearance_margin(placements, flp_ir):
    clearance_by_instance = {}
    for constraint in (flp_ir or {}).get("constraints", {}).get("hard", []):
        if constraint.get("constraint_type") not in {"minimum_clearance", "safety_clearance"}:
            continue
        parameters = constraint.get("parameters") or {}
        clearance_by_instance.update(parameters.get("clearance_by_instance") or {})
    if not clearance_by_instance:
        return None
    margins = []
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            required = max(
                float(clearance_by_instance.get(left.get("instance_id")) or 0.0),
                float(clearance_by_instance.get(right.get("instance_id")) or 0.0),
            )
            if required > 0:
                margins.append(_footprint_edge_distance(left["footprint"], right["footprint"]) - required)
    return min(margins) if margins else None


def _center_distance(left, right):
    left_position = left.get("position") or {}
    right_position = right.get("position") or {}
    dx = float(left_position.get("x_m") or left.get("x_m") or 0.0) - float(
        right_position.get("x_m") or right.get("x_m") or 0.0
    )
    dz = float(left_position.get("z_m") or left.get("z_m") or 0.0) - float(
        right_position.get("z_m") or right.get("z_m") or 0.0
    )
    return (dx * dx + dz * dz) ** 0.5


def _compare_evaluations(current_metrics, previous_evaluation):
    previous_metrics = (previous_evaluation or {}).get("metrics") if isinstance(previous_evaluation, dict) else None
    previous_metrics = previous_metrics or {}
    if not previous_metrics or not current_metrics:
        return {
            "baseline_available": False,
            "objective_delta": None,
            "relative_objective_improvement": None,
            "utilization_delta": None,
            "compactness_delta": None,
            "violation_delta": None,
            "improved": None,
        }
    previous_objective = _optional_float(previous_metrics.get("objective_value"))
    current_objective = _optional_float(current_metrics.get("objective_value"))
    objective_delta = None
    relative_improvement = None
    if previous_objective is not None and current_objective is not None:
        objective_delta = previous_objective - current_objective
        relative_improvement = objective_delta / max(abs(previous_objective), GEOMETRY_EPSILON_M)
    violation_delta = int(previous_metrics.get("violation_count") or 0) - int(
        current_metrics.get("violation_count") or 0
    )
    improved = violation_delta > 0 or (
        violation_delta == 0 and objective_delta is not None and objective_delta > 1e-6
    )
    return {
        "baseline_available": True,
        "objective_delta": objective_delta,
        "relative_objective_improvement": relative_improvement,
        "utilization_delta": float(current_metrics.get("site_utilization") or 0.0)
        - float(previous_metrics.get("site_utilization") or 0.0),
        "compactness_delta": float(current_metrics.get("layout_compactness") or 0.0)
        - float(previous_metrics.get("layout_compactness") or 0.0),
        "violation_delta": violation_delta,
        "improved": improved,
    }


def _evaluator_assisted_recovery_scope(task_contract, feedback):
    """Limit disclosed recovery to edit targets named by current violations."""

    original_movable = list(task_contract.get("movable_instance_ids") or [])
    original_preserve = list(task_contract.get("preserve_instance_ids") or [])
    violating_ids = {
        str(instance_id)
        for issue in (
            (feedback or {}).get("typed_violations")
            or (feedback or {}).get("issues")
            or []
        )
        if isinstance(issue, dict)
        for instance_id in (issue.get("instance_ids") or [])
        if instance_id
    }
    movable = [
        instance_id
        for instance_id in original_movable
        if instance_id not in set(original_preserve)
        and str(instance_id) in violating_ids
    ]
    movable_set = set(movable)
    preserve = list(dict.fromkeys([
        *original_preserve,
        *[
            instance_id
            for instance_id in original_movable
            if instance_id not in movable_set
        ],
    ]))
    return movable, preserve


def _project_clearance_recovery_seed(layout, violations, movable_instance_ids):
    """Apply typed one-target clearance translations before bounded GA search."""

    movable = set(movable_instance_ids or [])
    if not layout or not movable:
        return None
    result = copy.deepcopy(layout)
    placements = {
        str(item.get("instance_id") or ""): item
        for item in result.get("placements") or []
        if item.get("instance_id")
    }
    for issue in violations or []:
        if not isinstance(issue, dict) or issue.get("code") != "CLEARANCE_DEFICIT":
            continue
        ids = [str(value) for value in issue.get("instance_ids") or []]
        targets = [value for value in ids if value in movable and value in placements]
        fixed = [value for value in ids if value not in movable and value in placements]
        if len(targets) != 1 or len(fixed) != 1:
            continue
        displacement = issue.get("minimum_repair_displacement") or {}
        try:
            dx = abs(float(displacement.get("dx_m") or 0.0))
            dz = abs(float(displacement.get("dz_m") or 0.0))
        except (TypeError, ValueError):
            continue
        axis = "x_m" if dx >= dz and dx > 0.0 else ("z_m" if dz > 0.0 else None)
        if axis is None:
            continue
        target = placements[targets[0]]
        obstacle = placements[fixed[0]]
        target_position = target.get("position") or target.get("pose") or {}
        obstacle_position = obstacle.get("position") or obstacle.get("pose") or {}
        try:
            relative = float(target_position.get(axis) or 0.0) - float(
                obstacle_position.get(axis) or 0.0
            )
        except (TypeError, ValueError):
            continue
        amount = dx if axis == "x_m" else dz
        delta = amount * (-1.0 if relative < 0.0 else 1.0)
        target_position[axis] = float(target_position.get(axis) or 0.0) + delta
        footprint = target.get("footprint") or {}
        lower, upper = ("min_x", "max_x") if axis == "x_m" else ("min_z", "max_z")
        if lower in footprint and upper in footprint:
            footprint[lower] = float(footprint[lower]) + delta
            footprint[upper] = float(footprint[upper]) + delta
    return result


def _preserve_requested_edit_rotations(layout, flp_ir, canonical_spec, task_contract, movable):
    movable = set(movable or [])
    requested_by_id = {
        str(item.get("instance_id") or ""): item
        for item in (layout or {}).get("placements") or []
        if item.get("instance_id")
    }
    constraints = (canonical_spec or {}).get("instance_constraints") or {}
    targets = []
    for operation in task_contract.get("edit_operations") or []:
        if str(operation.get("op_type") or "").casefold() != "rotate":
            continue
        instance_id = str(operation.get("instance_id") or "")
        placement = requested_by_id.get(instance_id)
        if not instance_id or instance_id not in movable or placement is None:
            continue
        yaw = float(
            placement.get("yaw_deg")
            if placement.get("yaw_deg") is not None
            else (placement.get("rotation") or {}).get("yaw_deg") or 0.0
        ) % 360.0
        allowed = (constraints.get(instance_id) or {}).get("allowed_orientations_deg") or []
        if allowed and not any(
            abs((yaw - float(value) + 180.0) % 360.0 - 180.0) <= GEOMETRY_EPSILON_M
            for value in allowed
        ):
            continue
        targets.append(instance_id)
    if targets:
        from tool.execution.violation_resolution_executor import _preserve_parent_orientations

        _preserve_parent_orientations(flp_ir, canonical_spec, layout, targets)
    return targets


def _evaluator_blocked_candidate_feedback(
    validation_report,
    validator_feedback,
    task_contract,
    revision,
    *,
    has_layout,
):
    disposition = validator_feedback.get("disposition")
    task_type = task_contract.get("task_type")
    violations = copy.deepcopy(
        validator_feedback.get("typed_violations")
        or validator_feedback.get("issues")
        or validation_report.get("errors")
        or []
    )
    recovery_movable, recovery_preserve = _evaluator_assisted_recovery_scope(
        task_contract,
        {"typed_violations": violations},
    )
    can_recover_edit = bool(
        disposition == "edit_conflict"
        and task_type == "DirectiveEdit"
        and has_layout
        and recovery_movable
    )
    actions = []
    if can_recover_edit:
        actions.append(
            {
                "action_id": f"evaluator_recovery_{revision:03d}",
                "action_type": "local_feasibility_search",
                "target_agent": "DesignerAgent",
                "instance_ids": copy.deepcopy(
                    recovery_movable
                ),
                "preserve_instance_ids": copy.deepcopy(
                    recovery_preserve
                ),
                "allowed_repair_operations": [
                    "minimum_translation",
                    "allowed_orientation_change",
                ],
                "objective": "prefer_minimum_movement_from_requested_edit",
            }
        )
    return {
        "schema_version": "evaluator-assisted-recovery-feedback-v1",
        "feedback_type": "evaluator_assisted_recovery",
        "trigger_source": "evaluator_blocked_candidate_review",
        "source": "evaluator",
        "target_agent": "DesignerAgent",
        "source_revision_id": f"evaluation_v{revision}",
        "disposition": (
            "disclosed_local_feasible_recovery"
            if can_recover_edit
            else "not_recoverable"
        ),
        "should_revise": can_recover_edit,
        "requires_user_clarification": False,
        "deviates_from_user_directive": can_recover_edit,
        "exact_directive_satisfied": False if can_recover_edit else None,
        "actions": actions,
        "issues": violations,
        "typed_violations": violations,
        "missing_fields": [],
        "user_notice": {
            "required": True,
            "original_status": validation_report.get("status"),
            "resolution_status": (
                "FEASIBLE_ALTERNATIVE_PROPOSED"
                if can_recover_edit
                else "USER_INPUT_REQUIRED"
            ),
            "exact_directive_satisfied": False if can_recover_edit else None,
            "deviation_will_be_disclosed": can_recover_edit,
            "missing_fields": [],
            "reason_codes": [
                item.get("code") or item.get("error_type")
                for item in violations
                if isinstance(item, dict)
                and (item.get("code") or item.get("error_type"))
            ],
        },
    }


def _evaluator_assisted_recovery_result(
    requested_layout,
    recovered_layout,
    task_contract,
    feedback,
    canonical_spec,
    *,
    recovery_movable_instance_ids=None,
    recovery_preserve_instance_ids=None,
):
    recovered_layout = copy.deepcopy(recovered_layout or requested_layout or {})
    requested_by_id = {
        item.get("instance_id"): item
        for item in (requested_layout or {}).get("placements") or []
        if item.get("instance_id")
    }
    recovered_by_id = {
        item.get("instance_id"): item
        for item in recovered_layout.get("placements") or []
        if item.get("instance_id")
    }
    changed = []
    deviation = []
    recovery_movable = (
        task_contract.get("movable_instance_ids") or []
        if recovery_movable_instance_ids is None
        else recovery_movable_instance_ids
    )
    recovery_preserve = (
        task_contract.get("preserve_instance_ids") or []
        if recovery_preserve_instance_ids is None
        else recovery_preserve_instance_ids
    )
    for instance_id in recovery_movable:
        before = requested_by_id.get(instance_id)
        after = recovered_by_id.get(instance_id)
        if before is None or after is None:
            continue
        before_position = before.get("position") or {}
        after_position = after.get("position") or {}
        dx = float(after_position.get("x_m") or 0.0) - float(
            before_position.get("x_m") or 0.0
        )
        dz = float(after_position.get("z_m") or 0.0) - float(
            before_position.get("z_m") or 0.0
        )
        before_yaw = float((before.get("rotation") or {}).get("yaw_deg") or 0.0)
        after_yaw = float((after.get("rotation") or {}).get("yaw_deg") or 0.0)
        if (
            abs(dx) > GEOMETRY_EPSILON_M
            or abs(dz) > GEOMETRY_EPSILON_M
            or abs(after_yaw - before_yaw) > GEOMETRY_EPSILON_M
        ):
            changed.append(instance_id)
            deviation.append(
                {
                    "instance_id": instance_id,
                    "requested_position": copy.deepcopy(before_position),
                    "applied_position": copy.deepcopy(after_position),
                    "translation_from_requested_endpoint_m": {
                        "dx_m": dx,
                        "dz_m": dz,
                        "distance_m": (dx * dx + dz * dz) ** 0.5,
                    },
                    "requested_yaw_deg": before_yaw,
                    "applied_yaw_deg": after_yaw,
                    "yaw_deviation_deg": after_yaw - before_yaw,
                }
            )
    hard = CanonicalConstraintValidator().validate(
        recovered_layout,
        canonical_spec,
        reference_layout=task_contract.get("base_layout"),
        expected_instance_ids=task_contract.get("expected_instance_ids"),
        preserve_instance_ids=task_contract.get("preserve_instance_ids"),
    )
    status = (
        "EVALUATOR_RECOVERY_SUCCESS"
        if hard.get("status") == "VALID"
        else "EVALUATOR_RECOVERY_UNSAT"
    )
    disclosure = {
        "required": True,
        "original_task_type": "DirectiveEdit",
        "original_directive_satisfied": False,
        "recovery_status": status,
        "policy": "local_feasible_with_disclosure",
        "trigger_source": "evaluator",
        "changed_instance_ids": changed,
        "preserve_instance_ids": copy.deepcopy(
            recovery_preserve
        ),
        "requested_vs_applied": deviation,
        "reason_codes": copy.deepcopy(
            (feedback.get("user_notice") or {}).get("reason_codes") or []
        ),
    }
    return {
        "schema_version": "evaluator-assisted-recovery-v1",
        "status": status,
        "layout": recovered_layout,
        "candidate_changed": bool(changed),
        "changed_instance_ids": changed,
        "actions": copy.deepcopy(feedback.get("actions") or []),
        "deviated_from_user_directive": True,
        "exact_directive_satisfied": False,
        "disclosure": disclosure,
        "validation": {
            "hard_validation": hard,
            "task_contract_validation": validate_task_contract(
                recovered_layout,
                task_contract,
            ),
        },
        "trace": {
            "execution_type": "evaluator_assisted_recovery",
            "trigger_source": "evaluator",
            "task_type": "DirectiveEdit",
            "movable_instance_ids": copy.deepcopy(recovery_movable),
            "preserve_instance_ids": copy.deepcopy(recovery_preserve),
            "changed_instance_ids": changed,
            "deviated_from_user_directive": True,
            "final_status": status,
        },
    }


def _build_optimization_feedback(metrics, comparison, intent, solver_report, revision):
    actions = []
    issue_types = []
    should_revise = False
    stop_reason = "quality_accepted"
    if int(metrics.get("violation_count") or 0) > 0:
        stop_reason = "hard_constraint_feedback_belongs_to_validator"
    elif intent.get("intent_type") in {
        "optimize_layout_from_scratch",
        "optimize_existing_layout",
    } and not comparison["baseline_available"]:
        should_revise = True
        issue_types.append("quality_baseline_requires_comparison_candidate")
        actions.append(
            {
                "action_id": "optimization_action_001",
                "action_type": "global_resolve",
                "reason": "Keep the legal candidate and run one bounded improvement solve for comparison.",
                "target_metrics": ["objective_value", "equipment_grouping_score", "layout_compactness"],
                "preserve_user_constraints": True,
            }
        )
        stop_reason = None
    elif comparison["baseline_available"]:
        relative = comparison.get("relative_objective_improvement")
        if relative is not None and relative > 0.05 and revision < 3:
            should_revise = True
            issue_types.append("objective_still_improving")
            actions.append(
                {
                    "action_id": "optimization_action_001",
                    "action_type": "continue_improvement_search",
                    "reason": "The last revision improved the objective by more than 5 percent.",
                    "preserve_user_constraints": True,
                }
            )
            stop_reason = None
        elif comparison.get("improved") is False:
            stop_reason = "objective_stagnation"
    gap = _optional_float(solver_report.get("gap"))
    if not should_revise and gap is not None and gap > 0.1 and revision < 3:
        should_revise = True
        issue_types.append("large_optimality_gap")
        actions.append(
            {
                "action_id": f"optimization_action_{len(actions) + 1:03d}",
                "action_type": "intensify_search_within_fixed_budget",
                "reason": f"Solver gap {gap:.3f} leaves material improvement headroom.",
                "solver_parameter_changes": {},
                "preserve_user_constraints": True,
            }
        )
        stop_reason = None
    return {
        "source": "evaluator",
        "source_revision_id": f"evaluation_v{revision}",
        "should_revise": should_revise,
        "issue_types": issue_types,
        "actions": actions,
        "stop_reason": stop_reason,
    }


def _evaluation_notes(metrics, comparison, feedback):
    notes = [
        f"Hard-constraint violations: {int(metrics.get('violation_count') or 0)}.",
        f"Objective value: {metrics.get('objective_value')}.",
        f"Layout compactness: {float(metrics.get('layout_compactness') or 0.0):.3f}.",
    ]
    if comparison.get("baseline_available"):
        notes.append(f"Objective delta versus previous revision: {comparison.get('objective_delta')}.")
    notes.append(
        "Designer revision requested." if feedback.get("should_revise") else f"Stop reason: {feedback.get('stop_reason')}."
    )
    return notes


def _site_dimensions(evidence_package):
    facts = evidence_package.get("site_facts_from_user", []) if evidence_package else []
    if not facts:
        return None
    value = facts[0].get("value", {})
    width = value.get("width_m")
    depth = value.get("depth_m")
    if not width or not depth:
        return None
    return {
        "site_id": facts[0].get("site_id") or "SITE_INLINE_001",
        "width_m": float(width),
        "depth_m": float(depth),
    }


def _no_overlap(placements):
    for index, left in enumerate(placements):
        for right in placements[index + 1 :]:
            if _overlaps(left["footprint"], right["footprint"]):
                return False
    return True


def _inside_boundary(placements, site):
    min_x = -site["width_m"] / 2
    max_x = site["width_m"] / 2
    min_z = -site["depth_m"] / 2
    max_z = site["depth_m"] / 2
    for placement in placements:
        footprint = placement["footprint"]
        if (
            footprint["min_x"] < min_x
            or footprint["max_x"] > max_x
            or footprint["min_z"] < min_z
            or footprint["max_z"] > max_z
        ):
            return False
    return True


def _overlaps(left, right):
    return (
        left["min_x"] < right["max_x"] - GEOMETRY_EPSILON_M
        and left["max_x"] > right["min_x"] + GEOMETRY_EPSILON_M
        and left["min_z"] < right["max_z"] - GEOMETRY_EPSILON_M
        and left["max_z"] > right["min_z"] + GEOMETRY_EPSILON_M
    )


def _trace(
    agent_name,
    responsibility,
    input_ref,
    output_ref,
    *,
    iteration=0,
    input_artifacts=None,
    output_artifacts=None,
    decision=None,
    evidence=None,
    trigger_source=None,
    task_type=None,
    movable_instance_ids=None,
    preserve_instance_ids=None,
    changed_instance_ids=None,
    deviated_from_user_directive=None,
    final_status=None,
):
    trace = {
        "agent_name": agent_name,
        "contract_version": CONTRACT_VERSION,
        "responsibility": responsibility,
        "input_ref": input_ref,
        "output_ref": output_ref,
        "iteration": int(iteration),
        "input_artifacts": list(input_artifacts or []),
        "output_artifacts": list(output_artifacts or []),
        "decision": decision,
        "evidence": list(evidence or []),
    }
    execution_fields = {
        "trigger_source": trigger_source,
        "task_type": task_type,
        "movable_instance_ids": movable_instance_ids,
        "preserve_instance_ids": preserve_instance_ids,
        "changed_instance_ids": changed_instance_ids,
        "deviated_from_user_directive": deviated_from_user_directive,
        "final_status": final_status,
    }
    trace.update(
        {
            key: copy.deepcopy(value)
            for key, value in execution_fields.items()
            if value is not None
        }
    )
    return trace


def _event(
    event,
    session_id,
    turn_id,
    agent_name,
    status,
    summary,
    *,
    payload=None,
    viewport_command=None,
    iteration=None,
):
    item = {
        "event": event,
        "session_id": session_id,
        "turn_id": turn_id,
        "agent_name": agent_name,
        "status": status,
        "summary": summary,
        "payload": payload or {},
    }
    if iteration is not None:
        item["iteration"] = int(iteration)
    if viewport_command is not None:
        item["viewport_command"] = viewport_command
    return item


def _attach_assistant_message(result, llm_client, user_message):
    result["assistant_message"] = None
    result["assistant_message_source"] = None
    result["assistant_message_error"] = None
    if _is_clean_validated_layout_result(result):
        result["assistant_message"] = _validated_layout_receipt(
            result,
            user_message,
        )
        result["assistant_message_source"] = "validated_layout_receipt"
        if "layout_response" in result:
            result["layout_response"]["assistant_message"] = result[
                "assistant_message"
            ]
            result["layout_response"]["assistant_message_source"] = result[
                "assistant_message_source"
            ]
            result["layout_response"]["assistant_message_error"] = None
        return
    if result.get("status") == "NEED_CLARIFICATION":
        result["assistant_message"] = _clarification_assistant_reply(
            result,
            user_message,
        )
        result["assistant_message_source"] = "clarification_projection"
        if "layout_response" in result:
            result["layout_response"]["assistant_message"] = result[
                "assistant_message"
            ]
            result["layout_response"]["assistant_message_source"] = result[
                "assistant_message_source"
            ]
            result["layout_response"]["assistant_message_error"] = None
        return
    if llm_client is None or not hasattr(llm_client, "compose_assistant_reply"):
        result["assistant_message_error"] = "assistant_reply_llm_unavailable"
        if "layout_response" in result:
            result["layout_response"]["assistant_message"] = None
            result["layout_response"]["assistant_message_source"] = None
            result["layout_response"]["assistant_message_error"] = result["assistant_message_error"]
        return
    try:
        result["assistant_message"] = llm_client.compose_assistant_reply(
            user_message,
            _assistant_result_summary(result),
        )
        result["assistant_message_source"] = "llm"
    except Exception as exc:  # pragma: no cover - integration error reporting path.
        result["assistant_message_error"] = str(exc)
    if "layout_response" in result:
        result["layout_response"]["assistant_message"] = result["assistant_message"]
        result["layout_response"]["assistant_message_source"] = result["assistant_message_source"]
        result["layout_response"]["assistant_message_error"] = result["assistant_message_error"]


def _is_clean_validated_layout_result(result):
    if result.get("status") != "VALID_SOLVER_READY":
        return False
    validation = result.get("validation") or {}
    report = validation.get("validation_report") or {}
    layout_validation = validation.get("layout_validation") or {}
    canonical = layout_validation.get("canonical_hard_validation") or {}
    viewport = layout_validation.get("viewport_precheck") or {}
    return bool(
        report.get("status") == "VALID_SOLVER_READY"
        and not report.get("errors")
        and not report.get("warnings")
        and layout_validation.get("status") == "VALID"
        and canonical.get("status") == "VALID"
        and int(canonical.get("violation_count") or 0) == 0
        and viewport.get("status") == "VALID"
        and viewport.get("valid") is True
    )


def _validated_layout_receipt(result, user_message):
    """Describe an accepted layout from validated records, not model speculation."""

    design = result.get("design") or {}
    validation = result.get("validation") or {}
    evaluation = result.get("evaluation") or {}
    semantic_intent = result.get("semantic_intent") or {}
    rough_layout = design.get("rough_layout") or {}
    flp_ir = design.get("flp_ir") or {}
    task_contract = design.get("task_contract") or {}
    solver_report = design.get("solver_report") or {}
    layout_validation = validation.get("layout_validation") or {}
    canonical = layout_validation.get("canonical_hard_validation") or {}
    viewport = layout_validation.get("viewport_precheck") or {}

    placements = list(rough_layout.get("placements") or [])
    declared_requested = sum(
        max(1, int(item.get("quantity") or 1))
        for item in semantic_intent.get("asset_mentions") or []
        if isinstance(item, dict)
    )
    task_type = str(
        task_contract.get("task_type")
        or (result.get("intent") or {}).get("task_type")
        or "Generate"
    )
    required_instance_ids = list(task_contract.get("expected_instance_ids") or [])
    requested = (
        len(required_instance_ids)
        if task_type in {"Optimize", "DirectiveEdit", "ConstraintRepair"}
        and required_instance_ids
        else declared_requested or len(placements)
    )
    changes = _layout_change_details(
        task_contract.get("base_layout") or {},
        rough_layout,
    )
    preserved_count = len(
        set(task_contract.get("preserve_instance_ids") or [])
        & {
            str(item.get("instance_id"))
            for item in placements
            if isinstance(item, dict) and item.get("instance_id")
        }
    )
    geometry_records = list(flp_ir.get("equipment_instances") or [])
    catalogue_geometry_used = bool(geometry_records) and all(
        isinstance(item, dict)
        and item.get("model_asset_id")
        and isinstance(item.get("geometry"), dict)
        and (item.get("geometry") or {}).get("dimensions_m")
        for item in geometry_records
    )
    viewport_count = len(viewport.get("observed_instance_ids") or [])
    hard_violations = int(canonical.get("violation_count") or 0)
    solver_used = solver_report.get("solver_used") or result.get("solver_used")
    metrics = evaluation.get("metrics") or {}
    objective_value = metrics.get("objective_value")
    try:
        objective_text = f"{float(objective_value):.3f}"
    except (TypeError, ValueError):
        objective_text = None
    pareto_front = list(solver_report.get("pareto_front") or [])
    selected_candidate_id = solver_report.get("selected_candidate_id")
    selection_policy = _selection_policy_text(
        solver_report.get("selection_policy")
    )
    selected_candidate = next(
        (
            item
            for item in pareto_front
            if isinstance(item, dict)
            and item.get("candidate_id") == selected_candidate_id
        ),
        None,
    )
    objective_values = copy.deepcopy(
        (selected_candidate or {}).get("objective_values")
        or solver_report.get("objective_values")
        or {}
    )
    pareto_history = list(solver_report.get("pareto_history") or [])

    first_key = "minimize_agv_transport_cost" if "minimize_agv_transport_cost" in objective_values else "minimize_weighted_flow_distance"
    first_label = "AGV transport cost index/day" if first_key == "minimize_agv_transport_cost" else "weighted flow distance"
    second_key = RELOCATION_EFFORT_OBJECTIVE if RELOCATION_EFFORT_OBJECTIVE in objective_values else "minimize_high_flow_distance_excess"
    second_label = "relocation effort" if second_key == RELOCATION_EFFORT_OBJECTIVE else "high-flow distance excess"
    def vector_text(objective_type):
        try:
            return f"{float(objective_values[objective_type]):.3f}"
        except (KeyError, TypeError, ValueError):
            return None

    english_request = not re.search(r"[\u3400-\u9fff]", str(user_message or ""))
    if english_request:
        heading = {
            "Generate": "The initial layout is ready.",
            "Optimize": "The optimized layout is ready.",
            "DirectiveEdit": "The requested layout update is ready.",
            "ConstraintRepair": "The repaired layout is ready.",
        }.get(task_type, "The layout is ready.")
        lines = [heading, ""]
        if task_type == "Generate":
            lines.append(
                f"- Placed: {len(placements)} of {requested} requested units."
            )
        else:
            lines.append(f"- Layout units: {len(placements)} of {requested} retained.")
            lines.append(
                f"- Change from the parent layout: {len(changes)} units changed; "
                f"{preserved_count} protected units retained."
            )
        if task_type == "Generate" and catalogue_geometry_used:
            lines.append(
                "- Geometry: the resolved catalog records supplied the equipment footprints."
            )
        lines.append(
            f"- Validation: {hard_violations} hard-constraint violations; "
            f"viewport precheck passed for {viewport_count} units."
        )
        if solver_used:
            lines.append(f"- Solver: {solver_used}.")
        if objective_text is not None:
            lines.append(f"- Evaluation: objective value {objective_text}.")
        weighted_flow = vector_text(first_key)
        high_flow_excess = vector_text(second_key)
        if selected_candidate_id and pareto_front:
            policy = f" using {selection_policy}" if selection_policy else ""
            lines.append(
                f"- Trade-off: selected Pareto candidate {selected_candidate_id} from "
                f"{len(pareto_front)} nondominated candidates{policy}."
            )
        if weighted_flow is not None or high_flow_excess is not None:
            vector_parts = []
            if weighted_flow is not None:
                vector_parts.append(f"{first_label} {weighted_flow}")
            if high_flow_excess is not None:
                vector_parts.append(f"{second_label} {high_flow_excess}")
            lines.append("- Objective vector: " + "; ".join(vector_parts) + ".")
        if pareto_history:
            lines.append(
                f"- Pareto history: {len(pareto_history)} recorded optimization steps."
            )
        if pareto_front:
            lines.extend(["", "Pareto candidates:"])
            for candidate in pareto_front:
                if not isinstance(candidate, dict):
                    continue
                candidate_values = candidate.get("objective_values") or {}
                try:
                    candidate_j1 = (
                        f"{float(candidate_values[first_key]):.3f}"
                    )
                except (KeyError, TypeError, ValueError):
                    candidate_j1 = "n/a"
                try:
                    candidate_j2 = (
                        f"{float(candidate_values[second_key]):.3f}"
                    )
                except (KeyError, TypeError, ValueError):
                    candidate_j2 = "n/a"
                selected = " (selected)" if candidate.get("candidate_id") == selected_candidate_id else ""
                lines.append(
                    f"- {candidate.get('candidate_id') or 'candidate'}{selected}: "
                    f"{first_label} {candidate_j1}; {second_label} {candidate_j2}."
                )
        if task_type in {"DirectiveEdit", "ConstraintRepair"} and changes:
            lines.extend(["", "Applied changes:"])
            lines.extend(
                f"- {_format_layout_change(change)}" for change in changes
            )
        return "\n".join(lines)

    heading = {
        "Generate": "初始布局已生成并通过验证。",
        "Optimize": "优化后的布局已通过验证。",
        "DirectiveEdit": "请求的布局更新已通过验证。",
        "ConstraintRepair": "修复后的布局已通过验证。",
    }.get(task_type, "布局已生成并通过验证。")
    lines = [heading, ""]
    if task_type == "Generate":
        lines.append(f"- 已放置：{len(placements)}/{requested} 台设备。")
    else:
        lines.append(f"- 布局设备：保留 {len(placements)}/{requested} 台。")
        lines.append(
            f"- 相对父布局：变更 {len(changes)} 台；保留受保护设备 {preserved_count} 台。"
        )
    if task_type == "Generate" and catalogue_geometry_used:
        lines.append("- 几何：设备占地尺寸来自已解析的目录记录。")
    lines.append(
        f"- 验证：硬约束违规 {hard_violations} 项；视口预检通过 {viewport_count} 台设备。"
    )
    if solver_used:
        lines.append(f"- 求解器：{solver_used}。")
    if objective_text is not None:
        lines.append(f"- 评估：目标值 {objective_text}。")
    weighted_flow = vector_text(first_key)
    high_flow_excess = vector_text(second_key)
    if selected_candidate_id and pareto_front:
        policy = f"；策略：{selection_policy}" if selection_policy else ""
        lines.append(
            f"- 权衡：从 {len(pareto_front)} 个非支配候选中选择了 {selected_candidate_id}{policy}。"
        )
    if weighted_flow is not None or high_flow_excess is not None:
        vector_parts = []
        if weighted_flow is not None:
            vector_parts.append(f"加权流距离 {weighted_flow}")
        if high_flow_excess is not None:
            vector_parts.append(f"高流量距离超额 {high_flow_excess}")
        lines.append("- 目标向量：" + "；".join(vector_parts) + "。")
    if pareto_history:
        lines.append(f"- Pareto 历史：记录了 {len(pareto_history)} 个优化步骤。")
    if pareto_front:
        lines.extend(["", "Pareto 候选："])
        for candidate in pareto_front:
            if not isinstance(candidate, dict):
                continue
            candidate_values = candidate.get("objective_values") or {}
            try:
                candidate_j1 = (
                    f"{float(candidate_values[first_key]):.3f}"
                )
            except (KeyError, TypeError, ValueError):
                candidate_j1 = "n/a"
            try:
                candidate_j2 = (
                    f"{float(candidate_values[second_key]):.3f}"
                )
            except (KeyError, TypeError, ValueError):
                candidate_j2 = "n/a"
            selected = "（已选择）" if candidate.get("candidate_id") == selected_candidate_id else ""
            lines.append(
                f"- {candidate.get('candidate_id') or '候选'}{selected}："
                f"加权流距离 {candidate_j1}；高流量超额 {candidate_j2}。"
            )
    if task_type in {"DirectiveEdit", "ConstraintRepair"} and changes:
        lines.extend(["", "已执行变更："])
        lines.extend(f"- {_format_layout_change(change)}" for change in changes)
    return "\n".join(lines)


def _layout_change_details(parent_layout, current_layout):
    parent = {
        str(item.get("instance_id")): item
        for item in (parent_layout or {}).get("placements") or []
        if isinstance(item, dict) and item.get("instance_id")
    }
    current = {
        str(item.get("instance_id")): item
        for item in (current_layout or {}).get("placements") or []
        if isinstance(item, dict) and item.get("instance_id")
    }
    changes = []
    for instance_id in sorted(set(parent) | set(current)):
        before = parent.get(instance_id)
        after = current.get(instance_id)
        if before is None:
            changes.append({"instance_id": instance_id, "change_type": "added"})
            continue
        if after is None:
            changes.append({"instance_id": instance_id, "change_type": "removed"})
            continue
        deltas = {}
        for axis in ("x_m", "y_m", "z_m", "yaw_deg"):
            before_value = _layout_pose_value(before, axis)
            after_value = _layout_pose_value(after, axis)
            if before_value is None or after_value is None:
                continue
            delta = after_value - before_value
            if abs(delta) > GEOMETRY_EPSILON_M:
                deltas[axis] = delta
        if deltas:
            changes.append(
                {
                    "instance_id": instance_id,
                    "change_type": "moved",
                    "deltas": deltas,
                }
            )
    return changes


def _selection_policy_text(value):
    if isinstance(value, dict):
        description = str(value.get("description") or "").strip()
        return description or "closest to the normalized ideal point"
    text = str(value or "").strip()
    if not text:
        return None
    return text.replace("_", " ")


def _layout_pose_value(placement, field):
    position = placement.get("position") or placement.get("pose") or {}
    value = placement.get(field, position.get(field))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_layout_change(change):
    instance_id = change.get("instance_id") or "unit"
    change_type = change.get("change_type")
    if change_type in {"added", "removed"}:
        return f"{instance_id} was {change_type}."
    deltas = change.get("deltas") or {}
    parts = []
    for axis, value in deltas.items():
        unit = "°" if axis == "yaw_deg" else "m"
        parts.append(f"Δ{axis.replace('_m', '').replace('_deg', '')} {value:+.3f} {unit}")
    return f"{instance_id}: " + "; ".join(parts) + "."


def _clarification_assistant_reply(result, user_message):
    repair = ((result.get("validation") or {}).get("repair") or {})
    questions = [
        str(question).strip()
        for question in repair.get("clarification_questions") or []
        if str(question).strip()
    ]
    english_request = not re.search(r"[\u3400-\u9fff]", str(user_message or ""))
    if english_request:
        if not questions:
            return "I need a little more information before I can continue."
        items = "\n".join(f"- {question}" for question in questions)
        return (
            "I need one or more choices before I can continue.\n\n"
            f"{items}\n\n"
            "Reply with the option description you want. I will keep the rest of your request unchanged."
        )
    if not questions:
        return "继续之前，我还需要一些信息。"
    items = "\n".join(f"- {question}" for question in questions)
    return f"继续之前，请确认以下信息：\n\n{items}"


def _assistant_result_summary(result):
    design = result.get("design") or {}
    validation = result.get("validation") or {}
    evaluation = result.get("evaluation") or {}
    rough_layout = design.get("rough_layout") or {}
    solver_report = design.get("solver_report") or {}
    validation_report = validation.get("validation_report") or {}
    repair = validation.get("repair") or {}
    metrics = evaluation.get("metrics") or {}
    semantic_intent = result.get("semantic_intent") or {}
    loop_report = result.get("loop_report") or {}
    evaluator_recovery = design.get("evaluator_assisted_recovery") or {}
    evaluator_clarification = (
        design.get("evaluator_clarification_review") or {}
    )
    requested_instances = sum(
        max(1, int(item.get("quantity") or 1))
        for item in semantic_intent.get("asset_mentions") or []
    )
    return {
        "status": result.get("status"),
        "intent_type": (result.get("intent") or {}).get("intent_type"),
        "semantic_intent_source": semantic_intent.get("source"),
        "semantic_reasoning_summary": semantic_intent.get("reasoning_summary"),
        "layout_id": rough_layout.get("layout_id"),
        "placed_instances": len(rough_layout.get("placements") or []),
        "requested_instances": requested_instances,
        "solver": solver_report.get("solver"),
        "solver_requested": solver_report.get("solver_requested"),
        "solver_used": solver_report.get("solver_used"),
        "solver_selection_source": solver_report.get("selection_source"),
        "default_fallback_used": solver_report.get("default_fallback_used"),
        "solver_status": solver_report.get("status"),
        "objective_value": solver_report.get("objective_value"),
        "constraint_penalty": solver_report.get("constraint_penalty"),
        "violation_count": solver_report.get("violation_count"),
        "gap": solver_report.get("gap"),
        "evaluation_status": evaluation.get("evaluation_status"),
        "metrics": {
            "site_utilization": metrics.get("site_utilization"),
            "flow_distance": metrics.get("flow_distance"),
            "overlap_violation_count": metrics.get("overlap_violation_count"),
            "boundary_violation_count": metrics.get("boundary_violation_count"),
            "non_placeable_violation_count": metrics.get("non_placeable_violation_count"),
            "clearance_violation_count": metrics.get("clearance_violation_count"),
        },
        "optimization_selection": {
            "objective_values": copy.deepcopy(
                solver_report.get("objective_values") or {}
            ),
            "pareto_front": copy.deepcopy(solver_report.get("pareto_front") or []),
            "pareto_history": copy.deepcopy(
                solver_report.get("pareto_history") or []
            ),
            "selected_candidate_id": solver_report.get("selected_candidate_id"),
            "selection_policy": solver_report.get("selection_policy"),
        },
        "validation_errors": validation_report.get("errors") or [],
        "validation_warnings": validation_report.get("warnings") or [],
        "clarification_questions": repair.get("clarification_questions") or [],
        "evaluator_assisted_recovery": {
            "status": evaluator_recovery.get("status"),
            "deviated_from_user_directive": evaluator_recovery.get(
                "deviated_from_user_directive"
            ),
            "exact_directive_satisfied": evaluator_recovery.get(
                "exact_directive_satisfied"
            ),
            "changed_instance_ids": evaluator_recovery.get(
                "changed_instance_ids"
            )
            or [],
            "disclosure": copy.deepcopy(
                evaluator_recovery.get("disclosure") or {}
            ),
        },
        "evaluator_clarification_review": {
            "status": evaluator_clarification.get("evaluation_status"),
            "notice": copy.deepcopy(
                evaluator_clarification.get("recovery_notice") or {}
            ),
        },
        "evaluator_recovery_notice": copy.deepcopy(
            evaluation.get("recovery_notice") or {}
        ),
        "agentic_loop": {
            "iterations": loop_report.get("iterations"),
            "converged": loop_report.get("converged"),
            "quality_converged": loop_report.get("quality_converged"),
            "termination_reason": loop_report.get("termination_reason"),
            "repair_rounds": loop_report.get("repair_rounds"),
            "optimization_rounds": loop_report.get("optimization_rounds"),
            "selected_revision": loop_report.get("selected_revision"),
            "artifact_chain": loop_report.get("artifact_chain") or [],
        },
    }
