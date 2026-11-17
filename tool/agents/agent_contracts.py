CONTRACT_VERSION = "1.1"


AGENT_CONTRACTS = {
    "IntentUnderstandingAgent": {
        "input_fields": ["user_message", "active_layout_id", "scene_context"],
        "output_fields": [
            "intent",
            "semantic_intent",
            "linguistic_frame",
            "intent_contract",
            "compiler_message",
            "completeness",
            "unresolved_requirements",
            "can_enter_designer",
            "semantic_evidence",
            "trace",
        ],
    },
    "DesignerAgent": {
        "input_fields": ["intent", "semantic_intent", "scene_context", "feedback"],
        "output_fields": [
            "design_status",
            "flp_ir",
            "rough_layout",
            "solver_report",
            "design_basis",
            "solver_evidence",
            "applied_feedback",
            "revision",
            "trace",
        ],
    },
    "MonolithicPlanningAgent": {
        "input_fields": ["user_message", "rag_evidence", "scene_context"],
        "output_fields": [
            "design_status",
            "flp_ir",
            "rough_layout",
            "solver_report",
            "design_basis",
            "solver_evidence",
            "applied_feedback",
            "revision",
            "trace",
        ],
    },
    "ValidatorAgent": {
        "input_fields": [
            "flp_ir",
            "intent",
            "semantic_intent",
            "rough_layout",
        ],
        "output_fields": [
            "validation_report",
            "intent_validation",
            "layout_validation",
            "validation_scope",
            "repair",
            "revision",
            "trace",
        ],
    },
    "EvaluatorAgent": {
        "input_fields": [
            "rough_layout",
            "flp_ir",
            "validation_report",
            "solver_report",
            "previous_evaluation",
        ],
        "output_fields": [
            "evaluation_status",
            "metrics",
            "comparison",
            "optimization_feedback",
            "revision",
            "trace",
        ],
    },
}


class AgentContractError(ValueError):
    pass


def contract_schema(agent_name):
    try:
        spec = AGENT_CONTRACTS[agent_name]
    except KeyError as exc:
        raise AgentContractError(f"Unknown agent contract: {agent_name}") from exc
    return {
        "contract_name": agent_name,
        "contract_version": CONTRACT_VERSION,
        "input_fields": list(spec["input_fields"]),
        "output_fields": list(spec["output_fields"]),
    }


def validate_agent_output(agent_name, payload):
    schema = contract_schema(agent_name)
    if not isinstance(payload, dict):
        return [f"{agent_name} output must be an object"]
    errors = [
        f"missing output field: {field}"
        for field in schema["output_fields"]
        if field not in payload
    ]
    if payload.get("agent_name") != agent_name:
        errors.append(f"agent_name must be {agent_name}")
    trace = payload.get("trace")
    if "trace" in payload and not isinstance(trace, dict):
        errors.append("trace must be an object")
    elif isinstance(trace, dict) and trace.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"trace.contract_version must be {CONTRACT_VERSION}")
    return errors


def enforce_agent_output(agent_name, payload):
    errors = validate_agent_output(agent_name, payload)
    if errors:
        raise AgentContractError(f"{agent_name} contract violation: {'; '.join(errors)}")
    result = dict(payload)
    result["contract"] = {
        **contract_schema(agent_name),
        "valid": True,
        "errors": [],
    }
    return result
