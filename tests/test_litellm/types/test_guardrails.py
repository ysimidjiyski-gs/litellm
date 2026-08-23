"""Regression tests for field collisions across the vendor config models merged into LitellmParams."""

from litellm.types.guardrails import GuardrailOptionalParams, LitellmParams


def test_policy_id_accepts_string_ids() -> None:
    """policy_id must accept string ids (grayswan) despite zscaler declaring it as int."""
    params = LitellmParams(guardrail="grayswan", mode="pre_call", policy_id="6a875d7aad35908d75bdf715")
    assert params.policy_id == "6a875d7aad35908d75bdf715"
    assert LitellmParams(**params.model_dump()).policy_id == "6a875d7aad35908d75bdf715"


def test_policy_id_still_accepts_int_ids() -> None:
    params = LitellmParams(guardrail="zscaler_ai_guard", mode="pre_call", policy_id=12345)
    assert params.policy_id == 12345


def test_optional_params_keys_survive_and_support_getattr() -> None:
    """Initializers read optional_params via getattr and check model_fields_set."""
    params = LitellmParams(
        guardrail="grayswan",
        mode="pre_call",
        optional_params={
            "policy_id": "6a875d7aad35908d75bdf715",
            "on_flagged_action": "block",
            "reasoning_mode": "hybrid",
        },
    )
    optional_params = params.optional_params
    assert type(optional_params) is GuardrailOptionalParams
    assert getattr(optional_params, "policy_id", None) == "6a875d7aad35908d75bdf715"
    assert getattr(optional_params, "on_flagged_action", None) == "block"
    assert getattr(optional_params, "reasoning_mode", None) == "hybrid"
    assert {"policy_id", "on_flagged_action", "reasoning_mode"} <= optional_params.model_fields_set


def test_optional_params_not_validated_against_first_base_vendor_model() -> None:
    """These key names collide with constrained fields of the first base in the MRO
    (cisco: timeout capped at 60, inspection_type a Literal) and must still pass."""
    params = LitellmParams(
        guardrail="grayswan",
        mode="pre_call",
        optional_params={"timeout": 100.0, "inspection_type": "not-a-cisco-surface"},
    )
    assert getattr(params.optional_params, "timeout", None) == 100.0
    assert getattr(params.optional_params, "inspection_type", None) == "not-a-cisco-surface"


def test_optional_params_keys_survive_for_second_vendor() -> None:
    params = LitellmParams(
        guardrail="ibm_guardrails",
        mode="pre_call",
        optional_params={"detector_params": {"threshold": 0.8}, "block_on_detection": False},
    )
    assert getattr(params.optional_params, "detector_params", None) == {"threshold": 0.8}
    assert getattr(params.optional_params, "block_on_detection", None) is False
