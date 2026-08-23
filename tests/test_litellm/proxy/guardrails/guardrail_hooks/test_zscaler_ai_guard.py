import pytest

from litellm.proxy.guardrails.guardrail_hooks.zscaler_ai_guard import initialize_guardrail
from litellm.types.guardrails import LitellmParams


def test_initializer_coerces_string_policy_id_to_int() -> None:
    """LitellmParams.policy_id is str | int; zscaler must still receive an int."""
    params = LitellmParams(guardrail="zscaler_ai_guard", mode="pre_call", api_key="key", policy_id="12345")
    callback = initialize_guardrail(params, {"guardrail_name": "zscaler-guard"})
    assert callback.policy_id == 12345


def test_initializer_keeps_int_policy_id() -> None:
    params = LitellmParams(guardrail="zscaler_ai_guard", mode="pre_call", api_key="key", policy_id=7)
    callback = initialize_guardrail(params, {"guardrail_name": "zscaler-guard-int"})
    assert callback.policy_id == 7


def test_initializer_rejects_non_numeric_policy_id_with_clear_error() -> None:
    params = LitellmParams(guardrail="zscaler_ai_guard", mode="pre_call", api_key="key", policy_id="not-a-number")
    with pytest.raises(ValueError, match="zscaler_ai_guard policy_id must be an integer"):
        initialize_guardrail(params, {"guardrail_name": "zscaler-guard-bad"})
