"""Gray Swan Cygnal guardrail integration for LiteLLM."""

from typing import TYPE_CHECKING, Final

from litellm.types.guardrails import SupportedGuardrailIntegrations

from .grayswan import (
    GraySwanGuardrail,
    GraySwanGuardrailAPIError,
    GraySwanGuardrailBlocked,
    GraySwanGuardrailConfigError,
    GraySwanGuardrailMissingSecrets,
)

if TYPE_CHECKING:
    from litellm.types.guardrails import Guardrail, LitellmParams


def initialize_guardrail(litellm_params: "LitellmParams", guardrail: "Guardrail") -> GraySwanGuardrail:
    import litellm

    guardrail_name: Final = guardrail.get("guardrail_name")
    if not guardrail_name:
        raise ValueError("Gray Swan guardrail requires a guardrail_name")

    optional_params: Final = getattr(litellm_params, "optional_params", None)

    grayswan_guardrail: Final = GraySwanGuardrail(
        guardrail_name=guardrail_name,
        api_key=litellm_params.api_key,
        api_base=litellm_params.api_base,
        on_flagged_action=_get_config_value(litellm_params, optional_params, "on_flagged_action"),
        violation_threshold=_get_config_value(litellm_params, optional_params, "violation_threshold"),
        reasoning_mode=_get_config_value(litellm_params, optional_params, "reasoning_mode"),
        categories=_get_config_value(litellm_params, optional_params, "categories"),
        policy_id=_get_config_value(litellm_params, optional_params, "policy_id"),
        streaming_end_of_stream_only=_get_config_value(litellm_params, optional_params, "streaming_end_of_stream_only")
        or False,
        streaming_sampling_rate=_get_config_value(litellm_params, optional_params, "streaming_sampling_rate") or 5,
        streaming_buffer_until_moderated=_get_config_value(
            litellm_params, optional_params, "streaming_buffer_until_moderated"
        ),
        fail_open=_get_config_value(litellm_params, optional_params, "fail_open"),
        fail_open_input=_get_config_value(litellm_params, optional_params, "fail_open_input"),
        fail_open_output=_get_config_value(litellm_params, optional_params, "fail_open_output"),
        guardrail_timeout=_get_config_value(litellm_params, optional_params, "guardrail_timeout"),
        flag_on_ipi=_get_config_value(litellm_params, optional_params, "flag_on_ipi"),
        flag_on_mutation=_get_config_value(litellm_params, optional_params, "flag_on_mutation"),
        block_status_code=_get_config_value(litellm_params, optional_params, "block_status_code"),
        block_message=_get_config_value(litellm_params, optional_params, "block_message"),
        extra_headers=_get_config_value(litellm_params, optional_params, "extra_headers"),
        monitor_headers=_get_config_value(litellm_params, optional_params, "monitor_headers"),
        event_hook=litellm_params.mode,
        default_on=litellm_params.default_on,
    )

    litellm.logging_callback_manager.add_litellm_callback(grayswan_guardrail)
    return grayswan_guardrail


def _get_config_value(litellm_params, optional_params, attribute_name):
    if optional_params is not None:
        value: Final = getattr(optional_params, attribute_name, None)
        if value is not None:
            return value
    return getattr(litellm_params, attribute_name, None)


guardrail_initializer_registry: Final = {
    SupportedGuardrailIntegrations.GRAYSWAN.value: initialize_guardrail,
}


guardrail_class_registry: Final = {
    SupportedGuardrailIntegrations.GRAYSWAN.value: GraySwanGuardrail,
}


__all__ = [
    "GraySwanGuardrail",
    "GraySwanGuardrailAPIError",
    "GraySwanGuardrailBlocked",
    "GraySwanGuardrailConfigError",
    "GraySwanGuardrailMissingSecrets",
    "initialize_guardrail",
]
