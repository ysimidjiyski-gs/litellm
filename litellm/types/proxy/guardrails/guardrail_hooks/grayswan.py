"""Gray Swan guardrail configuration models."""

from typing import Literal

from pydantic import Field

from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel


class GraySwanGuardrailConfigModelOptionalParams(LiteLLMBaseModel):
    """Optional parameters for the Gray Swan guardrail."""

    on_flagged_action: str | None = Field(
        default="passthrough",
        description="Action when a violation is detected: 'block' rejects the call (400 error), 'monitor' logs only, 'passthrough' replaces response content with violation message (200 status).",
    )
    violation_threshold: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Threshold between 0 and 1 at which Gray Swan violations trigger the configured action.",
    )
    reasoning_mode: str | None = Field(
        default=None,
        description="Gray Swan reasoning mode override. Accepted values: 'off', 'hybrid', 'thinking'.",
    )
    policy_id: str | None = Field(
        default=None,
        description="Gray Swan policy identifier to apply during monitoring.",
    )
    categories: dict[str, str] | None = Field(
        default=None,
        description="Default Gray Swan category definitions to send with each request.",
    )
    fail_open: bool | None = Field(
        default=True,
        description="If true (default), errors contacting Gray Swan are logged and the request proceeds. If false, errors propagate and block the request.",
    )
    fail_open_input: bool | None = Field(
        default=None,
        description="Overrides fail_open for request scans (pre_call and during_call). Defaults to fail_open.",
    )
    fail_open_output: bool | None = Field(
        default=None,
        description="Overrides fail_open for response scans (post_call). Defaults to fail_open.",
    )
    guardrail_timeout: float | None = Field(
        default=30.0,
        description="Timeout in seconds for calling the Gray Swan guardrail service.",
    )
    flag_on_ipi: bool | None = Field(
        default=True,
        description="Treat a detected indirect prompt injection as a violation even when the violation score is below violation_threshold. Gray Swan reports tool-result injections only through this flag.",
    )
    flag_on_mutation: bool | None = Field(
        default=True,
        description="Treat a detected mutation (obfuscated harmful intent) as a violation even when the violation score is below violation_threshold.",
    )
    block_status_code: Literal[400, 403, 422] | None = Field(
        default=400,
        description="HTTP status returned when on_flagged_action is 'block'. Limited to the statuses router retries and fallbacks treat as a guardrail intervention, so a block is never retried onto another deployment.",
    )
    block_message: str | None = Field(
        default=None,
        description="Message in the block response's detail.error. Defaults to 'Blocked by Gray Swan Guardrail'. detail.type is always 'grayswan_guardrail_block', and Gray Swan's own message is returned in detail.block_message.",
    )
    extra_headers: list[str] | None = Field(
        default=None,
        description="Inbound request header names to forward to Gray Swan in metadata.headers, in addition to the built-in Claude Code and Codex session headers. Credential headers (authorization, cookies, API keys, tokens, secrets) are never forwarded.",
    )
    monitor_headers: dict[str, str] | None = Field(
        default=None,
        description="Static headers added to every call to Gray Swan's monitor endpoint, e.g. {'x-cygnal-model': '...'} to pick the judge.",
    )
    streaming_buffer_until_moderated: bool | None = Field(
        default=True,
        description="If true (default), streamed responses are withheld until the end-of-stream post_call scan passes, so flagged output never reaches the client. Set false to stream live and scan afterwards.",
    )


class GraySwanGuardrailConfigModel(GuardrailConfigModel[GraySwanGuardrailConfigModelOptionalParams]):
    """Configuration parameters for the Gray Swan guardrail."""

    api_key: str | None = Field(
        default=None,
        description="API key for Gray Swan. Reads from the `GRAYSWAN_API_KEY` environment variable when omitted.",
    )
    api_base: str | None = Field(
        default=None,
        description="Override for the Gray Swan API base URL. Defaults to https://api.grayswan.ai and can be set via `GRAYSWAN_API_BASE`.",
    )
    policy_id: int | str | None = Field(
        default=None,
        union_mode="left_to_right",
        description="Policy ID. Gray Swan takes its string policy id. Zscaler AI Guard takes an integer and can also read ZSCALER_AI_GUARD_POLICY_ID.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Gray Swan Guardrail"
