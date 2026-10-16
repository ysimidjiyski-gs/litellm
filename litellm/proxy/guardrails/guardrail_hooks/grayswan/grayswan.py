"""Gray Swan Cygnal guardrail integration."""

import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate, chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Optional, Protocol, TypeAlias, cast

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, TypeAdapter
from typing_extensions import NotRequired, ReadOnly, TypedDict, Unpack

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import Timeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads
from litellm.llms.base_llm.guardrail_translation.utils import (
    effective_scan_only_tool_results_for_guardrail,
    effective_skip_system_message_for_guardrail,
    effective_skip_tool_message_for_guardrail,
    scoped_structured_message_indices,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy.common_utils.callback_utils import add_guardrail_scan_id
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs, GuardrailStatus

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
    from litellm.types.llms.anthropic import AllAnthropicToolsValues

GRAYSWAN_BLOCK_ERROR_MSG: Final = "Blocked by Gray Swan Guardrail"
GRAYSWAN_BLOCK_TYPE: Final = "grayswan_guardrail_block"
GRAYSWAN_INVALID_PARAMS_TYPE: Final = "grayswan_guardrail_invalid_params"
GRAYSWAN_BLOCK_STATUS_CODES: Final = frozenset({400, 403, 422})

DEFAULT_FORWARDED_HEADERS: Final = (
    "shade_scan_id",
    "x-claude-code-session-id",
    "x-claude-code-prompt-id",
    "session-id",
    "thread-id",
    "x-client-request-id",
    "originator",
    "user-agent",
)
SESSION_ID_HEADERS: Final = ("x-claude-code-session-id", "session-id")
_CREDENTIAL_HEADER_MARKERS: Final = (
    "auth",
    "cookie",
    "key",
    "token",
    "secret",
    "password",
    "credential",
    "signature",
    "cert",
    "jwt",
)
_RESERVED_MONITOR_HEADERS: Final = frozenset(
    {"authorization", "content-type", "grayswan-api-key", "x-grayswan-api-key"}
)
_SHAPE_REJECTION_STATUSES: Final = frozenset({400, 413, 422})
_MEDIA_PART_TYPES: Final = frozenset(
    {"image", "image_url", "input_image", "input_audio", "audio", "input_file", "file", "document", "video_url"}
)
_REBUILT_MESSAGE_KEYS: Final = frozenset({"role", "content", "tool_calls", "function_call", "tool_call_id"})
_FORWARDED_LITELLM_METADATA_FIELDS: Final = (
    "user_api_key_hash",
    "user_api_key_spend",
    "user_api_key_alias",
    "user_api_key_user_id",
    "user_api_key_user_email",
    "user_api_key_team_id",
    "user_api_key_team_alias",
    "user_api_key_org_id",
    "user_api_key_end_user_id",
    "user_api_key_request_route",
)
_CHAT_ROLES: Final = frozenset({"system", "developer", "user", "assistant", "tool", "function"})

ScanOutcome: TypeAlias = Literal["allow", "flagged", "block", "passthrough", "skipped", "fail_open", "fail_closed"]
_OUTCOME_STATUS: Final[Mapping[ScanOutcome, GuardrailStatus]] = MappingProxyType(
    {
        "allow": "success",
        "flagged": "guardrail_flagged",
        "block": "guardrail_intervened",
        "passthrough": "guardrail_intervened",
        "skipped": "not_run",
        "fail_open": "guardrail_failed_to_respond",
        "fail_closed": "guardrail_failed_to_respond",
    }
)
_FLAGGED_OUTCOME: Final[Mapping[str, ScanOutcome]] = MappingProxyType(
    {"block": "block", "passthrough": "passthrough", "monitor": "flagged"}
)


class _GraySwanMonitorResponse(TypedDict):
    """Body returned by Gray Swan's `/cygnal/monitor` endpoint."""

    violation: ReadOnly[NotRequired[float | None]]
    violated_rules: ReadOnly[NotRequired[list[object]]]
    violated_rule_descriptions: ReadOnly[NotRequired[list[object]]]
    mutation: ReadOnly[NotRequired[bool | None]]
    ipi: ReadOnly[NotRequired[bool | None]]
    block_message: ReadOnly[NotRequired[str | None]]
    activity_id: ReadOnly[NotRequired[str | None]]


@dataclass(frozen=True, slots=True)
class _MonitorFailure:
    error: Exception
    timed_out: bool

    @property
    def status_code(self) -> int | None:
        status: Final = getattr(self.error, "status_code", None)
        return status if isinstance(status, int) else None

    @property
    def rejected_shape(self) -> bool:
        return not self.timed_out and self.status_code in _SHAPE_REJECTION_STATUSES

    def rejected_payload(self, *, client_steered: bool) -> bool:
        status: Final = self.status_code
        return self.rejected_shape or (client_steered and status is not None and 400 <= status < 500)


@dataclass(frozen=True, slots=True)
class _ScanInput:
    messages: Sequence[Mapping[str, object]]
    tools: Sequence[object] | None
    fallback: Sequence[Mapping[str, object]] | None


class _CustomGuardrailOptions(TypedDict, total=False, extra_items=object):
    pass


class _GraySwanMonitorHTTPResponse(Protocol):
    def raise_for_status(self) -> object: ...

    def json(self) -> _GraySwanMonitorResponse: ...


class _GraySwanMonitorHTTPClient(Protocol):
    async def post(
        self,
        *,
        url: str,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> _GraySwanMonitorHTTPResponse: ...


class _MonitorMessage(TypedDict):
    role: ReadOnly[str]
    content: ReadOnly[NotRequired[str]]
    tool_calls: ReadOnly[NotRequired[tuple[Mapping[str, object], ...]]]


def _as_plain_dict(item: object) -> Mapping[str, object]:
    if isinstance(item, Mapping):
        return item
    if isinstance(item, BaseModel):
        return TypeAdapter(dict[str, object]).validate_python(item.model_dump(mode="json"))
    return cast("Mapping[str, object]", item)  # cast-ok: wire rows are message/tool-call dicts


def _is_chat_message(message: Mapping[str, object]) -> bool:
    return message.get("role") in _CHAT_ROLES and ("content" in message or "tool_calls" in message)


def _is_credential_header(name: str) -> bool:
    return any(marker in name for marker in _CREDENTIAL_HEADER_MARKERS)


def _as_mapping(value: object) -> Mapping[str, object]:
    return cast("Mapping[str, object]", value) if isinstance(value, Mapping) else {}  # cast-ok: metadata dicts


def _configured_litellm_key_header() -> str | None:
    proxy_server: Final = sys.modules.get("litellm.proxy.proxy_server")
    name: Final = _as_mapping(getattr(proxy_server, "general_settings", None)).get("litellm_key_header_name")
    return name.lower() if isinstance(name, str) else None


def _as_items(value: object) -> tuple[object, ...]:
    return tuple(cast("Sequence[object]", value)) if isinstance(value, (list, tuple)) else ()  # cast-ok: JSON arrays


def _is_media_part(part: object) -> bool:
    return _as_mapping(part).get("type") in _MEDIA_PART_TYPES


def _is_wire_tool_call(call: object) -> bool:
    fields: Final = _as_mapping(call)
    function: Final = _as_mapping(fields.get("function"))
    call_id: Final = fields.get("id")
    return (
        fields.get("type") == "function"
        and isinstance(call_id, str)
        and bool(call_id)
        and isinstance(function.get("name"), str)
        and isinstance(function.get("arguments"), str)
    )


def _needs_normalization(message: Mapping[str, object]) -> bool:
    content: Final = message.get("content")
    tool_calls: Final = message.get("tool_calls")
    return (
        message.get("role") == "function"
        or "function_call" in message
        or not all(_is_wire_tool_call(call) for call in _as_items(tool_calls))
        or any(_is_media_part(part) for part in _as_items(content))
    )


def _arguments(value: object) -> str:
    if isinstance(value, str):
        return value
    return "" if value is None else json.dumps(value)


def _wire_tool_call(call: object, fallback_id: str) -> Mapping[str, object]:
    fields: Final = _as_plain_dict(call)
    is_custom: Final = fields.get("type") == "custom"
    custom: Final = _as_mapping(fields.get("custom"))
    function: Final = _as_mapping(fields.get("function"))
    name: Final = custom.get("name") if is_custom else function.get("name")
    custom_input: Final = custom.get("input")
    custom_arguments: Final = custom_input if isinstance(custom_input, str) else _arguments(custom_input)
    call_id: Final = fields.get("id")
    return {
        "id": call_id if isinstance(call_id, str) and call_id else fallback_id,
        "type": "function",
        "function": {
            "name": name if isinstance(name, str) else "",
            "arguments": custom_arguments if is_custom else _arguments(function.get("arguments")),
        },
    }


def _legacy_tool_calls(function_call: object, call_id: str | None) -> tuple[Mapping[str, object], ...]:
    fields: Final = _as_mapping(function_call)
    if not fields or call_id is None:
        return ()
    name: Final = fields.get("name")
    return (
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name if isinstance(name, str) else "",
                "arguments": _arguments(fields.get("arguments")),
            },
        },
    )


def _without_media(content: object) -> object:
    parts: Final = _as_items(content)
    if not parts:
        return content
    return tuple(part for part in parts if not _is_media_part(part)) or None


def _normalize_message(message: Mapping[str, object], index: int, legacy_call_id: str | None) -> Mapping[str, object]:
    is_function_result: Final = message.get("role") == "function"
    raw_tool_calls: Final = _as_items(message.get("tool_calls"))
    tool_calls: Final = (
        tuple(_wire_tool_call(call, f"call_{index}_{position}") for position, call in enumerate(raw_tool_calls))
        if raw_tool_calls
        else _legacy_tool_calls(message.get("function_call"), legacy_call_id)
    )
    tool_call_id: Final = message.get("tool_call_id") or (legacy_call_id if is_function_result else None)
    return {
        **{key: value for key, value in message.items() if key not in _REBUILT_MESSAGE_KEYS},
        "role": "tool" if is_function_result else message.get("role"),
        **({"content": _without_media(message.get("content"))} if "content" in message else {}),
        **({"tool_calls": tool_calls} if tool_calls else {}),
        **({"tool_call_id": tool_call_id} if tool_call_id else {}),
    }


def _latest_call_id(previous: str | None, current: str | None) -> str | None:
    return current or previous


def _monitor_wire_messages(messages: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    """Messages in the shape Cygnal's CreateMonitorRequest validates.

    Legacy `function_call` turns and their `function` results become tool calls and tool results
    paired by a synthesized id, custom tool calls become function calls, and media parts are dropped.
    """
    legacy_call_ids: Final = tuple(
        accumulate(
            (f"call_legacy_{index}" if "function_call" in message else None for index, message in enumerate(messages)),
            _latest_call_id,
        )
    )
    return tuple(
        _normalize_message(message, index, legacy_call_ids[index]) if _needs_normalization(message) else message
        for index, message in enumerate(messages)
    )


def _is_string_mapping(value: object) -> bool:
    mapping: Final = _as_mapping(value)
    return isinstance(value, Mapping) and all(isinstance(text, str) for text in mapping.values())


def _is_categories(value: object) -> bool:
    items: Final = _as_items(value)
    is_array: Final = isinstance(value, (list, tuple))
    return _is_string_mapping(value) or (
        is_array and (all(isinstance(item, str) for item in items) or all(_is_string_mapping(item) for item in items))
    )


def _is_string(value: object) -> bool:
    return isinstance(value, str)


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _invalid_dynamic_params(dynamic_body: Mapping[str, object], reasoning_modes: frozenset[str]) -> tuple[str, ...]:
    def is_reasoning_mode(value: object) -> bool:
        return isinstance(value, str) and value in reasoning_modes

    checks: Final[tuple[tuple[str, Callable[[object], bool]], ...]] = (
        ("reasoning_mode", is_reasoning_mode),
        ("categories", _is_categories),
        ("policy_id", _is_string),
        ("metadata", _is_mapping),
    )
    return tuple(name for name, is_valid in checks if name in dynamic_body and not is_valid(dynamic_body[name]))


def _monitor_headers(configured: object) -> Mapping[str, str]:
    if configured is None:
        return MappingProxyType({})
    if not isinstance(configured, Mapping):
        raise GraySwanGuardrailConfigError("Gray Swan Guardrail: monitor_headers must map header names to strings")
    entries: Final = tuple(cast("Mapping[object, object]", configured).items())  # cast-ok: checked as a Mapping above
    if not all(isinstance(value, str) for _, value in entries):
        raise GraySwanGuardrailConfigError("Gray Swan Guardrail: monitor_headers values must be strings")
    return MappingProxyType(
        {str(name).lower(): str(value) for name, value in entries if str(name).lower() not in _RESERVED_MONITOR_HEADERS}
    )


def _block_status_code(configured: object) -> int:
    try:
        status: Final = 400 if configured is None else int(cast("int | str", configured))  # cast-ok: parsed below
    except (TypeError, ValueError) as exc:
        raise GraySwanGuardrailConfigError(f"Gray Swan Guardrail: invalid block_status_code {configured!r}") from exc
    if status not in GRAYSWAN_BLOCK_STATUS_CODES:
        raise GraySwanGuardrailConfigError(
            f"Gray Swan Guardrail: block_status_code must be one of {sorted(GRAYSWAN_BLOCK_STATUS_CODES)} "
            "so router retries and fallbacks treat the block as a guardrail intervention"
        )
    return status


def _chat_shape_tools(handler_cls: type["BaseTranslation"], raw_tools: Sequence[object]) -> tuple[object, ...]:
    from litellm.llms.anthropic.chat.guardrail_translation.handler import AnthropicMessagesHandler
    from litellm.llms.anthropic.pass_through.adapters.transformation import (
        LiteLLMAnthropicMessagesAdapter,
        is_provider_native_tool_dict,
    )
    from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
    from litellm.responses.litellm_completion_transformation.transformation import (
        LiteLLMCompletionResponsesConfig,
    )

    if issubclass(handler_cls, AnthropicMessagesHandler):
        translated, _ = LiteLLMAnthropicMessagesAdapter().translate_anthropic_tools_to_openai(
            cast("list[AllAnthropicToolsValues]", list(raw_tools))  # cast-ok: client tools, the adapter reads them
        )
        return tuple(tool for tool in translated if not is_provider_native_tool_dict(tool))
    if issubclass(handler_cls, OpenAIResponsesHandler):
        forms: Final = LiteLLMCompletionResponsesConfig.responses_tools_to_chat_forms(
            tuple(_as_plain_dict(tool) for tool in raw_tools)
        )
        return tuple(chain.from_iterable(form.chat_tools for form in forms))
    return tuple(raw_tools)


class GraySwanGuardrailMissingSecrets(Exception):
    """Raised when the Gray Swan API key is missing."""


class GraySwanGuardrailConfigError(Exception):
    """Raised for an invalid Gray Swan configuration, so the proxy refuses to start rather than skip the guardrail."""


class GraySwanGuardrailBlocked(HTTPException):
    """A Gray Swan block. `code` is the machine-readable marker surfaces without a structured body still carry."""

    code: Final = GRAYSWAN_BLOCK_TYPE


class GraySwanGuardrailAPIError(Exception):
    """Raised when the Gray Swan API returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class GraySwanGuardrail(CustomGuardrail):
    """
    Guardrail that calls Gray Swan's Cygnal monitoring endpoint.

    Uses the unified guardrail system via `apply_guardrail` method,
    which automatically works with all LiteLLM endpoints:
    - OpenAI Chat Completions
    - OpenAI Responses API
    - OpenAI Text Completions
    - Anthropic Messages
    - Image Generation
    - And more...

    see: https://docs.grayswan.ai/cygnal/monitor-requests
    """

    SUPPORTED_ON_FLAGGED_ACTIONS = {"block", "monitor", "passthrough"}
    DEFAULT_ON_FLAGGED_ACTION = "monitor"
    BASE_API_URL = "https://api.grayswan.ai"
    MONITOR_PATH = "/cygnal/monitor"
    SUPPORTED_REASONING_MODES = {"off", "hybrid", "thinking"}

    def __init__(
        self,
        guardrail_name: str | None = "grayswan",
        api_key: str | None = None,
        api_base: str | None = None,
        on_flagged_action: str | None = None,
        violation_threshold: float | None = None,
        reasoning_mode: str | None = None,
        categories: dict[str, str] | None = None,
        policy_id: str | None = None,
        streaming_end_of_stream_only: bool = False,
        streaming_sampling_rate: int = 5,
        streaming_buffer_until_moderated: bool | None = True,
        fail_open: bool | None = True,
        fail_open_input: bool | None = None,
        fail_open_output: bool | None = None,
        guardrail_timeout: float | None = 30.0,
        flag_on_ipi: bool | None = True,
        flag_on_mutation: bool | None = True,
        block_status_code: int | None = 400,
        block_message: str | None = None,
        extra_headers: Sequence[str] | None = None,
        monitor_headers: Mapping[str, str] | None = None,
        **kwargs: Unpack[_CustomGuardrailOptions],
    ) -> None:
        self.async_handler: _GraySwanMonitorHTTPClient = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        api_key_value: Final = api_key or os.getenv("GRAYSWAN_API_KEY")
        if not api_key_value:
            raise GraySwanGuardrailMissingSecrets(
                "Gray Swan API key missing. Set `GRAYSWAN_API_KEY` or pass `api_key`."
            )
        self.api_key: str = api_key_value

        base: Final = api_base or os.getenv("GRAYSWAN_API_BASE") or self.BASE_API_URL
        self.api_base = base.rstrip("/")
        self.monitor_url = f"{self.api_base}{self.MONITOR_PATH}"

        action: Final = on_flagged_action
        if action and action.lower() in self.SUPPORTED_ON_FLAGGED_ACTIONS:
            self.on_flagged_action = action.lower()
        else:
            if action:
                verbose_proxy_logger.warning(
                    "Gray Swan Guardrail: Unsupported on_flagged_action '%s', defaulting to '%s'.",
                    action,
                    self.DEFAULT_ON_FLAGGED_ACTION,
                )
            self.on_flagged_action = self.DEFAULT_ON_FLAGGED_ACTION

        self.block_status_code = _block_status_code(block_status_code)
        self.block_message = block_message or GRAYSWAN_BLOCK_ERROR_MSG

        self.violation_threshold = self._resolve_threshold(violation_threshold)
        self.flag_on_ipi = flag_on_ipi is not False
        self.flag_on_mutation = flag_on_mutation is not False
        self.reasoning_mode = self._resolve_reasoning_mode(reasoning_mode)
        self.categories = categories
        self.policy_id = None if policy_id is None else str(policy_id)
        self.fail_open = True if fail_open is None else bool(fail_open)
        self.fail_open_input = self.fail_open if fail_open_input is None else bool(fail_open_input)
        self.fail_open_output = self.fail_open if fail_open_output is None else bool(fail_open_output)
        self.guardrail_timeout = 30.0 if guardrail_timeout is None else float(guardrail_timeout)
        self.forwarded_header_names = frozenset(
            (*DEFAULT_FORWARDED_HEADERS, *(name.lower() for name in extra_headers or ()))
        )
        self.monitor_headers: Mapping[str, str] = _monitor_headers(monitor_headers)

        # Streaming configuration
        self.streaming_end_of_stream_only = streaming_end_of_stream_only
        self.streaming_sampling_rate = streaming_sampling_rate
        self.streaming_buffer_until_moderated = streaming_buffer_until_moderated is not False

        verbose_proxy_logger.debug(
            "GraySwan __init__: streaming_end_of_stream_only=%s, streaming_sampling_rate=%s",
            streaming_end_of_stream_only,
            streaming_sampling_rate,
        )

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=list(self.get_supported_event_hooks()),
            **kwargs,
        )

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
        ]

    # ------------------------------------------------------------------
    # Debug override to trace post_call issues
    # ------------------------------------------------------------------

    def should_run_guardrail(self, data, event_type) -> bool:
        """Override to add debug logging."""
        result: Final = super().should_run_guardrail(data, event_type)
        # Check if apply_guardrail is in __dict__
        has_apply_guardrail: Final = "apply_guardrail" in type(self).__dict__
        verbose_proxy_logger.debug(
            "GraySwan DEBUG: should_run_guardrail event_type=%s, result=%s, event_hook=%s, has_apply_guardrail=%s, class=%s",
            event_type,
            result,
            self.event_hook,
            has_apply_guardrail,
            type(self).__name__,
        )
        return result

    # ------------------------------------------------------------------
    # Unified Guardrail Interface (works with ALL endpoints automatically)
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        """
        Scan the conversation the unified guardrail system extracted.

        Request scans send the scoped `structured_messages` conversation and `tools`
        (OpenAI chat shape), falling back to the extracted texts as user turns when the
        surface provides no chat-shaped conversation. Response scans send the scoped
        request conversation followed by the model reply.

        Raises:
            HTTPException: If content is blocked (block mode)
            GraySwanGuardrailAPIError: If Gray Swan cannot be reached and the scan fails closed
        """
        is_output: Final = input_type == "response"
        start_time: Final = time.time()
        scan: Final = (
            self._response_scan_input(inputs, request_data, logging_obj)
            if is_output
            else self._request_scan_input(inputs)
        )
        if not scan.messages:
            verbose_proxy_logger.debug("Gray Swan Guardrail: nothing to scan for %s", input_type)
            self._record_scan(request_data, logging_obj, is_output=is_output, outcome="skipped", start_time=start_time)
            return inputs

        dynamic_body: Final = _as_mapping(self.get_guardrail_dynamic_request_body_params(request_data))
        invalid_params: Final = _invalid_dynamic_params(dynamic_body, frozenset(self.SUPPORTED_REASONING_MODES))
        if invalid_params:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": f"Invalid Gray Swan guardrail parameters in extra_body: {', '.join(invalid_params)}",
                    "type": GRAYSWAN_INVALID_PARAMS_TYPE,
                },
            )
        result: Final = await self._monitor_scan(scan, dynamic_body, request_data, logging_obj)
        if isinstance(result, _MonitorFailure):
            return self._handle_monitor_failure(
                result,
                inputs,
                request_data,
                logging_obj,
                is_output=is_output,
                start_time=start_time,
                client_steered=bool(dynamic_body),
            )

        outcome: Final = _FLAGGED_OUTCOME[self.on_flagged_action] if self._is_flagged(result) else "allow"
        self._record_scan(
            request_data,
            logging_obj,
            is_output=is_output,
            outcome=outcome,
            start_time=start_time,
            response_json=result,
        )
        return self._process_response_internal(
            response_json=result,
            request_data=request_data,
            inputs=inputs,
            is_output=is_output,
        )

    def _request_scan_input(self, inputs: GenericGuardrailAPIInputs) -> _ScanInput:
        structured: Final = tuple(_as_plain_dict(message) for message in inputs.get("structured_messages") or ())
        tools: Final = tuple(_as_plain_dict(tool) for tool in inputs.get("tools") or ()) or None
        texts: Final = tuple(_MonitorMessage(role="user", content=text) for text in inputs.get("texts", []))
        if structured and all(_is_chat_message(message) for message in structured):
            return _ScanInput(messages=structured, tools=tools, fallback=texts or None)
        return _ScanInput(messages=texts, tools=tools, fallback=None)

    def _response_scan_input(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> _ScanInput:
        texts: Final = inputs.get("texts", [])
        response_tool_calls: Final = tuple(_as_plain_dict(call) for call in (inputs.get("tool_calls") or ()))
        if not texts and not response_tool_calls:
            return _ScanInput(messages=(), tools=None, fallback=None)
        context, tools = self._post_call_context(
            request_data, logging_obj, effective_scan_only_tool_results_for_guardrail(self)
        )
        merged_tail: Final = (
            _MonitorMessage(role="assistant", content=texts[-1], tool_calls=response_tool_calls)
            if len(texts) == 1 and response_tool_calls
            else None
        )
        reply: Final = (
            *(_MonitorMessage(role="assistant", content=text) for text in (texts[:-1] if merged_tail else texts)),
            *((merged_tail,) if merged_tail else ()),
            *(
                (_MonitorMessage(role="assistant", tool_calls=response_tool_calls),)
                if response_tool_calls and not merged_tail
                else ()
            ),
        )
        return _ScanInput(messages=(*context, *reply), tools=tools, fallback=reply if context else None)

    async def _monitor_scan(
        self,
        scan: _ScanInput,
        dynamic_body: Mapping[str, object],
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> _GraySwanMonitorResponse | _MonitorFailure:
        first: Final = await self._monitor(
            self._prepare_payload(
                _monitor_wire_messages(scan.messages), dynamic_body, request_data, logging_obj, tools=scan.tools
            )
        )
        if not isinstance(first, _MonitorFailure) or not first.rejected_shape or scan.fallback is None:
            return first
        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: monitor rejected the conversation (status_code=%s), rescanning the extracted text",
            first.status_code,
        )
        return await self._monitor(
            self._prepare_payload(_monitor_wire_messages(scan.fallback), dynamic_body, request_data, logging_obj)
        )

    async def _monitor(self, payload: Mapping[str, object]) -> _GraySwanMonitorResponse | _MonitorFailure:
        try:
            return await self._call_grayswan_api(payload)
        except Exception as exc:
            cause: Final = exc.__cause__ or exc
            return _MonitorFailure(error=exc, timed_out=isinstance(cause, (Timeout, httpx.TimeoutException)))

    def _handle_monitor_failure(
        self,
        failure: _MonitorFailure,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],  # mutable-ok: the logging helpers append to the shared request dict
        logging_obj: Optional["LiteLLMLoggingObj"],
        *,
        is_output: bool,
        start_time: float,
        client_steered: bool,
    ) -> GenericGuardrailAPIInputs:
        fail_open: Final = not failure.rejected_payload(client_steered=client_steered) and (
            self.fail_open_output if is_output else self.fail_open_input
        )
        status_code: Final = failure.status_code
        self._record_scan(
            request_data,
            logging_obj,
            is_output=is_output,
            outcome="fail_open" if fail_open else "fail_closed",
            start_time=start_time,
            failure=failure,
        )
        verbose_proxy_logger.error(
            "Gray Swan Guardrail: API request failed%s (timed_out=%s, fail_open=%s): %s",
            f" (status_code={status_code})" if status_code else "",
            failure.timed_out,
            fail_open,
            failure.error,
        )
        if fail_open:
            return inputs
        raise GraySwanGuardrailAPIError(str(failure.error), status_code=status_code) from failure.error

    def _buffers_streams(self) -> bool:
        from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import (
            UnifiedLLMGuardrails,
        )

        configured: Final = UnifiedLLMGuardrails().resolve_streaming_flag(
            self, "streaming_buffer_until_moderated", True
        )
        return configured is not False and not self.mask_response_content

    def _request_stage(self) -> GuardrailEventHooks:
        hooks: Final = self.event_hook if isinstance(self.event_hook, list) else (self.event_hook,)
        only_during_call: Final = GuardrailEventHooks.during_call in hooks and GuardrailEventHooks.pre_call not in hooks
        return GuardrailEventHooks.during_call if only_during_call else GuardrailEventHooks.pre_call

    def _is_flagged(self, response_json: _GraySwanMonitorResponse) -> bool:
        violation_score: Final = float(response_json.get("violation", 0.0) or 0.0)
        return (
            violation_score >= self.violation_threshold
            or (self.flag_on_ipi and bool(response_json.get("ipi")))
            or (self.flag_on_mutation and bool(response_json.get("mutation")))
        )

    def _block_exception(
        self,
        response_json: _GraySwanMonitorResponse,
        violated_rules: object,
        is_output: bool,
    ) -> GraySwanGuardrailBlocked:
        return GraySwanGuardrailBlocked(
            status_code=self.block_status_code,
            detail={
                "error": self.block_message,
                "type": GRAYSWAN_BLOCK_TYPE,
                "violation_location": "output" if is_output else "input",
                "violation": float(response_json.get("violation", 0.0) or 0.0),
                "violated_rules": violated_rules,
                "mutation": response_json.get("mutation"),
                "ipi": response_json.get("ipi"),
                "block_message": response_json.get("block_message"),
                "activity_id": response_json.get("activity_id"),
            },
        )

    def _record_scan(
        self,
        request_data: dict[str, object],  # mutable-ok: the logging helpers append to the shared request dict
        logging_obj: Optional["LiteLLMLoggingObj"],
        *,
        is_output: bool,
        outcome: ScanOutcome,
        start_time: float,
        response_json: _GraySwanMonitorResponse | None = None,
        failure: _MonitorFailure | None = None,
    ) -> None:
        end_time: Final = time.time()
        stage: Final = GuardrailEventHooks.post_call if is_output else self._request_stage()
        activity_id: Final = response_json.get("activity_id") if response_json else None
        verdict: Final = response_json or {}
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response={
                "outcome": outcome,
                "input_type": "response" if is_output else "request",
                "litellm_call_id": request_data.get("litellm_call_id") or getattr(logging_obj, "litellm_call_id", None),
                "activity_id": activity_id,
                "violation": verdict.get("violation"),
                "ipi": verdict.get("ipi"),
                "mutation": verdict.get("mutation"),
                "error": None if failure is None else ("timeout" if failure.timed_out else str(failure.error)),
            },
            request_data=request_data,
            guardrail_status=_OUTCOME_STATUS[outcome],
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            guardrail_provider="grayswan",
            event_type=stage,
        )
        add_guardrail_scan_id(
            request_data,
            activity_id if isinstance(activity_id, str) else None,
            guardrail_name=self.guardrail_name,
            provider="grayswan",
            stage=stage,
        )

    # ------------------------------------------------------------------
    # Legacy Test Interface (for backward compatibility)
    # ------------------------------------------------------------------

    async def run_grayswan_guardrail(self, payload: dict[str, object]) -> _GraySwanMonitorResponse:
        """
        Run the GraySwan guardrail on a payload.

        This is a legacy method for testing purposes.

        Args:
            payload: The payload to scan

        Returns:
            Dict containing the GraySwan API response
        """
        response_json: Final = await self._call_grayswan_api(payload)
        # Call the legacy response processor (for test compatibility)
        self._process_grayswan_response(response_json)
        return response_json

    def _process_grayswan_response(
        self,
        response_json: _GraySwanMonitorResponse,
        data: dict | None = None,
        hook_type: GuardrailEventHooks | None = None,
    ) -> None:
        """
        Legacy method for processing GraySwan API responses.

        This method is maintained for backward compatibility with existing tests.
        It handles the test scenarios where responses need to be processed with
        knowledge of the request context (pre/during/post call hooks).

        Args:
            response_json: Response from GraySwan API
            data: Optional request data (for passthrough exceptions)
            hook_type: Optional GuardrailEventHooks for determining behavior
        """
        violation_score: Final = float(response_json.get("violation", 0.0) or 0.0)
        violated_rules: Final = response_json.get("violated_rules", [])
        mutation_detected: Final = response_json.get("mutation")
        ipi_detected: Final = response_json.get("ipi")

        if not self._is_flagged(response_json):
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: content passed (score=%s, threshold=%s)",
                violation_score,
                self.violation_threshold,
            )
            return

        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: flagged (score=%.3f, threshold=%.3f, ipi=%s, mutation=%s)",
            violation_score,
            self.violation_threshold,
            ipi_detected,
            mutation_detected,
        )

        detection_info: Final = {
            "guardrail": "grayswan",
            "flagged": True,
            "violation_score": violation_score,
            "violated_rules": violated_rules,
            "mutation": mutation_detected,
            "ipi": ipi_detected,
        }

        # Determine if this is input (pre-call/during-call) or output (post-call)
        if hook_type is not None:
            is_input = hook_type in [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ]
        else:
            is_input = True

        if self.on_flagged_action == "block":
            raise self._block_exception(response_json, violated_rules, is_output=not is_input)
        elif self.on_flagged_action == "passthrough":
            # For passthrough mode, we need to handle violations
            detections: Final = [detection_info]
            violation_message: Final = self._format_violation_message(detections, is_output=not is_input)
            verbose_proxy_logger.info("Gray Swan Guardrail: Passthrough mode - handling violation")

            # If hook_type is provided and in pre/during call, raise exception
            if hook_type in [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ]:
                # Raise ModifyResponseException to short-circuit LLM call
                if data is None:
                    data = {}
                self.raise_passthrough_exception(
                    violation_message=violation_message,
                    request_data=data,
                    detection_info=detection_info,
                )
            elif hook_type == GuardrailEventHooks.post_call:
                # For post-call, store detection info in metadata
                if data is None:
                    data = {}
                if "metadata" not in data:
                    data["metadata"] = {}
                if "guardrail_detections" not in data["metadata"]:
                    data["metadata"]["guardrail_detections"] = []
                data["metadata"]["guardrail_detections"].append(detection_info)

    # ------------------------------------------------------------------
    # Core GraySwan API interaction
    # ------------------------------------------------------------------

    async def _call_grayswan_api(self, payload: Mapping[str, object]) -> _GraySwanMonitorResponse:
        """Call the GraySwan monitoring API."""
        headers: Final = self._prepare_headers()

        try:
            response: Final = await self.async_handler.post(
                url=self.monitor_url,
                headers=headers,
                json=dict(payload),
                timeout=self.guardrail_timeout,
            )
            response.raise_for_status()
            result: Final = response.json()
            verbose_proxy_logger.debug("Gray Swan Guardrail: monitor response %s", safe_dumps(result))
            return result
        except Exception as exc:
            status_code: Final = getattr(exc, "status_code", None) or getattr(exc, "exception_status_code", None)
            raise GraySwanGuardrailAPIError(str(exc), status_code=status_code) from exc

    def _process_response_internal(
        self,
        response_json: _GraySwanMonitorResponse,
        request_data: dict,
        inputs: GenericGuardrailAPIInputs,
        is_output: bool,
    ) -> GenericGuardrailAPIInputs:
        """
        Process GraySwan API response and handle violations.

        Args:
            response_json: Response from GraySwan API
            request_data: Original request data
            inputs: The inputs being scanned
            is_output: True if scanning model output, False for input

        Returns:
            GenericGuardrailAPIInputs - possibly modified with violation message

        Raises:
            HTTPException: If content is blocked (block mode)
        """
        violation_score: Final = float(response_json.get("violation", 0.0) or 0.0)
        violated_rules: Final = response_json.get("violated_rule_descriptions", [])
        mutation_detected: Final = response_json.get("mutation")
        ipi_detected: Final = response_json.get("ipi")

        if not self._is_flagged(response_json):
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: content passed (score=%s, threshold=%s)",
                violation_score,
                self.violation_threshold,
            )
            return inputs

        verbose_proxy_logger.warning(
            "Gray Swan Guardrail: flagged (score=%.3f, threshold=%.3f, ipi=%s, mutation=%s)",
            violation_score,
            self.violation_threshold,
            ipi_detected,
            mutation_detected,
        )

        detection_info: Final = {
            "guardrail": "grayswan",
            "flagged": True,
            "violation_score": violation_score,
            "violated_rules": violated_rules,
            "mutation": mutation_detected,
            "ipi": ipi_detected,
        }

        if self.on_flagged_action == "block":
            raise self._block_exception(response_json, violated_rules, is_output=is_output)
        elif self.on_flagged_action == "monitor":
            verbose_proxy_logger.info("Gray Swan Guardrail: Monitoring mode - allowing flagged content")
            return inputs
        elif self.on_flagged_action == "passthrough":
            # Replace content with violation message
            violation_message: Final = self._format_violation_message(detection_info, is_output=is_output)
            verbose_proxy_logger.info(
                "Gray Swan Guardrail: Passthrough mode - replacing content with violation message"
            )

            if is_output and request_data.get("stream") is True and not self._buffers_streams():
                raise self._block_exception(response_json, violated_rules, is_output=is_output)
            if not is_output or request_data.get("stream") is True:
                self.raise_passthrough_exception(
                    violation_message=violation_message,
                    request_data=request_data,
                    detection_info=detection_info,
                )

            # For post-call (response), replace texts and let unified system apply them
            inputs["texts"] = [violation_message]
            return inputs

        return inputs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_headers(self) -> dict[str, str]:
        return {
            **self.monitor_headers,
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
            "grayswan-api-key": self.api_key,
        }

    def _extract_inbound_headers(
        self,
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> Mapping[str, str] | None:
        litellm_params: Final = _as_mapping(
            _as_mapping(getattr(logging_obj, "model_call_details", None)).get("litellm_params")
        )
        candidates: Final = (
            _as_mapping(request_data.get("proxy_server_request")).get("headers"),
            request_data.get("headers"),
            _as_mapping(request_data.get("metadata")).get("headers"),
            _as_mapping(litellm_params.get("metadata")).get("headers"),
        )
        headers: Final = _as_mapping(next((candidate for candidate in candidates if candidate), None))
        forwarded_headers: Final = {
            name.lower(): str(value) for name, value in headers.items() if self._forwards_header(name.lower())
        }
        return forwarded_headers or None

    def _forwards_header(self, name: str) -> bool:
        return (
            name in self.forwarded_header_names
            and not _is_credential_header(name)
            and name != _configured_litellm_key_header()
        )

    def _monitor_metadata(
        self,
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"],
        dynamic_metadata: object,
    ) -> Mapping[str, object]:
        headers: Final = self._extract_inbound_headers(request_data, logging_obj) or {}
        session_id: Final = next((headers[name] for name in SESSION_ID_HEADERS if headers.get(name)), None)
        return {
            **({"headers": headers} if headers else {}),
            **({"session_id": session_id} if session_id else {}),
            **_as_mapping(dynamic_metadata),
        }

    def _forwarded_litellm_metadata(
        self,
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"],
    ) -> Mapping[str, object]:
        source: Final = {
            **_as_mapping(request_data.get("metadata")),
            **_as_mapping(request_data.get("litellm_metadata")),
        }
        identity: Final = {
            name: source[name]
            for name in _FORWARDED_LITELLM_METADATA_FIELDS
            if isinstance(source.get(name), (str, int, float, bool))
        }
        call_id: Final = request_data.get("litellm_call_id") or getattr(logging_obj, "litellm_call_id", None)
        model: Final = request_data.get("model")
        return {
            **identity,
            **({"litellm_call_id": call_id} if isinstance(call_id, str) else {}),
            **({"model": model} if isinstance(model, str) else {}),
        }

    def _post_call_context(
        self,
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"],
        scan_only_tool_results: bool,
    ) -> tuple[tuple[Mapping[str, object], ...], tuple[object, ...] | None]:
        """Request conversation in OpenAI shape, scoped like the pre-call path.

        Returns the scoped context messages plus the request's tool definitions in
        OpenAI chat shape, or ``((), None)`` when the request surface cannot be resolved.
        """
        from litellm.litellm_core_utils.api_route_to_call_types import get_call_types_for_route
        from litellm.llms import load_guardrail_translation_mappings

        litellm_metadata: Final = request_data.get("litellm_metadata")
        request_route: Final = (
            litellm_metadata.get("user_api_key_request_route") if isinstance(litellm_metadata, Mapping) else None
        )
        route_call_types: Final = get_call_types_for_route(request_route) if isinstance(request_route, str) else None
        call_type: Final = (
            (route_call_types[0].value if route_call_types else None)
            or (logging_obj.call_type if logging_obj is not None else None)
            or getattr(request_data.get("litellm_logging_obj"), "call_type", None)
        )
        if not isinstance(call_type, str):
            return (), None
        try:
            mapped: Final = CallTypes(call_type)
        except ValueError:
            return (), None
        handler_cls: Final = load_guardrail_translation_mappings().get(mapped)
        if handler_cls is None:
            return (), None
        raw_tools: Final = request_data.get("tools")
        try:
            structured: Final = handler_cls().get_structured_messages(dict(request_data)) or ()
            tools: Final = (
                _chat_shape_tools(handler_cls, raw_tools)
                if not scan_only_tool_results and isinstance(raw_tools, list) and raw_tools
                else ()
            )
        except Exception as exc:
            verbose_proxy_logger.debug(
                "Gray Swan Guardrail: could not resolve request context for call_type %s: %s",
                call_type,
                exc,
            )
            return (), None
        indices: Final = scoped_structured_message_indices(
            structured,
            scan_only_tool_results=scan_only_tool_results,
            skip_system=effective_skip_system_message_for_guardrail(self),
            skip_tool=effective_skip_tool_message_for_guardrail(self),
        )
        if not indices:
            return (), None
        return tuple(_as_plain_dict(structured[index]) for index in indices), tools or None

    def _prepare_payload(
        self,
        messages: Sequence[Mapping[str, object]],
        dynamic_body: Mapping[str, object],
        request_data: Mapping[str, object],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
        *,
        tools: Sequence[object] | None = None,
    ) -> Mapping[str, object]:
        categories: Final = dynamic_body.get("categories") or self.categories
        policy_id: Final = dynamic_body.get("policy_id") or self.policy_id
        reasoning_mode: Final = dynamic_body.get("reasoning_mode") or self.reasoning_mode
        metadata: Final = self._monitor_metadata(request_data, logging_obj, dynamic_body.get("metadata"))
        litellm_metadata: Final[object] = safe_json_loads(
            safe_dumps(self._forwarded_litellm_metadata(request_data, logging_obj)), default={}
        )
        return {
            "messages": messages,
            **({"tools": tools} if tools else {}),
            **({"categories": categories} if categories else {}),
            **({"policy_id": policy_id} if policy_id else {}),
            **({"reasoning_mode": reasoning_mode} if reasoning_mode else {}),
            **({"metadata": metadata} if metadata else {}),
            **(
                {"litellm_metadata": litellm_metadata}
                if isinstance(litellm_metadata, dict) and litellm_metadata
                else {}
            ),
        }

    def _format_violation_message(self, detection_info: object, is_output: bool = False) -> str:
        """
        Format detection info into a user-friendly violation message.

        Args:
            detection_info: Can be either:
                - A single dict with violation_score, violated_rules, mutation, ipi keys
                - A list of such dicts (legacy format)
            is_output: True if violation is in model output, False if in input

        Returns:
            Formatted violation message string
        """
        # Handle legacy format where detection_info is a list
        if isinstance(detection_info, list) and len(detection_info) > 0:
            detection_info = detection_info[0]

        # Extract fields from detection_info dict
        detection_dict: Final[dict] = detection_info if isinstance(detection_info, dict) else {}
        violation_score: Final = detection_dict.get("violation_score", 0.0)
        violated_rules: Final = detection_dict.get("violated_rules", [])
        mutation: Final = detection_dict.get("mutation", False)
        ipi: Final = detection_dict.get("ipi", False)

        violation_location: Final = "model response" if is_output else "input query"

        message_parts: Final = [
            f"Sorry I can't help with that. According to the Gray Swan Cygnal Guardrail, "
            f"the {violation_location} has a violation score of {violation_score:.2f}.",
        ]

        if violated_rules:
            formatted_rules: Final = self._format_violated_rules(violated_rules)
            if formatted_rules:
                message_parts.append(f"It was violating the rule(s): {formatted_rules}.")

        if mutation:
            message_parts.append("Mutation effort to make the harmful intention disguised was DETECTED.")

        if ipi:
            message_parts.append("Indirect Prompt Injection was DETECTED.")

        return "\n".join(message_parts)

    def _format_violated_rules(self, violated_rules: list) -> str:
        """Format violated rules list into a readable string."""
        formatted: Final[list[str]] = []
        for rule in violated_rules:
            if isinstance(rule, dict):
                # New format: {'rule': 6, 'name': 'Illegal Activities...', 'description': '...'}
                rule_num = rule.get("rule", "")
                rule_name = rule.get("name", "")
                rule_desc = rule.get("description", "")
                if rule_num and rule_name:
                    if rule_desc:
                        formatted.append(f"#{rule_num} {rule_name}: {rule_desc}")
                    else:
                        formatted.append(f"#{rule_num} {rule_name}")
                elif rule_name:
                    formatted.append(rule_name)
                else:
                    formatted.append(str(rule))
            else:
                # Legacy format: simple value
                formatted.append(str(rule))

        return ", ".join(formatted)

    def _resolve_threshold(self, value: float | None) -> float:
        if value is not None:
            return float(value)
        env_val: Final = os.getenv("GRAYSWAN_VIOLATION_THRESHOLD")
        if env_val:
            try:
                return float(env_val)
            except ValueError:
                pass
        return 0.5

    def _resolve_reasoning_mode(self, value: str | None) -> str | None:
        if value and value.lower() in self.SUPPORTED_REASONING_MODES:
            return value.lower()
        env_val: Final = os.getenv("GRAYSWAN_REASONING_MODE")
        if env_val and env_val.lower() in self.SUPPORTED_REASONING_MODES:
            return env_val.lower()
        return None
