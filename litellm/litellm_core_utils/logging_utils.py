import asyncio
import copy
import functools
import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_logger
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
    Choices,
    Delta,
    Function,
    FunctionCall,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    TextCompletionResponse,
    Usage,
)

from litellm.litellm_core_utils.streaming_chunk_builder_utils import (
    concatenate_base64_list,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm import ModelResponse as _ModelResponse
    from litellm.litellm_core_utils.litellm_logging import (
        Logging as LiteLLMLoggingObject,
    )

    LiteLLMModelResponse = _ModelResponse
    Span = Union[_Span, Any]
else:
    LiteLLMModelResponse = Any
    LiteLLMLoggingObject = Any
    Span = Any


import litellm

"""
Helper utils used for logging callbacks
"""

# Global service logger instance to avoid recreating it
_service_logger = None


def _get_service_logger():
    """Get or create the global ServiceLogging instance"""
    global _service_logger
    if _service_logger is None:
        from litellm._service_logger import ServiceLogging

        _service_logger = ServiceLogging()
    return _service_logger


class StreamingAccumulator:
    """Aggregate streaming chunks without storing individual objects."""

    def __init__(self, messages: Optional[List[Dict[str, Any]]] = None) -> None:
        self.messages = messages
        self.clear()

    def clear(self) -> None:
        self._base_id: Optional[str] = None
        self._base_object: Optional[str] = None
        self._base_created: Optional[int] = None
        self._base_model: Optional[str] = None
        self._system_fingerprint: Optional[str] = None
        self._index: int = 0
        self._role: Optional[str] = None
        self._finish_reason: Optional[str] = None
        self._content_parts: List[str] = []
        self._reasoning_parts: List[str] = []
        self._thinking_text_parts: List[str] = []
        self._thinking_signature: Optional[str] = None
        self._thinking_redacted_data: Optional[str] = None
        self._thinking_redacted_type: Optional[str] = None
        self._tool_calls: Dict[int, Dict[str, Any]] = {}
        self._function_call_name: Optional[str] = None
        self._function_call_arguments: List[str] = []
        self._audio_data_parts: List[str] = []
        self._audio_transcript_parts: List[str] = []
        self._audio_expires_at: Optional[int] = None
        self._audio_id: Optional[str] = None
        self._images: List[Any] = []
        self._annotations: List[Any] = []
        self._provider_specific_fields: Dict[str, Any] = {}
        self._usage_data: Optional[Dict[str, Any]] = None
        self._hidden_params: Dict[str, Any] = {}
        self._response_headers: Optional[Dict[str, Any]] = None
        self._final_response: Optional[Union[ModelResponse, TextCompletionResponse]] = None
        self._has_updates: bool = False

    def update(self, chunk: Union[ModelResponse, ModelResponseStream, Any]) -> None:
        if chunk is None:
            return

        if isinstance(chunk, (ModelResponse, TextCompletionResponse)):
            self._final_response = chunk
            self._has_updates = True
            return

        if not isinstance(chunk, ModelResponseStream):
            return

        if self._base_id is None:
            self._base_id = getattr(chunk, "id", None)
        if self._base_object is None:
            self._base_object = getattr(chunk, "object", None)
        if self._base_created is None:
            self._base_created = getattr(chunk, "created", None)
        if self._base_model is None:
            self._base_model = getattr(chunk, "model", None)

        if getattr(chunk, "system_fingerprint", None):
            self._system_fingerprint = chunk.system_fingerprint

        hidden_params = getattr(chunk, "_hidden_params", {}) or {}
        if hidden_params:
            for key, value in hidden_params.items():
                if key == "usage":
                    usage_value = value
                    if isinstance(usage_value, Usage):
                        usage_value = usage_value.model_dump()
                    elif hasattr(usage_value, "model_dump"):
                        usage_value = usage_value.model_dump()  # type: ignore[call-arg]
                    else:
                        usage_value = copy.deepcopy(usage_value)
                    if usage_value is not None:
                        self._usage_data = usage_value
                    continue
                self._hidden_params[key] = copy.deepcopy(value)

        usage_obj = getattr(chunk, "usage", None)
        if usage_obj is not None:
            if isinstance(usage_obj, Usage):
                self._usage_data = usage_obj.model_dump()
            elif hasattr(usage_obj, "model_dump"):
                self._usage_data = usage_obj.model_dump()  # type: ignore[call-arg]
            else:
                self._usage_data = copy.deepcopy(usage_obj)

        if getattr(chunk, "_response_headers", None):
            self._response_headers = copy.deepcopy(chunk._response_headers)

        for choice in getattr(chunk, "choices", []):
            if choice is None:
                continue
            self._index = getattr(choice, "index", self._index)
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is not None:
                self._finish_reason = finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "role", None):
                self._role = delta.role

            content_piece = getattr(delta, "content", None)
            if isinstance(content_piece, str):
                self._content_parts.append(content_piece)
            elif isinstance(content_piece, list):
                for part in content_piece:
                    if isinstance(part, str):
                        self._content_parts.append(part)
                    elif isinstance(part, dict):
                        self._content_parts.append(json.dumps(part))

            reasoning_piece = getattr(delta, "reasoning_content", None)
            if isinstance(reasoning_piece, str):
                self._reasoning_parts.append(reasoning_piece)

            thinking_blocks = getattr(delta, "thinking_blocks", None)
            if thinking_blocks:
                for block in thinking_blocks:
                    block_type = None
                    thinking_value = None
                    signature = None
                    data_value = None
                    if isinstance(block, dict):
                        block_type = block.get("type")
                        thinking_value = block.get("thinking")
                        signature = block.get("signature")
                        data_value = block.get("data")
                    else:
                        block_type = getattr(block, "type", None)
                        thinking_value = getattr(block, "thinking", None)
                        signature = getattr(block, "signature", None)
                        data_value = getattr(block, "data", None)

                    if block_type == "redacted_thinking":
                        if data_value:
                            self._thinking_redacted_data = data_value
                            self._thinking_redacted_type = block_type
                    else:
                        if isinstance(thinking_value, str):
                            self._thinking_text_parts.append(thinking_value)
                        if isinstance(signature, str):
                            self._thinking_signature = signature

            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call is None:
                        continue
                    index = getattr(tool_call, "index", 0)
                    entry = self._tool_calls.setdefault(
                        index,
                        {
                            "id": None,
                            "type": None,
                            "function_name": None,
                            "arguments": [],
                        },
                    )
                    tool_id = getattr(tool_call, "id", None)
                    if tool_id:
                        entry["id"] = tool_id
                    tool_type = getattr(tool_call, "type", None)
                    if tool_type:
                        entry["type"] = tool_type
                    function = getattr(tool_call, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        if name:
                            entry["function_name"] = name
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            entry.setdefault("arguments", []).append(arguments)

            function_call = getattr(delta, "function_call", None)
            if function_call is not None:
                name = getattr(function_call, "name", None)
                if name:
                    self._function_call_name = name
                arguments = getattr(function_call, "arguments", None)
                if arguments:
                    self._function_call_arguments.append(arguments)

            audio = getattr(delta, "audio", None)
            if audio:
                if isinstance(audio, ChatCompletionAudioResponse):
                    if isinstance(audio.data, str):
                        self._audio_data_parts.append(audio.data)
                    if isinstance(audio.transcript, str):
                        self._audio_transcript_parts.append(audio.transcript)
                    if isinstance(audio.expires_at, int):
                        self._audio_expires_at = audio.expires_at
                    if isinstance(audio.id, str):
                        self._audio_id = audio.id
                elif isinstance(audio, dict):
                    data_val = audio.get("data")
                    transcript_val = audio.get("transcript")
                    expires_at_val = audio.get("expires_at")
                    audio_id_val = audio.get("id")
                    if isinstance(data_val, str):
                        self._audio_data_parts.append(data_val)
                    if isinstance(transcript_val, str):
                        self._audio_transcript_parts.append(transcript_val)
                    if isinstance(expires_at_val, int):
                        self._audio_expires_at = expires_at_val
                    if isinstance(audio_id_val, str):
                        self._audio_id = audio_id_val

            images = getattr(delta, "images", None)
            if images:
                for image in images:
                    if hasattr(image, "model_dump"):
                        self._images.append(image.model_dump())
                    else:
                        self._images.append(copy.deepcopy(image))

            annotations = getattr(delta, "annotations", None)
            if annotations:
                for annotation in annotations:
                    if hasattr(annotation, "model_dump"):
                        self._annotations.append(annotation.model_dump())
                    else:
                        self._annotations.append(copy.deepcopy(annotation))

            provider_fields = getattr(delta, "provider_specific_fields", None)
            if provider_fields:
                if isinstance(provider_fields, dict):
                    self._provider_specific_fields.update(
                        copy.deepcopy(provider_fields)
                    )

        self._has_updates = True

    def has_data(self) -> bool:
        return self._has_updates

    def get_accumulated_content(self) -> str:
        return "".join(self._content_parts)

    def current_usage(self) -> Optional[Usage]:
        if self._usage_data is None:
            return None
        return Usage(**self._usage_data)

    def _build_stream_chunk(self) -> Optional[ModelResponseStream]:
        if not self._has_updates or self._final_response is not None:
            return None

        delta_kwargs: Dict[str, Any] = {}
        if self._role:
            delta_kwargs["role"] = self._role
        if self._content_parts:
            delta_kwargs["content"] = "".join(self._content_parts)
        if self._reasoning_parts:
            delta_kwargs["reasoning_content"] = "".join(self._reasoning_parts)

        thinking_blocks: List[Dict[str, Any]] = []
        if self._thinking_redacted_data:
            thinking_blocks.append(
                {
                    "type": self._thinking_redacted_type or "redacted_thinking",
                    "data": self._thinking_redacted_data,
                }
            )
        elif self._thinking_text_parts:
            block: Dict[str, Any] = {
                "type": "thinking",
                "thinking": "".join(self._thinking_text_parts),
            }
            if self._thinking_signature:
                block["signature"] = self._thinking_signature
            thinking_blocks.append(block)
        if thinking_blocks:
            delta_kwargs["thinking_blocks"] = thinking_blocks

        tool_calls: List[ChatCompletionDeltaToolCall] = []
        for index in sorted(self._tool_calls):
            data = self._tool_calls[index]
            function_name = data.get("function_name")
            arguments_list = data.get("arguments", [])
            arguments_text = "".join(arguments_list) if arguments_list else ""
            function = Function(
                name=function_name,
                arguments=arguments_text or "{}",
            )
            tool_calls.append(
                ChatCompletionDeltaToolCall(
                    id=data.get("id"),
                    type=data.get("type"),
                    index=index,
                    function=function,
                )
            )
        if tool_calls:
            delta_kwargs["tool_calls"] = tool_calls

        if self._function_call_name or self._function_call_arguments:
            delta_kwargs["function_call"] = FunctionCall(
                name=self._function_call_name,
                arguments="".join(self._function_call_arguments) or "{}",
            )

        if self._audio_data_parts or self._audio_transcript_parts or self._audio_id:
            delta_kwargs["audio"] = ChatCompletionAudioResponse(
                data=concatenate_base64_list(self._audio_data_parts)
                if self._audio_data_parts
                else None,
                transcript="".join(self._audio_transcript_parts),
                expires_at=self._audio_expires_at,
                id=self._audio_id,
            )

        if self._images:
            delta_kwargs["images"] = self._images
        if self._annotations:
            delta_kwargs["annotations"] = self._annotations
        if self._provider_specific_fields:
            delta_kwargs["provider_specific_fields"] = self._provider_specific_fields

        delta = Delta(**delta_kwargs)
        streaming_choice = StreamingChoices(
            index=self._index,
            finish_reason=self._finish_reason,
            delta=delta,
        )
        chunk = ModelResponseStream(
            id=self._base_id,
            object=self._base_object or "chat.completion.chunk",
            created=self._base_created,
            model=self._base_model,
            system_fingerprint=self._system_fingerprint,
            choices=[streaming_choice],
        )

        hidden_params = copy.deepcopy(self._hidden_params)
        if self._usage_data is not None:
            hidden_params["usage"] = Usage(**self._usage_data)
            setattr(chunk, "usage", Usage(**self._usage_data))
        if hidden_params:
            chunk._hidden_params = hidden_params
        if self._response_headers is not None:
            chunk._response_headers = copy.deepcopy(self._response_headers)

        return chunk

    def finalize(
        self,
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        logging_obj: Optional[Any] = None,
    ) -> Optional[Union[ModelResponse, TextCompletionResponse]]:
        if not self._has_updates:
            return None

        if self._final_response is not None:
            response = self._final_response
            self.clear()
            return response

        chunk = self._build_stream_chunk()
        if chunk is None:
            self.clear()
            return None

        try:
            response = litellm.stream_chunk_builder(
                chunks=[chunk],
                messages=messages or self.messages,
                start_time=start_time,
                end_time=end_time,
                logging_obj=logging_obj,
            )
        except Exception as exc:
            verbose_logger.exception(
                f"Error building stream chunk from accumulator: {str(exc)}"
            )
            response = None

        self.clear()
        return response

def _get_parent_otel_span_from_logging_obj(
    logging_obj: Optional[LiteLLMLoggingObject] = None,
) -> Optional[Span]:
    """
    Extract the parent OTEL span from the logging object using existing helper.

    Args:
        logging_obj: The LiteLLM logging object containing model call details

    Returns:
        The parent OTEL span if found, None otherwise
    """
    try:
        if logging_obj is None or not hasattr(logging_obj, "model_call_details"):
            return None

        # Reuse existing function by passing model_call_details as kwargs
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )

        return _get_parent_otel_span_from_kwargs(logging_obj.model_call_details)

    except Exception as e:
        verbose_logger.exception(
            f"Error in _get_parent_otel_span_from_logging_obj: {str(e)}"
        )
        return None


def convert_litellm_response_object_to_str(
    response_obj: Union[Any, LiteLLMModelResponse],
) -> Optional[str]:
    """
    Get the string of the response object from LiteLLM

    """
    if isinstance(response_obj, litellm.ModelResponse):
        response_str = ""
        for choice in response_obj.choices:
            if isinstance(choice, litellm.Choices):
                if choice.message.content and isinstance(choice.message.content, str):
                    response_str += choice.message.content
        return response_str

    return None


def _assemble_complete_response_from_streaming_chunks(
    result: Union[ModelResponse, TextCompletionResponse, ModelResponseStream],
    start_time: datetime,
    end_time: datetime,
    request_kwargs: dict,
    accumulator: StreamingAccumulator,
    is_async: bool,
):
    """
    Assemble a complete response from a streaming chunks

    - assemble a complete streaming response if result.choices[0].finish_reason is not None
    - else append the chunk to the streaming_chunks


    Args:
        result: ModelResponse
        start_time: datetime
        end_time: datetime
        request_kwargs: dict
        streaming_chunks: List[Any]
        is_async: bool

    Returns:
        Optional[Union[ModelResponse, TextCompletionResponse]]: Complete streaming response

    """
    complete_streaming_response: Optional[
        Union[ModelResponse, TextCompletionResponse]
    ] = None

    if isinstance(result, ModelResponse):
        return result

    accumulator.update(result)

    if not isinstance(result, ModelResponseStream):
        return None

    if result.choices[0].finish_reason is not None:  # if it's the last chunk
        try:
            complete_streaming_response = accumulator.finalize(
                messages=request_kwargs.get("messages", None),
                start_time=start_time,
                end_time=end_time,
            )
        except Exception as e:
            log_message = (
                "Error occurred building stream chunk in {} success logging: {}".format(
                    "async" if is_async else "sync", str(e)
                )
            )
            verbose_logger.exception(log_message)
            complete_streaming_response = None
    return complete_streaming_response


def _set_duration_in_model_call_details(
    logging_obj: Any,  # we're not guaranteed this will be `LiteLLMLoggingObject`
    start_time: datetime,
    end_time: datetime,
):
    """Helper to set duration in model_call_details, with error handling"""
    try:
        duration_ms = (end_time - start_time).total_seconds() * 1000
        if logging_obj and hasattr(logging_obj, "model_call_details"):
            logging_obj.model_call_details["llm_api_duration_ms"] = duration_ms
        else:
            verbose_logger.debug(
                "`logging_obj` not found - unable to track `llm_api_duration_ms"
            )
    except Exception as e:
        verbose_logger.warning(f"Error setting `llm_api_duration_ms`: {str(e)}")


def track_llm_api_timing():
    """
    Decorator to track LLM API call timing for both sync and async functions.
    The logging_obj is expected to be passed as an argument to the decorated function.
    Logs timing using ServiceLogging similar to Redis cache.
    """

    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_time_float = time.time()
            logging_obj = kwargs.get("logging_obj", None)

            # Extract parent OTEL span from logging object
            parent_otel_span = _get_parent_otel_span_from_logging_obj(logging_obj)

            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                end_time_float = time.time()
                duration = end_time_float - start_time_float

                # Set duration in model call details
                _set_duration_in_model_call_details(
                    logging_obj=logging_obj,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Log timing using ServiceLogging (like Redis cache)
                try:
                    from litellm.types.services import ServiceTypes

                    service_logger = _get_service_logger()

                    # Get function name for call_type
                    call_type = f"{func.__name__} <- track_llm_api_timing"

                    # Create async task for service logging (similar to Redis cache pattern)
                    asyncio.create_task(
                        service_logger.async_service_success_hook(
                            service=ServiceTypes.LITELLM,
                            duration=duration,
                            call_type=call_type,
                            start_time=start_time_float,
                            end_time=end_time_float,
                            parent_otel_span=parent_otel_span,
                        )
                    )
                except Exception as e:
                    verbose_logger.debug(f"Error in service logging: {str(e)}")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = datetime.now()
            start_time_float = time.time()
            logging_obj = kwargs.get("logging_obj", None)

            # Extract parent OTEL span from logging object
            parent_otel_span = _get_parent_otel_span_from_logging_obj(logging_obj)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = datetime.now()
                end_time_float = time.time()
                duration = end_time_float - start_time_float

                # Set duration in model call details
                _set_duration_in_model_call_details(
                    logging_obj=logging_obj,
                    start_time=start_time,
                    end_time=end_time,
                )

                # Log timing using ServiceLogging (like Redis cache)
                try:
                    from litellm.types.services import ServiceTypes

                    service_logger = _get_service_logger()

                    # Get function name for call_type
                    call_type = f"{func.__name__} <- track_llm_api_timing"

                    # Use sync service logging for sync functions
                    service_logger.service_success_hook(
                        service=ServiceTypes.LITELLM,
                        duration=duration,
                        call_type=call_type,
                        start_time=start_time_float,
                        end_time=end_time_float,
                        parent_otel_span=parent_otel_span,
                    )
                except Exception as e:
                    verbose_logger.debug(f"Error in service logging: {str(e)}")

        # Check if the function is async or sync
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
