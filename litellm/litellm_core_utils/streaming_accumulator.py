"""Utilities for accumulating streaming responses without retaining every chunk."""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    FunctionCall,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    TextCompletionResponse,
    Usage,
)


@dataclass
class _ContentAccumulator:
    text_segments: List[str] = field(default_factory=list)
    structured_parts: List[Any] = field(default_factory=list)
    has_structured_content: bool = False
    text_buffer: List[str] = field(default_factory=list)

    def add_text(self, text: str) -> None:
        if not text:
            return
        self.text_buffer.append(text)
        if self.has_structured_content:
            self.structured_parts.append(text)
        else:
            self.text_segments.append(text)

    def _ensure_structured(self) -> None:
        if not self.has_structured_content:
            self.has_structured_content = True
            if self.text_segments:
                self.structured_parts.extend(self.text_segments)
                self.text_segments = []

    def add_structured(self, part: Any, text_value: Optional[str]) -> None:
        if part is None:
            return
        self._ensure_structured()
        self.structured_parts.append(part)
        if text_value:
            self.text_buffer.append(text_value)

    def delta_content(self) -> Optional[Union[str, List[Any]]]:
        if self.has_structured_content and self.structured_parts:
            return copy.deepcopy(self.structured_parts)
        if self.text_segments:
            return "".join(self.text_segments)
        return None

    def accumulated_text(self) -> str:
        return "".join(self.text_buffer)

    def reset(self) -> None:
        self.text_segments.clear()
        self.structured_parts.clear()
        self.has_structured_content = False
        self.text_buffer.clear()


@dataclass
class _ThinkingAccumulator:
    text_parts: List[str] = field(default_factory=list)
    signature: Optional[str] = None
    redacted_data: Optional[str] = None
    redacted_type: Optional[str] = None

    def add_text(self, text: str) -> None:
        if text:
            self.text_parts.append(text)

    def set_signature(self, signature: Optional[str]) -> None:
        if signature:
            self.signature = signature

    def set_redacted(self, data: Optional[str], block_type: Optional[str]) -> None:
        if data:
            self.redacted_data = data
            self.redacted_type = block_type

    def to_blocks(self) -> List[Dict[str, Any]]:
        if self.redacted_data:
            return [
                {
                    "type": self.redacted_type or "redacted_thinking",
                    "data": self.redacted_data,
                }
            ]
        if self.text_parts:
            block: Dict[str, Any] = {
                "type": "thinking",
                "thinking": "".join(self.text_parts),
            }
            if self.signature:
                block["signature"] = self.signature
            return [block]
        return []

    def reset(self) -> None:
        self.text_parts.clear()
        self.signature = None
        self.redacted_data = None
        self.redacted_type = None


@dataclass
class _ToolCallState:
    id: Optional[str] = None
    type: Optional[str] = None
    function_name: Optional[str] = None
    arguments: List[str] = field(default_factory=list)

    def add_arguments(self, arguments: Optional[str]) -> None:
        if arguments:
            self.arguments.append(arguments)


@dataclass
class _AudioAccumulator:
    data_parts: List[str] = field(default_factory=list)
    transcript_parts: List[str] = field(default_factory=list)
    expires_at: Optional[int] = None
    audio_id: Optional[str] = None

    def update_from_response(self, audio: ChatCompletionAudioResponse) -> None:
        if isinstance(audio.data, str):
            self.data_parts.append(audio.data)
        if isinstance(audio.transcript, str):
            self.transcript_parts.append(audio.transcript)
        if isinstance(audio.expires_at, int):
            self.expires_at = audio.expires_at
        if isinstance(audio.id, str):
            self.audio_id = audio.id

    def update_from_dict(self, audio_dict: Dict[str, Any]) -> None:
        data_val = audio_dict.get("data")
        transcript_val = audio_dict.get("transcript")
        expires_at_val = audio_dict.get("expires_at")
        audio_id_val = audio_dict.get("id")
        if isinstance(data_val, str):
            self.data_parts.append(data_val)
        if isinstance(transcript_val, str):
            self.transcript_parts.append(transcript_val)
        if isinstance(expires_at_val, int):
            self.expires_at = expires_at_val
        if isinstance(audio_id_val, str):
            self.audio_id = audio_id_val

    def build(self) -> Optional[ChatCompletionAudioResponse]:
        if not (self.data_parts or self.transcript_parts or self.audio_id):
            return None
        data_value = (
            _concatenate_base64_parts(self.data_parts)
            if self.data_parts
            else None
        )
        return ChatCompletionAudioResponse(
            data=data_value,
            transcript="".join(self.transcript_parts),
            expires_at=self.expires_at,
            id=self.audio_id,
        )

    def reset(self) -> None:
        self.data_parts.clear()
        self.transcript_parts.clear()
        self.expires_at = None
        self.audio_id = None


def _concatenate_base64_parts(parts: List[str]) -> str:
    """Concatenate base64-encoded strings without importing the chunk builder."""

    decoded_segments: List[bytes] = []
    for part in parts:
        if not isinstance(part, str) or not part:
            continue
        try:
            decoded_segments.append(base64.b64decode(part))
        except Exception:
            # Skip invalid segments but continue combining whatever we have.
            continue

    if not decoded_segments:
        return ""

    return base64.b64encode(b"".join(decoded_segments)).decode("utf-8")


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
        self._content = _ContentAccumulator()
        self._thinking = _ThinkingAccumulator()
        self._tool_calls: Dict[int, _ToolCallState] = {}
        self._reasoning_parts: List[str] = []
        self._function_call_name: Optional[str] = None
        self._function_call_arguments: List[str] = []
        self._audio = _AudioAccumulator()
        self._images: List[Any] = []
        self._annotations: List[Any] = []
        self._provider_specific_fields: Dict[str, Any] = {}
        self._usage_data: Optional[Dict[str, Any]] = None
        self._hidden_params: Dict[str, Any] = {}
        self._response_headers: Optional[Dict[str, Any]] = None
        self._final_response: Optional[Union[ModelResponse, TextCompletionResponse]] = None
        self._has_updates: bool = False

    def _normalize_structured_content(self, value: Any) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except TypeError:
                return value.model_dump()  # type: ignore[misc]
        if isinstance(value, dict):
            return copy.deepcopy(value)
        try:
            return copy.deepcopy(value)
        except Exception:
            return value

    def _extract_text_from_content(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            for key in ("text", "output_text", "input_text", "content"):
                text_value = value.get(key)
                if isinstance(text_value, str):
                    return text_value
            annotations = value.get("annotations") if isinstance(value, dict) else None
            if isinstance(annotations, list):
                collected = "".join(
                    annotation.get("text", "")
                    for annotation in annotations
                    if isinstance(annotation, dict)
                    and isinstance(annotation.get("text"), str)
                )
                if collected:
                    return collected
        return None

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
                self._content.add_text(content_piece)
            elif isinstance(content_piece, list):
                for part in content_piece:
                    normalized_part = self._normalize_structured_content(part)
                    if normalized_part is None:
                        continue
                    text_value = self._extract_text_from_content(normalized_part)
                    self._content.add_structured(normalized_part, text_value)
            elif content_piece is not None:
                normalized_part = self._normalize_structured_content(content_piece)
                if normalized_part is not None:
                    text_value = self._extract_text_from_content(normalized_part)
                    self._content.add_structured(normalized_part, text_value)

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
                        self._thinking.set_redacted(data_value, block_type)
                    else:
                        if isinstance(thinking_value, str):
                            self._thinking.add_text(thinking_value)
                        if isinstance(signature, str):
                            self._thinking.set_signature(signature)

            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call is None:
                        continue
                    index = getattr(tool_call, "index", 0)
                    entry = self._tool_calls.setdefault(index, _ToolCallState())
                    tool_id = getattr(tool_call, "id", None)
                    if tool_id:
                        entry.id = tool_id
                    tool_type = getattr(tool_call, "type", None)
                    if tool_type:
                        entry.type = tool_type
                    function = getattr(tool_call, "function", None)
                    if function is not None:
                        name = getattr(function, "name", None)
                        if name:
                            entry.function_name = name
                        arguments = getattr(function, "arguments", None)
                        if arguments:
                            entry.add_arguments(arguments)

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
                    self._audio.update_from_response(audio)
                elif isinstance(audio, dict):
                    self._audio.update_from_dict(audio)

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
                if hasattr(provider_fields, "model_dump"):
                    try:
                        provider_fields = provider_fields.model_dump()
                    except TypeError:
                        provider_fields = provider_fields.model_dump()  # type: ignore[misc]
                if isinstance(provider_fields, dict):
                    self._provider_specific_fields.update(
                        copy.deepcopy(provider_fields)
                    )

        self._has_updates = True

    def has_data(self) -> bool:
        return self._has_updates

    def get_accumulated_content(self) -> str:
        return self._content.accumulated_text()

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

        content_value = self._content.delta_content()
        if content_value is not None:
            delta_kwargs["content"] = content_value

        if self._reasoning_parts:
            delta_kwargs["reasoning_content"] = "".join(self._reasoning_parts)

        thinking_blocks = self._thinking.to_blocks()
        if thinking_blocks:
            delta_kwargs["thinking_blocks"] = thinking_blocks

        tool_calls: List[ChatCompletionDeltaToolCall] = []
        for index in sorted(self._tool_calls):
            state = self._tool_calls[index]
            function = Function(
                name=state.function_name,
                arguments="".join(state.arguments) if state.arguments else None,
            )
            delta = Delta(
                id=state.id,
                type=state.type,
                function=function,
            )
            tool_calls.append(
                ChatCompletionDeltaToolCall(
                    index=index,
                    id=state.id,
                    type=state.type,
                    function=FunctionCall(
                        name=state.function_name,
                        arguments="".join(state.arguments)
                        if state.arguments
                        else None,
                    ),
                    delta=delta,
                )
            )

        if tool_calls:
            delta_kwargs["tool_calls"] = tool_calls

        if self._function_call_name or self._function_call_arguments:
            delta_kwargs["function_call"] = FunctionCall(
                name=self._function_call_name,
                arguments="".join(self._function_call_arguments)
                if self._function_call_arguments
                else None,
            )

        audio_value = self._audio.build()
        if audio_value is not None:
            delta_kwargs["audio"] = audio_value

        if self._images:
            delta_kwargs["images"] = copy.deepcopy(self._images)

        if self._annotations:
            delta_kwargs["annotations"] = copy.deepcopy(self._annotations)

        if self._provider_specific_fields:
            delta_kwargs["provider_specific_fields"] = copy.deepcopy(
                self._provider_specific_fields
            )

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
        usage_obj: Optional[Usage] = None
        if self._usage_data is not None:
            usage_obj = Usage(**self._usage_data)
            hidden_params["usage"] = usage_obj
            setattr(chunk, "usage", usage_obj)
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


__all__ = ["StreamingAccumulator"]
