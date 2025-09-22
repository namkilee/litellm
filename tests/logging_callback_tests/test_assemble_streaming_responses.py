"""
Testing for _assemble_complete_response_from_streaming_chunks

- Test 1 - ModelResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
- Test 2 - TextCompletionResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
- Test 3 - Have multiple lists of streaming chunks, Assert that chunks are added to the correct list and that complete_streaming_response is None. After final chunk sent assert complete_streaming_response is not None
- Test 4 - build a complete response when 1 chunk is poorly formatted

"""

import asyncio
import importlib.util
import os
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))
sys.path.insert(0, ROOT_DIR)  # Adds the parent directory to the system path

for _module_name in list(sys.modules):
    if _module_name.startswith("litellm"):
        del sys.modules[_module_name]

spec = importlib.util.spec_from_file_location(
    "litellm",
    os.path.join(ROOT_DIR, "litellm/__init__.py"),
    submodule_search_locations=[os.path.join(ROOT_DIR, "litellm")],
)
_litellm_module = importlib.util.module_from_spec(spec)
sys.modules["litellm"] = _litellm_module
assert spec.loader is not None
spec.loader.exec_module(_litellm_module)

try:
    import respx  # noqa: F401

    RESPX_AVAILABLE = True
except ModuleNotFoundError:
    RESPX_AVAILABLE = False


import pytest

import litellm
from litellm import (
    Choices,
    Message,
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
    TextChoices,
)

from litellm.litellm_core_utils.logging_utils import (
    _assemble_complete_response_from_streaming_chunks,
)
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_accumulator import StreamingAccumulator
from litellm.types.utils import CallTypes


@pytest.mark.skipif(
    not RESPX_AVAILABLE, reason="respx is required for streaming assembly tests"
)
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_1(is_async):
    """
    Test 1 - ModelResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
    """

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    accumulator = StreamingAccumulator(messages=request_kwargs["messages"])
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert accumulator.has_data() is True
    assert accumulator.get_accumulated_content() == "hello in response"

    # Add final chunk
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                finish_reason="stop",
                delta=litellm.utils.Delta(
                    content="end of response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator,
        is_async=is_async,
    )

    print("complete_streaming_response", complete_streaming_response)

    # this is the 2nd chunk - complete_streaming_response should not be None
    assert complete_streaming_response is not None

    assert isinstance(complete_streaming_response, ModelResponse)
    assert isinstance(complete_streaming_response.choices[0], Choices)
    assert accumulator.has_data() is False
    assert accumulator.get_accumulated_content() == ""

    pass


@pytest.mark.skipif(
    not RESPX_AVAILABLE, reason="respx is required for streaming assembly tests"
)
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_2(is_async):
    """
    Test 2 - TextCompletionResponse with 1 list of streaming chunks. Assert chunks are added to the streaming_chunks, after final chunk sent assert complete_streaming_response is not None
    """

    from litellm.utils import TextCompletionStreamWrapper

    _text_completion_stream_wrapper = TextCompletionStreamWrapper(
        completion_stream=None, model="test_model"
    )

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    accumulator = StreamingAccumulator(messages=request_kwargs["messages"])
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    chunk = _text_completion_stream_wrapper.convert_to_text_completion_object(chunk)

    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert accumulator.has_data() is True

    # Add final chunk
    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                finish_reason="stop",
                delta=litellm.utils.Delta(
                    content="end of response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    chunk = _text_completion_stream_wrapper.convert_to_text_completion_object(chunk)
    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator,
        is_async=is_async,
    )

    print("complete_streaming_response", complete_streaming_response)

    # this is the 2nd chunk - complete_streaming_response should not be None
    assert complete_streaming_response is not None

    assert isinstance(complete_streaming_response, TextCompletionResponse)
    assert isinstance(complete_streaming_response.choices[0], TextChoices)
    assert accumulator.has_data() is False

    pass


@pytest.mark.skipif(
    not RESPX_AVAILABLE, reason="respx is required for streaming assembly tests"
)
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_3(is_async):

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    accumulator_1 = StreamingAccumulator(messages=request_kwargs["messages"])
    accumulator_2 = StreamingAccumulator(messages=request_kwargs["messages"])

    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                delta=litellm.utils.Delta(
                    content="hello in response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)
    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator_1,
        is_async=is_async,
    )

    # this is the 1st chunk - complete_streaming_response should be None

    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert accumulator_1.has_data() is True
    assert accumulator_2.has_data() is False

    # now add a chunk to the 2nd list

    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator_2,
        is_async=is_async,
    )

    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None
    assert accumulator_2.has_data() is True
    assert accumulator_1.has_data() is True

    # finalize the first accumulator with a stop chunk
    final_chunk = ModelResponseStream(
        **{
            "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
            "choices": [
                litellm.utils.StreamingChoices(
                    finish_reason="stop",
                    delta=litellm.utils.Delta(
                        content="end",
                        function_call=None,
                        role=None,
                        tool_calls=None,
                    ),
                )
            ],
            "created": 1721353246,
            "model": "gpt-3.5-turbo",
            "object": "chat.completion.chunk",
            "system_fingerprint": None,
            "usage": None,
        }
    )

    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=final_chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator_1,
        is_async=is_async,
    )

    assert complete_streaming_response is not None
    assert accumulator_1.has_data() is False
    assert accumulator_2.has_data() is True


@pytest.mark.skipif(
    not RESPX_AVAILABLE, reason="respx is required for streaming assembly tests"
)
@pytest.mark.parametrize("is_async", [True, False])
def test_assemble_complete_response_from_streaming_chunks_4(is_async):
    """
    Test 4 - build a complete response when 1 chunk is poorly formatted

    - Assert complete_streaming_response is None
    - Assert accumulator retains the chunk information
    """

    request_kwargs = {
        "model": "test_model",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    }

    accumulator = StreamingAccumulator(messages=request_kwargs["messages"])

    chunk = {
        "id": "chatcmpl-9mWtyDnikZZoB75DyfUzWUxiiE2Pi",
        "choices": [
            litellm.utils.StreamingChoices(
                finish_reason="stop",
                delta=litellm.utils.Delta(
                    content="end of response",
                    function_call=None,
                    role=None,
                    tool_calls=None,
                ),
                index=0,
                logprobs=None,
            )
        ],
        "created": 1721353246,
        "model": "gpt-3.5-turbo",
        "object": "chat.completion.chunk",
        "system_fingerprint": None,
        "usage": None,
    }
    chunk = ModelResponseStream(**chunk)

    # remove attribute id from chunk
    del chunk.object

    complete_streaming_response = _assemble_complete_response_from_streaming_chunks(
        result=chunk,
        start_time=datetime.now(),
        end_time=datetime.now(),
        request_kwargs=request_kwargs,
        accumulator=accumulator,
        is_async=is_async,
    )

    print("complete_streaming_response", complete_streaming_response)
    assert complete_streaming_response is None

    assert accumulator.has_data() is True


def _build_streaming_model_response() -> ModelResponse:
    return ModelResponse(
        model="gpt-3.5-turbo",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="streamed response"),
                finish_reason="stop",
                logprobs=None,
            )
        ],
    )


def _prepare_logging_obj(
    *,
    stream: bool,
    call_type: str,
) -> Logging:
    logging_obj = Logging(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
        call_type=call_type,
        litellm_call_id="test-call-id",
        function_id="test-function-id",
        start_time=datetime.now(),
    )
    logging_obj.model_call_details.update(
        {
            "messages": logging_obj.messages,
            "input": logging_obj.messages,
            "stream": stream,
            "call_type": call_type,
            "optional_params": {},
        }
    )
    return logging_obj


def _assert_streaming_keys_pruned(logging_obj: Logging) -> None:
    for key in (
        "complete_streaming_response",
        "async_complete_streaming_response",
        "standard_logging_object",
        "complete_response",
    ):
        assert key not in logging_obj.model_call_details


def test_success_handler_prunes_streaming_payloads(monkeypatch):
    monkeypatch.setattr(litellm, "success_callback", [])
    logging_obj = _prepare_logging_obj(
        stream=True, call_type=CallTypes.completion.value
    )

    response = _build_streaming_model_response()

    logging_obj.success_handler(result=response)

    _assert_streaming_keys_pruned(logging_obj)


def test_async_success_handler_prunes_streaming_payloads(monkeypatch):
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    logging_obj = _prepare_logging_obj(
        stream=True, call_type=CallTypes.acompletion.value
    )

    response = _build_streaming_model_response()

    asyncio.run(logging_obj.async_success_handler(result=response))

    _assert_streaming_keys_pruned(logging_obj)
