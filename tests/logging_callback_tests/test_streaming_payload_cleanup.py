import asyncio
import importlib.util
import os
import sys
from datetime import datetime

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repo_root)

for module_name in list(sys.modules.keys()):
    if module_name.startswith("litellm"):
        sys.modules.pop(module_name)

spec = importlib.util.spec_from_file_location(
    "litellm", os.path.join(repo_root, "litellm", "__init__.py")
)
litellm = importlib.util.module_from_spec(spec)
litellm.__path__ = [os.path.join(repo_root, "litellm")]
sys.modules["litellm"] = litellm
spec.loader.exec_module(litellm)

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import CallTypes


def test_streaming_payload_cleanup_waits_for_async_callbacks():
    asyncio.run(_run_streaming_cleanup_assertion())


async def _run_streaming_cleanup_assertion() -> None:
    original_success_callbacks = litellm.success_callback
    original_async_callbacks = litellm._async_success_callback
    litellm.success_callback = []
    litellm._async_success_callback = []
    try:
        sync_done = asyncio.Event()
        async_callback_started = asyncio.Event()
        observed_async_payloads = []

        async def async_callback(kwargs, response_obj, start_time, end_time):
            async_callback_started.set()
            await sync_done.wait()
            observed_async_payloads.append(
                kwargs.get("async_complete_streaming_response")
            )

        logging_obj = Logging(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            call_type=CallTypes.acompletion.value,
            start_time=datetime.now(),
            litellm_call_id="test-call",
            function_id="test-function",
            dynamic_async_success_callbacks=[async_callback],
        )

        logging_obj.mark_async_success_pending()

        chunk_payload = {
            "id": "chatcmpl-test",
            "choices": [
                litellm.utils.StreamingChoices(
                    finish_reason="stop",
                    delta=litellm.utils.Delta(
                        content="streamed response",
                        function_call=None,
                        role=None,
                        tool_calls=None,
                    ),
                    index=0,
                    logprobs=None,
                )
            ],
            "created": 0,
            "model": "gpt-3.5-turbo",
            "object": "chat.completion.chunk",
            "system_fingerprint": None,
            "usage": None,
        }
        chunk = litellm.ModelResponseStream(**chunk_payload)
        logging_obj.streaming_accumulator.update(chunk)
        logging_obj.sync_streaming_accumulator.update(chunk)

        start_time = datetime.now()
        end_time = datetime.now()

        async_task = asyncio.create_task(
            logging_obj.async_success_handler(
                result=chunk, start_time=start_time, end_time=end_time
            )
        )
        await async_callback_started.wait()

        sync_task = asyncio.create_task(
            asyncio.to_thread(
                logging_obj.success_handler,
                chunk,
                start_time,
                end_time,
            )
        )

        await sync_task
        sync_done.set()
        await async_task

        assert observed_async_payloads, "Async callback did not receive payload"
        async_payload = observed_async_payloads[0]
        assert isinstance(async_payload, litellm.ModelResponse)
        assert async_payload.choices[0].message.content == "streamed response"

        assert (
            "async_complete_streaming_response"
            not in logging_obj.model_call_details
        )
        assert (
            "complete_streaming_response"
            not in logging_obj.model_call_details
        )
    finally:
        litellm.success_callback = original_success_callbacks
        litellm._async_success_callback = original_async_callbacks
