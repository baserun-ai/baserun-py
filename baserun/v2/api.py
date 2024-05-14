import asyncio
import atexit
import json
import logging
import os
import signal
import sys
from queue import Empty, Queue
from threading import Lock, Thread
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import httpx
import tiktoken
from openai.types.chat import ChatCompletionMessageParam

from baserun.v2.utils import deep_merge

if TYPE_CHECKING:
    from baserun.v2.wrappers.openai import (
        WrappedAsyncStream,
        WrappedChatCompletion,
        WrappedOpenAIBaseClient,
        WrappedSyncStream,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

exporter_queue: Queue = Queue()
exporter_thread: Union[Thread, None] = None
thread_lock = Lock()


def count_prompt_tokens(
    messages: list[ChatCompletionMessageParam],
    encoder="cl100k_base",
):
    """FYI this doesn't count "name" keys correctly"""
    return sum([count_message_tokens(json.dumps(list(m.values())), encoder=encoder) for m in messages]) + (
        len(messages) * 3
    )


def count_message_tokens(text: str, encoder="cl100k_base"):
    return len(tiktoken.get_encoding(encoder).encode(text))


async def worker(queue: Queue, base_url: str, api_key: str):
    logger.debug(f"Starting worker with base_url: {base_url}")
    async with httpx.AsyncClient(http2=True) as client:
        while True:
            try:
                item: Dict = queue.get(timeout=1)
            except Empty:
                continue

            if item is None:
                break

            try:
                endpoint = item.pop("endpoint")
                data = item.pop("data")
                logger.debug(f"Submitting {data} to Baserun at {base_url}/api/public/{endpoint}")
                result = await client.post(
                    f"{base_url}/api/public/{endpoint}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=data,
                )
                logger.debug(f"Got response from Baserun: {result} {result.headers}")
            except Exception as e:
                logger.warning(f"Failed to submit {endpoint} to Baserun: {e}")
            finally:
                queue.task_done()


def start_worker(base_url: str, api_key: str):
    global exporter_thread
    with thread_lock:
        if exporter_thread is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            exporter_thread = Thread(target=loop.run_until_complete, args=(worker(exporter_queue, base_url, api_key),))
            exporter_thread.daemon = True
            exporter_thread.start()


class ApiClient:
    def __init__(
        self,
        client: Union["WrappedOpenAIBaseClient"],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.environment = os.getenv("BASERUN_ENVIRONMENT", os.getenv("ENVIRONMENT", "production"))
        self.client = client

        def signal_function(sig, frame):
            self.exit_handler(sig, frame)

        signal.signal(signal.SIGINT, signal_function)
        signal.signal(signal.SIGTERM, signal_function)
        atexit.register(self.exit_handler)

        start_worker(
            base_url or os.getenv("BASERUN_API_URL") or "https://app.baserun.ai",
            api_key or os.getenv("BASERUN_API_KEY") or "",
        )

    def exit_handler(self, *args) -> None:
        logger.debug("Baserun exiting")
        exporter_queue.put(None)

        try_count = 0
        while not exporter_queue.empty() and try_count < 5:
            logger.debug("Baserun finishing export of spans")
            sleep(0.25)
            try_count += 1

        if len(args) > 1:
            sys.exit(0)

    def submit_completion(self, completion: "WrappedChatCompletion"):
        dict_items = completion.model_dump()
        dict_items.pop("client", None)
        start_timestamp = dict_items.pop("start_timestamp", None)
        end_timestamp = dict_items.pop("end_timestamp", None)
        first_token_timestamp = dict_items.pop("first_token_timestamp", None)
        input_messages = []
        for message in dict_items.pop("input_messages", []):
            tool_calls = message.pop("tool_calls", [])
            input_messages.append({**message, "tool_calls": [*tool_calls]})

        output = {
            **dict_items,
            "name": completion.name,
            "trace_id": completion.trace_id,
            "tool_results": completion.tool_results,
            "evals": [e.model_dump() for e in completion.evals if e.score is not None],
            "tags": [tag.model_dump() for tag in completion.tags],
            "config_params": dict_items.pop("config_params", {}),
            "environment": self.environment,
            "input_messages": input_messages,
            "trace": self._trace_data(),
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat() if end_timestamp else None,
            "first_token_timestamp": first_token_timestamp.isoformat() if first_token_timestamp else None,
            "request_id": self.client.request_ids.get(completion.id),
        }
        try:
            logger.debug(f"Submitting completion:\n{json.dumps(output, indent=2)}")
        except Exception as e:
            import pdb

            pdb.set_trace()
        self._post("completions", output)

    def submit_stream(self, stream: Union["WrappedAsyncStream", "WrappedSyncStream"]):
        if not stream.id:
            return

        merged_choice = deep_merge([c.model_dump() for c in stream.captured_choices])
        merged_choice.pop("delta", None)
        merged_delta = deep_merge([c.delta.model_dump() for c in stream.captured_choices])
        merged_choice["message"] = merged_delta

        output = {
            "id": stream.id,
            "name": stream.name,
            "completion_id": stream.completion_id,
            "model": stream.config_params.get("model"),
            "trace_id": stream.trace_id,
            "tags": [tag.model_dump() for tag in stream.tags],
            "tool_results": stream.tool_results,
            "evals": [e.model_dump() for e in stream.evals if e.score is not None],
            "input_messages": stream.input_messages,
            "choices": [merged_choice],
            "usage": {
                "completion_tokens": len(stream.captured_choices) + 1,
                "prompt_tokens": count_prompt_tokens(stream.input_messages),
                "total_tokens": count_prompt_tokens(stream.input_messages) + len(stream.captured_choices) + 1,
            },
            "config_params": stream.config_params or {},
            "start_timestamp": stream.start_timestamp.isoformat(),
            "end_timestamp": stream.end_timestamp.isoformat() if stream.end_timestamp else None,
            "first_token_timestamp": stream.first_token_timestamp.isoformat() if stream.first_token_timestamp else None,
            "request_id": self.client.request_ids.get(stream.id),
            "environment": self.environment,
            "trace": self._trace_data(),
        }
        logger.debug(f"Submitting streamed completion:\n{json.dumps(output, indent=2)}")
        self._post("completions", output)

    def submit_trace(self):
        output = self._trace_data()

        logger.debug(f"Submitting trace:\n{json.dumps(output, indent=2)}")
        self._post("traces", output)

    def _post(self, endpoint: str, data: Dict[str, Any]):
        exporter_queue.put({"endpoint": endpoint, "data": data})

    def _trace_data(self) -> Dict[str, Any]:
        return {
            "id": self.client.trace_id,
            "name": self.client.name,
            "tags": [tag.model_dump() for tag in self.client.tags],
            "evals": [e.model_dump() for e in self.client.evals if e.score is not None],
            "environment": self.environment,
            "result": self.client.result,
            "start_timestamp": self.client.start_timestamp.isoformat(),
            "end_timestamp": self.client.end_timestamp.isoformat() if self.client.end_timestamp else None,
            "error": self.client.error,
            "end_user_identifier": self.client.user,
            "end_user_session_identifier": self.client.session,
            "metadata": self.client.metadata,
        }
