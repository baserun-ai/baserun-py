import asyncio
import atexit
import json
import logging
import os
from datetime import datetime
from multiprocessing import Process, Queue
from queue import Empty
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

import httpx

from baserun.utils import count_prompt_tokens, deep_merge

if TYPE_CHECKING:
    try:
        from baserun.wrappers.openai import (
            WrappedAsyncStream,
            WrappedChatCompletion,
            WrappedOpenAIBaseClient,
            WrappedSyncStream,
        )
    except ImportError:
        pass

    try:
        from baserun.wrappers.anthropic import WrappedAnthropicBaseClient, WrappedMessage
    except ImportError:
        pass

logger = logging.getLogger(__name__)

exporter_queue: Queue = Queue()
tasks: List[asyncio.Task] = []  # Make tasks a global variable
exporter_process: Union[Process, None] = None


async def post_and_log_response(
    base_url: str, api_key: str, endpoint: str, data: Dict[str, Any], client: httpx.AsyncClient
):
    response = await client.post(
        f"{base_url}/api/public/{endpoint}",
        headers={"Authorization": f"Bearer {api_key}"},
        json=data,
    )
    if response.status_code != 200:
        logger.warning(f"Response from {base_url}/api/public/{endpoint}: {response.status_code}")
        logger.warning(response.json())


async def worker(queue: Queue, base_url: str, api_key: str):
    logger = logging.getLogger(__name__ + ".worker")
    logger.debug(f"Starting worker with base_url: {base_url}")
    try:
        async with httpx.AsyncClient(http2=True) as client:
            while True:
                try:
                    item: Dict = queue.get_nowait()
                except Empty:
                    await asyncio.sleep(1)
                    continue

                if item is None:
                    break

                try:
                    endpoint = item.pop("endpoint")
                    data = item.pop("data")
                    logger.debug(f"Submitting {data} to Baserun at {base_url}/api/public/{endpoint}")

                    task = asyncio.create_task(post_and_log_response(base_url, api_key, endpoint, data, client))
                    task.add_done_callback(lambda t: tasks.remove(t))
                    task.add_done_callback(lambda t: logger.debug(f"Task {t} finished"))
                    tasks.append(task)
                except Exception as e:
                    logger.warning(f"Failed to submit {endpoint} to Baserun: {e}")

    finally:
        logger.debug(f"Waiting for {len(tasks)} tasks to finish")
        await asyncio.gather(*tasks)
        logger.debug("Exiting worker")
        loop = asyncio.get_running_loop()
        loop.stop()


def run_worker(queue: Queue, base_url: str, api_key: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(worker(queue, base_url, api_key))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


def start_worker(base_url: str, api_key: str):
    global exporter_process
    if exporter_process is None:
        exporter_process = Process(
            target=run_worker,
            args=(exporter_queue, base_url, api_key),
        )
        exporter_process.daemon = False
        exporter_process.start()


def stop_worker():
    global exporter_process, exporter_queue, tasks

    # Signal the worker to stop
    if exporter_queue is not None and not getattr(exporter_queue, "_closed"):
        exporter_queue.put(None)

    if exporter_process is not None and exporter_process.is_alive():
        loop = asyncio.get_event_loop()
        if tasks:
            loop.run_until_complete(asyncio.gather(*tasks))

        # Terminate the process
        exporter_process.terminate()
        exporter_process.join()

    # Close and join the queue
    if exporter_queue is not None and not getattr(exporter_queue, "_closed"):
        exporter_queue.close()
        exporter_queue.join_thread()


class ApiClient:
    def __init__(
        self,
        client: Union["WrappedOpenAIBaseClient", "WrappedAnthropicBaseClient"],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.environment = os.getenv("BASERUN_ENVIRONMENT", os.getenv("ENVIRONMENT", "production"))
        self.client = client

        atexit.register(self.exit_handler)

        start_worker(
            base_url or os.getenv("BASERUN_API_URL") or "https://app.baserun.ai",
            api_key or os.getenv("BASERUN_API_KEY") or "",
        )

    def exit_handler(self, *args) -> None:
        stop_worker()

    def submit_completion(self, completion: Union["WrappedChatCompletion", "WrappedMessage"]):
        dict_items = completion.model_dump()
        dict_items.pop("client", None)

        choices = dict_items.pop("choices", [])
        contents = dict_items.pop("content", None)
        if not choices and contents:
            for content in contents:
                choices.append({"message": {"content": content.get("text", ""), "role": "assistant"}})

        usage = dict_items.pop("usage", {})
        if usage:
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            dict_items["usage"] = {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": usage.get("total_tokens", completion_tokens + prompt_tokens),
            }

        start_timestamp = dict_items.pop("start_timestamp", datetime.now())
        first_token_timestamp = dict_items.pop("first_token_timestamp", datetime.now())
        end_timestamp = dict_items.pop("end_timestamp", datetime.now())

        input_messages = []
        for message in dict_items.pop("input_messages", []):
            message_dict = {**message}
            if "tool_calls" in message:
                tool_calls = message.pop("tool_calls", [])
                message_dict["tool_calls"] = [*tool_calls]

            content = message.get("content", None)
            if isinstance(content, Iterator):
                message_dict["content"] = json.dumps([c for c in content])

            input_messages.append(message_dict)

        data = {
            **dict_items,
            "choices": choices,
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
        logger.debug(f"Submitting completion:\n{json.dumps(data, indent=2)}")
        self._post("completions", data)

    def submit_stream(self, stream: Union["WrappedAsyncStream", "WrappedSyncStream"]):
        if not stream.id:
            return

        choices = []
        usage = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        if hasattr(stream, "captured_choices"):
            merged_choice = deep_merge([c.model_dump() for c in stream.captured_choices])
            merged_choice.pop("delta", None)
            merged_delta = deep_merge([c.delta.model_dump() for c in stream.captured_choices])
            merged_choice["message"] = merged_delta
            choices.append(merged_choice)
            usage = {
                "completion_tokens": len(stream.captured_choices) + 1,
                "prompt_tokens": count_prompt_tokens(stream.input_messages),
                "total_tokens": count_prompt_tokens(stream.input_messages) + len(stream.captured_choices) + 1,
            }
        elif hasattr(stream, "captured_messages"):
            for message in stream.captured_messages:
                usage["completion_tokens"] += message.usage.output_tokens
                usage["prompt_tokens"] += message.usage.input_tokens
                choices.append(message.model_dump())

        usage["total_tokens"] = usage["completion_tokens"] + usage["prompt_tokens"]

        config_params = stream.config_params or {}
        config_params.pop("event_handler", None)

        data = {
            "id": stream.id,
            "name": stream.name,
            "completion_id": stream.completion_id,
            "model": config_params.get("model"),
            "trace_id": stream.trace_id,
            "tags": [tag.model_dump() for tag in stream.tags],
            "tool_results": stream.tool_results,
            "evals": [e.model_dump() for e in stream.evals if e.score is not None],
            "input_messages": stream.input_messages,
            "choices": choices,
            "usage": usage,
            "config_params": config_params,
            "start_timestamp": stream.start_timestamp.isoformat(),
            "end_timestamp": stream.end_timestamp.isoformat() if stream.end_timestamp else None,
            "first_token_timestamp": stream.first_token_timestamp.isoformat() if stream.first_token_timestamp else None,
            "request_id": self.client.request_ids.get(stream.id),
            "environment": self.environment,
            "trace": self._trace_data(),
        }
        logger.debug(f"Submitting streamed completion:\n{json.dumps(data, indent=2)}")
        self._post("completions", data)

    def submit_trace(self):
        data = self._trace_data()

        logger.debug(f"Submitting trace:\n{json.dumps(data, indent=2)}")
        self._post("traces", data)

    def _post(self, endpoint: str, data: Dict[str, Any]):
        exporter_queue.put({"endpoint": endpoint, "data": data})

    def _trace_data(self) -> Dict[str, Any]:
        return {
            "id": self.client.trace_id,
            "name": self.client.name,
            "tags": [tag.model_dump() for tag in self.client.tags],
            "evals": [e.model_dump() for e in self.client.evals if e.score is not None],
            "environment": self.environment,
            "output": self.client.output,
            "start_timestamp": self.client.start_timestamp.isoformat(),
            "end_timestamp": self.client.end_timestamp.isoformat() if self.client.end_timestamp else None,
            "error": self.client.error,
            "end_user_identifier": self.client.user,
            "end_user_session_identifier": self.client.session,
            "metadata": self.client.metadata,
        }
