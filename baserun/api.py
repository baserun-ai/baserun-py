import asyncio
import atexit
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

if TYPE_CHECKING:
    from baserun.wrappers.generic import GenericClient, GenericCompletion

logger = logging.getLogger(__name__)

exporter_queue: Queue = Queue()
tasks: List[asyncio.Task] = []
exporter_process: Union[Process, None] = None


async def post_and_log_response(
    base_url: str, api_key: str, endpoint: str, data: Dict[str, Any], client: httpx.AsyncClient
):
    response = await client.post(
        f"{base_url}/api/public/{endpoint}",
        headers={"Authorization": f"Bearer {api_key}"},
        json=data,
    )
    if response.status_code < 200 or response.status_code >= 300:
        logger.warning(f"Response from {base_url}/api/public/{endpoint}: {response.status_code}")
        logger.debug("Text: " + response.text)
        logger.debug("Data: " + json.dumps(data, indent=2))


async def worker(queue: Queue, base_url: str, api_key: str):
    logger = logging.getLogger(__name__ + ".worker")
    logger.debug(f"Starting worker with base_url: {base_url}")
    requests_in_progress: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    tasks = []

    try:
        async with httpx.AsyncClient(http2=True) as client:
            while True:
                item: Dict = await queue.get()

                if item is None:
                    break

                id = item.get("completion_id", item.get("trace_id"))
                endpoint = item.pop("endpoint")
                data = item.pop("data")

                async with requests_in_progress[id]:
                    logger.debug(f"Submitting {data} to Baserun at {base_url}/api/public/{endpoint}")
                    try:
                        task = asyncio.create_task(post_and_log_response(base_url, api_key, endpoint, data, client))
                        tasks.append(task)
                        task.add_done_callback(lambda t: tasks.remove(t))
                        task.add_done_callback(lambda t: logger.debug(f"Task {t} finished"))
                    except Exception as e:
                        logger.warning(f"Failed to submit {endpoint}/{id} to Baserun: {e}")

    finally:
        logger.debug(f"Waiting for {len(tasks)} tasks to finish")
        await asyncio.gather(*tasks, return_exceptions=True)
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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.environment = os.getenv("BASERUN_ENVIRONMENT", os.getenv("ENVIRONMENT", "production"))

        atexit.register(self.exit_handler)

        start_worker(
            base_url or os.getenv("BASERUN_API_URL") or "https://app.baserun.ai",
            api_key or os.getenv("BASERUN_API_KEY") or "",
        )

    def exit_handler(self, *args) -> None:
        stop_worker()

    def submit_completion(self, completion: "GenericCompletion"):
        data = self._completion_data(completion)
        logger.debug(f"Submitting completion:\n{json.dumps(data, indent=2)}")
        post("completions", data)

    def submit_stream(self, stream: "GenericCompletion"):
        if not stream.id:
            return

        data = self._completion_data(stream)
        logger.debug(f"Submitting streamed completion:\n{json.dumps(data, indent=2)}")
        post("completions", data)

    def submit_trace(self, client: "GenericClient"):
        data = self._trace_data(client)

        logger.debug(f"Submitting trace:\n{json.dumps(data, indent=2)}")
        post("traces", data)

    def _completion_data(self, completion: "GenericCompletion") -> Dict[str, Any]:
        start_timestamp = completion.start_timestamp or datetime.now()
        first_token_timestamp = completion.first_token_timestamp or datetime.now()
        end_timestamp = completion.end_timestamp or datetime.now()

        config_params = completion.config_params or {}
        config_params.pop("event_handler", None)

        return {
            "id": completion.id,
            "completion_id": completion.completion_id,
            "choices": [choice.model_dump() for choice in completion.choices],
            "usage": completion.usage.model_dump() if completion.usage else None,
            "model": config_params.get("model"),
            "name": completion.name,
            "trace_id": completion.trace_id,
            "tool_results": completion.tool_results,
            "evals": [e.model_dump() for e in completion.evals if e.score is not None],
            "tags": [tag.model_dump() for tag in completion.tags],
            "input_messages": [m.model_dump() for m in completion.input_messages],
            "request_id": completion.request_id,
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat() if end_timestamp else None,
            "first_token_timestamp": first_token_timestamp.isoformat() if first_token_timestamp else None,
            "config_params": config_params,
            "environment": self.environment,
            "error": completion.error,
            "trace": self._trace_data(completion.client.genericize()),
            "template": completion.template,
        }

    def _trace_data(self, client: "GenericClient") -> Dict[str, Any]:
        return {
            "id": client.trace_id,
            "name": client.name,
            "tags": [tag.model_dump() for tag in client.tags],
            "evals": [e.model_dump() for e in client.evals if e.score is not None],
            "environment": self.environment,
            "output": client._output,
            "start_timestamp": client.start_timestamp.isoformat() if client.start_timestamp else None,
            "end_timestamp": client.end_timestamp.isoformat() if client.end_timestamp else None,
            "error": client.error,
            "end_user_identifier": client.user,
            "end_user_session_identifier": client.session,
            "metadata": client.metadata,
        }


def post(endpoint: str, data: Dict[str, Any]):
    exporter_queue.put({"endpoint": endpoint, "data": data})
