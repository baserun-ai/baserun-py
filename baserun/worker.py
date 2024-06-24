import asyncio
import json
import logging
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any, Dict, List, Union

import httpx

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

exporter_queue: Queue = Queue()
tasks: List[asyncio.Task] = []
exporter_process: Union[Process, None] = None


async def post_and_log_response(
    base_url: str, api_key: str, endpoint: str, data: Dict[str, Any], client: httpx.AsyncClient
):
    logger = logging.getLogger(__name__ + ".poster")
    try:
        logger.debug(f"Posting to {base_url}/api/public/{endpoint}")
        response = await client.post(
            f"{base_url}/api/public/{endpoint}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=data,
        )
        logger.debug(f"Response from {base_url}/api/public/{endpoint}: {response.status_code}")
        if response.status_code < 200 or response.status_code >= 300:
            logger.debug("Text: " + response.text)
            logger.debug("Data: " + json.dumps(data, indent=2))
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(f"Request error occurred: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


async def worker(queue: Queue, base_url: str, api_key: str):
    logger = logging.getLogger(__name__ + ".worker")
    logger.debug(f"Starting worker with base_url: {base_url}")
    requests_in_progress: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    tasks = []

    async with httpx.AsyncClient(http2=True) as client:
        while True:
            item: Dict = await asyncio.to_thread(queue.get)
            logger.debug(f"Got item from queue: {item}")

            if item is None:
                break

            id = item.get("completion_id", item.get("trace_id"))
            endpoint = item.pop("endpoint")
            data = item.pop("data")

            async with requests_in_progress[id]:
                logger.debug(f"Submitting {data} to Baserun at {base_url}/api/public/{endpoint}")
                try:
                    task = asyncio.create_task(post_and_log_response(base_url, api_key, endpoint, data, client))
                    logger.debug(f"Task {task} created")
                    tasks.append(task)
                    task.add_done_callback(lambda t: tasks.remove(t))
                    task.add_done_callback(lambda t: logger.debug(f"Task {t} finished"))
                except Exception as e:
                    logger.warning(f"Failed to submit {endpoint}/{id} to Baserun: {e}")

    logger.debug(f"Waiting for {len(tasks)} tasks to finish")
    await asyncio.gather(*tasks, return_exceptions=True)
    logger.debug("Exiting worker")
    loop = asyncio.get_running_loop()
    loop.stop()


def run_worker(queue: Queue, base_url: str, api_key: str):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.ensure_future(worker(queue, base_url, api_key))
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
        try:
            loop = asyncio.get_event_loop()
            if tasks:
                loop.run_until_complete(asyncio.gather(*tasks))
        except RuntimeError:
            # If there's no running loop (i.e. it's already been stopped), just ignore this exception
            pass

        # Terminate the process
        exporter_process.terminate()
        exporter_process.join()

    # Close and join the queue
    if exporter_queue is not None and not getattr(exporter_queue, "_closed"):
        exporter_queue.close()
        exporter_queue.join_thread()
