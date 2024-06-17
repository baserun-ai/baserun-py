import atexit
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from baserun.models.dataset import DatasetMetadata, DatasetVersionMetadata
from baserun.worker import exporter_queue, start_worker, stop_worker

if TYPE_CHECKING:
    from baserun.wrappers.generic import GenericClient, GenericCompletion

logger = logging.getLogger(__name__)


class ApiClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.environment = os.getenv("BASERUN_ENVIRONMENT", os.getenv("ENVIRONMENT", "production"))
        self.base_url = base_url or os.getenv("BASERUN_API_URL") or "https://app.baserun.ai"
        self.api_key = api_key or os.getenv("BASERUN_API_KEY") or ""

        atexit.register(self.exit_handler)

        start_worker(self.base_url, self.api_key)

    def exit_handler(self, *args) -> None:
        stop_worker()

    async def list_datasets(self) -> List[DatasetMetadata]:
        data = await get(self.base_url, self.api_key, "datasets", httpx.AsyncClient())
        datasets = data.get("datasets", [])
        metadata = []
        if isinstance(datasets, list):
            for dataset in datasets:
                versions = [
                    DatasetVersionMetadata(id=version.get("id"), creation_timestamp=version.get("creationTimestamp"))
                    for version in dataset.get("versions", [])
                ]
                metadata.append(
                    DatasetMetadata(
                        id=dataset.get("id", ""),
                        name=dataset.get("name", ""),
                        length=dataset.get("length", 0),
                        features=dataset.get("schema", []),
                        creation_timestamp=dataset.get("creation_timestamp", None),
                        created_by=dataset.get("created_by_clerk_user_id", ""),
                        versions=versions,
                    )
                )

        return metadata

    async def get_dataset(self, id: str, version: Optional[str] = None) -> Dict[str, Any]:
        data = await get(
            self.base_url,
            self.api_key,
            f"datasets/{id}?version={version}" if version else f"datasets/{id}",
            httpx.AsyncClient(),
        )
        return data.get("dataset", {})

    def submit_completion(self, completion: "GenericCompletion"):
        data = self._completion_data(completion)
        logger.debug(f"Submitting completion:\n{json.dumps(data, indent=2)}")
        post("completions", data)

    def submit_dataset(
        self,
        data: Any,
        name: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fingerprint: Optional[str] = None,
    ):
        data = {
            "name": name,
            "data": data,
            "version": version,
            "fingerprint": fingerprint,
            "metadata": metadata,
        }
        logger.debug(f"Submitting dataset:\n{json.dumps(data, indent=2)}")
        post("datasets", data)

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
            "model": completion.model or config_params.get("model"),
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
            "metadata": completion.metadata,
        }

    def _trace_data(self, client: "GenericClient") -> Dict[str, Any]:
        return {
            "id": client.trace_id,
            "name": client.name,
            "tags": [tag.model_dump() for tag in client.tags],
            "evals": [e.model_dump() for e in client.evals if e.score is not None],
            "environment": self.environment,
            "output": client.output,
            "start_timestamp": client.start_timestamp.isoformat() if client.start_timestamp else None,
            "end_timestamp": client.end_timestamp.isoformat() if client.end_timestamp else None,
            "error": client.error,
            "end_user_identifier": client.user,
            "end_user_session_identifier": client.session,
            "metadata": client.metadata,
            "experiment": client.experiment.model_dump() if client.experiment else None,
        }


def post(endpoint: str, data: Dict[str, Any]):
    exporter_queue.put({"endpoint": endpoint, "data": data})


async def get(base_url: str, api_key: str, endpoint: str, client: httpx.AsyncClient) -> Dict[str, Any]:
    logger = logging.getLogger(__name__ + ".getter")
    logger.debug(f"Getting from {base_url}/api/public/{endpoint}")
    response = await client.get(
        f"{base_url}/api/public/{endpoint}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    logger.debug(f"Response from {base_url}/api/public/{endpoint}: {response.status_code}")
    if response.status_code < 200 or response.status_code >= 300:
        raise Exception(f"Failed to fetch data {endpoint} from Baserun: {response.text}")

    return response.json()
