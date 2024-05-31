import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from baserun.api import post, start_worker

logger = logging.getLogger(__name__)


class BaserunCallbackManager(BaseCallbackManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, handlers=[BaserunCallbackHandler()], **kwargs)


class BaserunCallbackHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    start_timestamp: Optional[datetime] = None
    first_token_timestamp: Optional[datetime] = None
    input_messages: List[Dict[str, Any]]

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> None:
        """Run when Chat Model starts running."""
        start_worker(
            os.getenv("BASERUN_API_URL") or "https://app.baserun.ai",
            os.getenv("BASERUN_API_KEY") or "",
        )
        self.start_timestamp = datetime.now()
        self.input_messages = [
            {
                "content": message.content,
                "role": "user" if message.type == "human" else message.type,
            }
            for message in messages[0]
        ]

    async def on_llm_new_token(self, *args, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        if not self.first_token_timestamp:
            self.first_token_timestamp = datetime.now()

    def on_llm_end(self, response: LLMResult, run_id: UUID, **kwargs: Any) -> None:
        """Run when LLM ends running."""
        environment = os.getenv("BASERUN_ENVIRONMENT", os.getenv("ENVIRONMENT", "production"))

        start_timestamp = self.start_timestamp or datetime.now()
        first_token_timestamp = self.first_token_timestamp if self.first_token_timestamp else datetime.now()
        end_timestamp = datetime.now()
        output = response.llm_output or {}
        usage = output.pop("token_usage", {})
        completion_id = uuid4()
        trace_id = uuid4()
        name = "Langchain Completion"

        choices = [
            {
                "message": {
                    "content": message.text,
                    "role": "assistant",
                },
                "index": i,
                "finish_reason": (message.generation_info or {}).get("finish_reason") or "stop",
            }
            for i, message in enumerate(response.generations[0])
        ]
        trace_output = response.generations[0][-1].text

        data = {
            "id": str(run_id),
            "name": name,
            "completion_id": str(completion_id),
            "model": output.get("model_name"),
            "trace_id": str(trace_id),
            "tags": [],
            "tool_results": [],
            "evals": [],
            "input_messages": self.input_messages,
            "choices": choices,
            "usage": usage,
            "config_params": output,
            "start_timestamp": start_timestamp.isoformat(),
            "end_timestamp": end_timestamp.isoformat(),
            "first_token_timestamp": first_token_timestamp.isoformat(),
            "request_id": kwargs.get("run_id"),
            "environment": environment,
            "trace": {
                "id": str(trace_id),
                "name": "Langchain LLM",
                "tags": [],
                "evals": [],
                "environment": environment,
                "output": trace_output,
                "start_timestamp": start_timestamp.isoformat(),
                "end_timestamp": end_timestamp.isoformat() if end_timestamp else None,
                "error": "",  # TODO?
                "metadata": {},
            },
        }
        logger.debug(f"Submitting streamed completion:\n{json.dumps(data, indent=2)}")
        post("completions", data)
