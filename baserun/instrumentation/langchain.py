import json
from typing import Any, Dict, List, Optional

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.trace import Span, get_current_span

import baserun
from baserun import Baserun


class BaserunCallbackHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    spans: Optional[Dict[str, Span]] = None

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        """Run when Chat Model starts running."""
        if self.spans is None:
            self.spans = {}

        run_id = str(kwargs.get("run_id"))
        span = self.spans.get(run_id)
        if not span:
            with baserun.start_trace("chat_model_start", end_on_exit=False):
                self.spans[run_id] = get_current_span()
                yield

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        if self.spans is None:
            self.spans = {}

        run = Baserun.current_run()
        if not run:
            return

        outputs: List[Any] = []
        for choice in response.generations:
            for generation in choice:
                if generation.text:
                    outputs.append(generation.text)
                elif isinstance(generation, ChatGeneration):
                    outputs.append(generation.message.additional_kwargs.get("tool_calls"))
                else:
                    outputs.append("")

        run.result = outputs[0] if len(outputs) == 1 else json.dumps(outputs)

        if kwargs.get("parent_run_id"):
            # This is executing in the context of an agent, so don't end the trace
            return

        span = self.spans.get(str(kwargs.get("run_id")))
        if span and span.is_recording():
            span.end()
            Baserun._finish_run(run)

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""

        if kwargs.get("parent_run_id"):
            # This is executing in the context of an agent, so don't start a trace
            return

        if self.spans is None:
            self.spans = {}

        with baserun.start_trace("on_chain_start", end_on_exit=False):
            self.spans[str(kwargs.get("run_id"))] = get_current_span()
            yield

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        if self.spans is None:
            self.spans = {}

        run = Baserun.current_run()
        if not run:
            return

        if text_output := outputs.get("text", outputs.get("output")):
            run.result = text_output

        if kwargs.get("parent_run_id"):
            # This is executing in the context of an agent, so don't end the trace
            return

        span = self.spans.get(str(kwargs.get("run_id")))
        if span and span.is_recording():
            span.end()

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        if self.spans is None:
            self.spans = {}

        with baserun.start_trace("on_agent_action", end_on_exit=False):
            self.spans[str(kwargs.get("run_id"))] = get_current_span()
            yield

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        if self.spans is None:
            self.spans = {}

        run = Baserun.current_run()
        if not run:
            return

        run.result = json.dumps(finish.return_values)

        if finish.log:
            baserun.log("Agent finish", finish.log)

        span = self.spans.get(str(kwargs.get("run_id")))
        if span and span.is_recording():
            span.end()
