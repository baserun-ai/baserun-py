import json
from typing import Any, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span, get_current_span

import baserun
from baserun import Baserun


class BaserunCallbackHandler(BaseCallbackHandler):
    """Base callback handler that can be used to handle callbacks from langchain."""

    spans: dict[str, Span] = None

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any) -> Any:
        """Run when Chat Model starts running."""
        if self.spans is None:
            self.spans = {}

        span = self.spans.get(kwargs.get("run_id"))
        if not span:
            with baserun.start_trace("chat_model_start", end_on_exit=False):
                self.spans[kwargs.get("run_id")] = get_current_span()
                yield

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        run = Baserun.current_run()
        outputs = []
        for choice in response.generations:
            for message in choice:
                if message.text:
                    outputs.append(message.text)
                else:
                    outputs.append(message.additional_kwargs.get("tool_calls"))

        run.result = outputs[0] if len(outputs) == 1 else json.dumps(outputs)

        if kwargs.get("parent_run_id"):
            # This is executing in the context of an agent, so don't end the trace
            return

        span = self.spans.get(kwargs.get("run_id"))
        if span and span.is_recording():
            span.end()
            Baserun._finish_run(run, span)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when LLM errors."""

    def on_chain_start(self, serialized: dict[str, Any], inputs: dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain starts running."""

        if kwargs.get("parent_run_id"):
            # This is executing in the context of an agent, so don't start a trace
            return

        if self.spans is None:
            self.spans = {}

        with baserun.start_trace("on_chain_start", end_on_exit=False) as run:
            self.spans[kwargs.get("run_id")] = get_current_span()
            yield

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> Any:
        """Run when chain ends running."""
        run = Baserun.current_run()
        if text_output := outputs.get("text", outputs.get("output")):
            run.result = text_output

        if kwargs.get("parent_run_id"):
            # This is executing in the context of an agent, so don't end the trace
            return

        span = self.spans.get(kwargs.get("run_id"))
        if span and span.is_recording():
            span.end()

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when chain errors."""

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        """Run when tool starts running."""

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Run when tool ends running."""

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Run when tool errors."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Run on agent action."""
        if self.spans is None:
            self.spans = {}

        with baserun.start_trace("on_agent_action", end_on_exit=False):
            self.spans[kwargs.get("run_id")] = get_current_span()
            yield

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Run on agent end."""
        run = Baserun.current_run()
        outputs = []
        for message in finish.messages:
            if message.content:
                outputs.append(message.content)
            else:
                outputs.append(message.additional_kwargs.get("tool_calls"))

        run.result = outputs[0] if len(outputs) == 1 else json.dumps(outputs)

        if finish.log:
            baserun.log("Agent finish", finish.log)

        span = self.spans.get(kwargs.get("run_id"))
        if span and span.is_recording():
            span.end()
