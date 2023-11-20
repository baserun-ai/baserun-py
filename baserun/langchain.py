from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

from baserun import Baserun
from baserun.constants import UNTRACED_SPAN_PARENT_NAME
from baserun.v1.baserun_pb2 import Run


class BaserunLangchainCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.run = Baserun.get_or_create_current_run(
            name=UNTRACED_SPAN_PARENT_NAME,
            trace_type=Run.RunType.RUN_TYPE_PRODUCTION,
        )
        super().__init__()

    def on_chain_start(self, serialized: dict[str, any], inputs: dict[str, any], **kwargs: any) -> any:
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: any) -> any:
        pass
