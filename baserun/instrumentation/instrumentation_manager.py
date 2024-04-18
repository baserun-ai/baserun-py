import logging
from typing import TYPE_CHECKING, Dict, Optional, Type

from baserun.helpers import BaserunProvider
from baserun.instrumentation.instrumentation import Instrumentation
from baserun.instrumentation.openai import OpenAIInstrumentation

if TYPE_CHECKING:
    from baserun.baserun import _Baserun

logger = logging.getLogger(__name__)


class InstrumentationManager:
    instrumentations: Dict[BaserunProvider, Instrumentation] = {}

    instrumentation_classes: Dict[BaserunProvider, Optional[Type[Instrumentation]]] = {
        BaserunProvider.ANTHROPIC: None,
        BaserunProvider.OPENAI: OpenAIInstrumentation,
        BaserunProvider.GOOGLE: None,
        BaserunProvider.LLAMA_INDEX: None,
    }

    try:
        from baserun.instrumentation.google import GoogleInstrumentation

        instrumentation_classes[BaserunProvider.GOOGLE] = GoogleInstrumentation
    except ImportError:
        logger.debug("google.ai.generativelanguage not available")

    try:
        from baserun.instrumentation.anthropic import AnthropicInstrumentation

        instrumentation_classes[BaserunProvider.ANTHROPIC] = AnthropicInstrumentation
    except ImportError:
        logger.debug("anthropic not available")

    try:
        from baserun.instrumentation.llamaindex import LLamaIndexInstrumentation

        instrumentation_classes[BaserunProvider.LLAMA_INDEX] = LLamaIndexInstrumentation
    except ImportError:
        logger.debug("llama_index not available")

    @classmethod
    def instrument_all(cls, baserun_inst: "_Baserun"):
        for k, v in cls.instrumentation_classes.items():
            if k not in cls.instrumentations and v:
                inst = v(baserun_inst)
                cls.instrumentations[k] = inst
                inst.instrument()

    @classmethod
    def uninstrument_all(cls):
        for inst in cls.instrumentations.values():
            inst.uninstrument()
